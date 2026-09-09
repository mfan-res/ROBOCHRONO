#!/usr/bin/env python3
# coding: utf-8
"""The API path against a local stub endpoint: no network, no key, no cost.

A thread-local HTTP server speaks just enough chat/completions for the full
loop — engine → adapter → HTTP → parse → store — to run on real questions.
What it pins is alignment: the API path must show a model byte-identical
prompts to the local path, send the declared dialect and generation fields,
survive a 429, and record what a real run would need for the post-mortem.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono import adapters, dimensions, engine  # noqa: E402
from robochrono.config.models import load_model  # noqa: E402
from robochrono.config.protocol import load_protocol  # noqa: E402
from robochrono.dataset.loader import load_questions  # noqa: E402
from robochrono.dataset.render import load_question_bank  # noqa: E402
from robochrono.results.store import ResultStore  # noqa: E402

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:56} {detail}")
    if not passed:
        failures.append(name)


DATA = ROOT / "scenarios"
if not (DATA / "make_tea_tianji" / "manifest.json").exists():
    print("skip: dataset not present")
    sys.exit(0)

# ── the stub endpoint ─────────────────────────────────────────────────────
REQUESTS: list[dict] = []
ANSWERS: dict[str, str] = {}          # substring of prompt -> reply text
FAIL_FIRST = {"remaining": 0}


class Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append(body)
        if FAIL_FIRST["remaining"] > 0:
            FAIL_FIRST["remaining"] -= 1
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        prompt_texts = " ".join(
            p.get("text", "") for m in body.get("messages", [])
            for p in (m["content"] if isinstance(m["content"], list) else []))
        reply = next((r for key, r in ANSWERS.items() if key in prompt_texts),
                     '{"choice": "A"}')
        payload = {"choices": [{"message": {"content": reply}}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):  # keep the test output clean
        pass


server = HTTPServer(("127.0.0.1", 0), Stub)
threading.Thread(target=server.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"

PROTOCOL = load_protocol(ROOT / "configs/protocol.json")
base = load_model(ROOT / "configs/models/api/qwen3-8-max.json")
MODEL = dataclasses.replace(base, api={**base.api, "url": URL, "key_env": None,
                                       "max_request_bytes": 200_000_000})

bank = load_question_bank(DATA)
items = load_questions(DATA, "make_tea_tianji", "current_action", bank=bank)[:2]
for q in items:
    # keyed on the full rendered question: the shared template makes any
    # fixed-length prefix identical across questions
    ANSWERS[q["question"]] = json.dumps({"choice": q["answer"]})
dim = dimensions.build("current_action", strip_reasoning=PROTOCOL.strip_reasoning)
frames = PROTOCOL.frames_for("current_action")

print("1. the full loop runs against the stub")
store = ResultStore(Path(tempfile.mkdtemp()) / "api" / "s" / "current_action.jsonl")
summary = engine.run(dim, items, adapters.build(MODEL, PROTOCOL), store,
                     data_root=DATA / "make_tea_tianji", frames=frames)
check("both questions answered", summary["answered"] == 2)
check("perfect score against scripted answers", summary["accuracy"] == 1.0)
rows = list(store.rows())
check("usage recorded per row",
      all(r.get("usage", {}).get("completion_tokens") == 5 for r in rows))

print("2. the request carries what the protocol declares")
body = REQUESTS[0]
check("system prompt first",
      body["messages"][0] == {"role": "system", "content": PROTOCOL.system_prompt})
check("temperature and max_tokens sent",
      body["temperature"] == 0.0 and body["max_tokens"] == 4096)
check("thinking dialect sent", body.get("enable_thinking") is False)
user = body["messages"][1]["content"]
kinds = [p["type"] for p in user]
check("video then question, as the dimension orders it",
      kinds == ["video_url", "text"])
check("media is inlined as a data url",
      user[0]["video_url"]["url"].startswith("data:video/mp4;base64,"))

print("3. prompts are byte-identical to the local path")
local_store = ResultStore(Path(tempfile.mkdtemp()) / "l" / "s" / "current_action.jsonl")
replay_model = dataclasses.replace(MODEL, adapter="replay")
table = {str(q["id"]): json.dumps({"choice": q["answer"]}) for q in items}
engine.run(dim, items, adapters.build(replay_model, PROTOCOL, {"replay_table": table}),
           local_store, data_root=DATA / "make_tea_tianji", frames=frames)
api_prompts = {r["id"]: r["prompt"] for r in store.rows()}
local_prompts = {r["id"]: r["prompt"] for r in local_store.rows()}
check("same ids", set(api_prompts) == set(local_prompts))
check("byte-identical prompts", api_prompts == local_prompts)
sent_text = [p["text"] for p in user if p["type"] == "text"][0]
check("what was sent is what was recorded",
      sent_text in api_prompts.values())

print("4. a 429 is retried, not returned")
FAIL_FIRST["remaining"] = 1
before = len(REQUESTS)
store2 = ResultStore(Path(tempfile.mkdtemp()) / "a" / "s" / "current_action.jsonl")
summary = engine.run(dim, items[:1], adapters.build(MODEL, PROTOCOL), store2,
                     data_root=DATA / "make_tea_tianji", frames=frames)
check("answered after retry", summary["answered"] == 1 and summary["errors"] == 0)
check("exactly one extra request", len(REQUESTS) - before == 2)

print("5. the request budget triggers media preparation")
tight = dataclasses.replace(MODEL, api={**MODEL.api, "max_request_bytes": 400_000})
store3 = ResultStore(Path(tempfile.mkdtemp()) / "t" / "s" / "current_action.jsonl")
summary = engine.run(dim, items[:1], adapters.build(
    tight, PROTOCOL, {"media_cache_dir": tempfile.mkdtemp()}), store3,
    data_root=DATA / "make_tea_tianji", frames=frames)
row = next(store3.rows())
check("transform recorded on the row",
      bool(row.get("media_transforms")), str(row.get("media_transforms"))[:60])
check("the shrunk video fits the budget",
      len(REQUESTS[-1]["messages"][1]["content"][0]["video_url"]["url"]) < 400_000)

server.shutdown()
print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
