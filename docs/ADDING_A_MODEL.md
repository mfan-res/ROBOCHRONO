# Adding a model

Two paths, depending on how new your model really is. Most additions are the
first path and touch no code at all.

## Which path are you on?

**Path 1 — a configuration file only.** Your model is served the way an
already-supported family is served. One JSON file in `configs/models/` makes
it part of the roster; the files in that directory *are* the model list,
there is no separate registry to edit. This covers more than it sounds like:

- another size of a supported local family — `qwen3-vl-32b-instruct.json` and
  `qwen3-vl-2b-instruct.json` differ only in `weights` and `name`;
- a checkpoint that *ships as* a supported architecture, whatever its brand —
  `mimo-vl-7b-rl.json` runs through the `qwen3_vl` adapter because the
  checkpoint loads as a Qwen-family model; the adapter keys on the processor
  convention, not the marketing name;
- any endpoint that speaks the OpenAI chat/completions dialect —
  `qwen3-8-max.json` and `doubao-seed-2-0-lite.json` are both
  `"adapter": "openai_compat"` with different URLs and quirk flags;
- the same model under different generation settings — `qwen3-8-max.json`
  and `qwen3-8-max-thinking.json` are one endpoint, two configurations,
  differing only in the declared thinking state.

**Path 2 — a new adapter.** The model needs loading or calling code that no
existing adapter provides: a new preprocessing pipeline, a new request
dialect, a chat method nothing else uses. Then you add one file under
`robochrono/adapters/` *and* a configuration file. See "Writing an adapter"
below.

The current adapter roster, from `robochrono/adapters/__init__.py`:
`qwen3_vl`, `rynnbrain`, `internvl`, `cosmos3_edge`, `openai_compat`,
`gemini`, and `replay` (a debugging adapter that answers from a recorded
table — see below).

## The model configuration file

One JSON file per model. The directory decides the kind: files under
`configs/models/local/` are local checkpoints, files under
`configs/models/api/` are served endpoints. **The filename (minus `.json`)
is the model's slug** — what you pass to `--models` and what names its
result directories.

Required at the top level (loading fails without them):

| Field | Meaning |
| --- | --- |
| `name` | Display name, must be unique across the roster. |
| `adapter` | One of the registered adapter names above. |
| `weights` | Local: path to the weights directory, relative to the repository root. API: an informal `api:<provider>/<id>` string — recorded in results, never dereferenced. |
| `environment` | Which Python environment runs it — a key of `configs/environments.json` (`transformers-4x` or `transformers-5x`). Preflight checks the pin; the orchestrator starts the model under that interpreter automatically. |

Optional blocks:

| Block | Meaning |
| --- | --- |
| `official` | What the model's documentation states, each entry graded (see below). At minimum declare `official.transformers` — write `"source": "none"` if the card says nothing; the consistency test fails on a missing declaration, not on an honest "the docs do not say". |
| `generation` | Per-model deviations from the protocol, graded. A model may **raise** `max_new_tokens`, never lower it, and declares its real `thinking` state when it cannot match the protocol's (`"disabled"` is the protocol default; use `"enabled"`, `"always_on"` or `"measured"` with a note when the switch does not exist). |
| `media` | Adapter-specific preprocessing knobs (e.g. the InternVL tiling budgets `max_image_tiles` / `max_video_tiles`). Only meaningful to adapters that read them. |
| `api` | API models only: endpoint and dialect. Keys never go in here — `key_env` names the environment variable holding the key. |
| `resources` | What it takes to run: `gpus_per_worker` for models one card cannot hold (the 122B declares 4) — a property of the weights, declared on the model so small models never inherit a big model's parallelism. |

Keys starting with `_` anywhere are comments and are ignored by the loader —
use `_note` liberally; the shipped configurations do.

### Template: local model

```jsonc
{
  "name": "My-Model-7B",                     // unique display name
  "adapter": "qwen3_vl",                     // a registered adapter
  "weights": "models/My-Model-7B",           // directory under the repo root
  "environment": "transformers-4x",          // or transformers-5x
  "official": {
    "transformers": {
      "value": ">=4.50.0",                   // what the card pins, or null
      "source": "L1",                        // see evidence grades below
      "note": "Stated in the model card's requirements section."
    }
  },
  "generation": {                            // only if deviating from protocol
    "max_new_tokens": {
      "value": 8192,                         // raise only, never lower
      "source": "L1",
      "note": "The card recommends 8k for long reasoning output."
    }
  },
  "resources": {
    "gpus_per_worker": 1                     // omit if one card suffices
  }
}
```

### Template: API model

```jsonc
{
  "name": "My-Served-Model",
  "adapter": "openai_compat",                // or "gemini"
  "weights": "api:provider/my-served-model", // informal id, recorded only
  "environment": "transformers-4x",          // interpreter for the light API stack
  "official": {
    "transformers": { "value": null, "source": "none",
                      "note": "Served remotely; no local stack." }
  },
  "api": {
    "url": "https://api.example.com/v1/chat/completions",
    "model": "my-served-model",              // the served model id sent per request
    "key_env": "MY_PROVIDER_API_KEY",        // env var holding the key; never the key
    "media_url_format": "data_url",          // inline media as data: URLs
    "max_request_bytes": 10000000,           // server body cap; oversized media is
                                             // shrunk to fit (spatially, never temporally)
    "min_video_seconds": 2.0,                // some servers refuse shorter clips
    "proxy": "http://proxy.example:8080"     // optional; only if the endpoint needs one
  }
}
```

Dialect quirks the `openai_compat` adapter understands, all optional:
`thinking_param` (the field name for the provider's thinking toggle),
`send_thinking` (send the toggle even when disabled), `generation_config`
(extra fields merged into the request — the Gemini adapter uses this for
`thinkingConfig`).

## Evidence grades

Every value in `official` and `generation` carries a `source` grade —
`configs/README.md` defines the scale. The short version: **L1–L4** mean the
model's own documentation says so (recommendation, example code, function
default, companion-library default); **L5** means inherited from the same
architecture; **L6** means our own decision; **`none`** means the
documentation is silent. Pick the grade honestly — the point is that a
reader can tell a vendor-recommended number from a choice we made. A missing
`source` fails `tests/test_config_consistency.py`; a wrong-but-honest one
does not, because nothing can machine-check honesty.

## Writing an adapter (path 2)

An adapter is one file under `robochrono/adapters/` exposing:

```python
def build(model, protocol, runtime=None) -> Adapter: ...
```

Subclass `Adapter` from `adapters/base.py`. The contract:

- `__init__` must be **cheap and import no inference stack** — construction
  happens in every environment, including ones without torch. Import heavy
  libraries lazily, inside methods. `tests/test_orchestrator_is_light.py`
  enforces this.
- `load()` does the expensive part (weights, processors); called only inside
  worker processes. A no-op for API adapters.
- `call(parts, *, frames, key="") -> AdapterResult` runs one model call.
  `parts` is the dimension-assembled list of `{"type": "text"|"image"|"video",
  ...}` parts with media paths already absolute; `frames` is the protocol's
  frame-sampling spec for the dimension. Return `AdapterResult(text=...)`,
  filling `frames_used`, `usage` and `media_transforms` with whatever is
  observable — those land in the result rows and cannot be reconstructed
  later.
- Respect the resolved generation settings the base class computed:
  `self.temperature`, `self.max_new_tokens`, `self.system_prompt`,
  `self.thinking`. The helpers in `base.py` (`generation_kwargs`,
  `post_with_retries`, `data_url`, …) cover the common ground.

Register the module name in the `_ADAPTERS` tuple in
`robochrono/adapters/__init__.py` — that is the whole registration.

Then pin your request structure in `tests/test_adapter_payloads.py` the way
the existing adapters do: build the adapter without any inference stack
installed and assert on the payload it would send. That test is what catches
a silently dropped field two refactors later.

### Debugging with the replay adapter

The `replay` adapter answers from a recorded table instead of a model, which
lets you exercise the whole pipeline — loading, rendering, calling, parsing,
scoring, storage, resumption — on a machine with no GPU and no keys. Point a
configuration at it (`"adapter": "replay"`, with `api.table` naming a JSON
file mapping unit keys to reply text — a *unit key* is the question's `id`
from the qa file; docs/DATA_FORMAT.md defines the format) and run `eval`
normally. Practical notes:

- put the file under `configs/models/local/` — the directory decides the
  model's kind, and `local` is the one whose checks a replay table can
  satisfy;
- set `weights` to any existing directory (`"configs"` works) — the replay
  adapter never reads it, and preflight's existence check passes;
- the `api.table` file must exist: the adapter reads it at construction, so
  a stale path breaks every command that builds the roster;
- the log will still say `pool: N pending unit(s) on M card(s)`: workers
  are laid out on cards as for any local model, but the replay adapter
  never allocates GPU memory — the line is bookkeeping, not occupancy;
- a replay configuration is a **temporary debugging file** — the test suite
  handles one being present, but every file in `configs/models/` is part of
  the roster, so remove debug entries before committing.

`tests/test_cli.py` builds exactly such a configuration in a temporary
directory (via `--models-dir`) if you want the pipes-and-all example.

## End-to-end verification checklist

After adding a model, in order:

```bash
# 1. Configuration, environment, weights/keys, data — before anything runs
.venvs/tf4/bin/python -m robochrono preflight --models <slug>

# 2. What a smoke run would cost; runs nothing
.venvs/tf4/bin/python -m robochrono eval --suite smoke --models <slug> \
    --limit-items 2 --dry-run

# 3. The same, for real (drop --dry-run; add --gpus N for a local model)
.venvs/tf4/bin/python -m robochrono eval --suite smoke --models <slug> \
    --limit-items 2

# 4. The tests that gate a configuration
.venvs/tf4/bin/python tests/test_config_consistency.py
.venvs/tf4/bin/python tests/test_adapter_payloads.py   # if you wrote an adapter
```

Every preflight line must be OK or SKIP. Expect the smoke run to end with a
`report.md` line and `errors=0` in its summaries.

Step 4's consistency test also cross-checks the suite files against the
local scenario pool, so it wants every pinned scenario's *manifest* present —
`tools/download_data.py --all --qa-only` (~50 MB) satisfies that without any
media. On a partial pool, failures naming scenarios (`… scenarios exist in
the pool`, the 34,713 question count) are about your download, not your
model; the checks that gate a model configuration are the ones naming your
slug. A new local model's first
real run is best started with `--limit-items` before committing to the full
34,713 questions.
