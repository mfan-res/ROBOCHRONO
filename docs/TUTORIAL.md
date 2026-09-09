# Running RoboChrono on a fresh machine

Two paths through this guide. **Part 1** runs local models on a GPU cluster;
**Part 2** runs API-served models and needs no GPU at all. Both start from a
machine with nothing on it.

Wherever a command mentions a model list, substitute the models you want to
evaluate. The worked examples below use one large and one small local model,
and the five configured API models:

- **GPU example**: `qwen3-vl-2b-instruct`, `rynnbrain1-1-122b-a10b`
- **API example**: `qwen3-8-max`, `qwen3-8-max-thinking`, `gemini-3-6-flash`,
  `doubao-seed-2-0-lite`, `qwen3-vl-235b-a22b-api`

---

## Part 1 — GPU cluster

### 1.0 What you need

| Requirement | Why |
| --- | --- |
| Linux, NVIDIA driver ≥ CUDA 12.4 | the pinned torch build is cu124 |
| `git`, `ffmpeg` (with `ffprobe`) | clone; media probing and shrinking |
| Disk: up to ~74 GB for the full dataset (a single scenario is 0.3–14 GB; per-scenario sizes in docs/DATA_FORMAT.md), ~12 GB for the two Python environments, plus your models' weights (4 GB for the 2B example; the largest models run to hundreds of GB — see the table in 1.4; results are megabytes) | see 1.3 and 1.4 |
| Access to the dataset (see 1.3) | the pool of scenario directories |
| [`uv`](https://docs.astral.sh/uv/) | builds the two Python environments |

Install `uv` if missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.1 Clone

```bash
git clone <repository-url> robochrono   # the URL of the repository this
cd robochrono                           # file came from
```

Everything below happens inside this directory. Data, weights and results all
live under it — nothing is installed system-wide.

### 1.2 Build the two environments

Two are required, not one: some model families only load under
transformers 4.x, others only under 5.x. The evaluator switches between them
automatically at run time; you only build them.

```bash
uv python install 3.11
UV_PROJECT_ENVIRONMENT=.venvs/tf4 uv sync --extra tf4 --python 3.11
UV_PROJECT_ENVIRONMENT=.venvs/tf5 uv sync --extra tf5 --python 3.11
```

Run them one after the other, not in parallel: two `uv` processes contend for
the same cache lock and the second gives up after five minutes. A cold first
build is dominated by downloading ~2.5 GB of torch per environment — think
minutes to tens of minutes depending on your link to PyPI; a warm-cache
rebuild is a couple of minutes.

If the default index is slow from where you are, the download dominates the
build. Point `uv` at a mirror near you once, in `~/.config/uv/uv.toml`:

```toml
[[index]]
url = "https://<your-nearest-pypi-mirror>/simple"
default = true
```

**Side effect to know about**: with a default mirror configured, `uv sync`
rewrites `uv.lock` so every source URL points at the mirror — hundreds of
changed lines that are easy to commit by accident. Restore the file after
building (`git checkout uv.lock`) and never commit a mirror-rewritten lock.
While the rewritten lock sits in your tree, every run you record carries
`dirty: true` in its code fingerprint (README, *Reading the results*).

Which models need which environment is declared in
`configs/models/local/*.json` (`environment` field); you never activate
either by hand.

### 1.3 Get the dataset

The dataset is a pool of self-contained scenario directories under
`scenarios/` — `scenarios/<scenario_id>/{manifest.json, qa/, media/}`.
Partial setups are per scenario: fetch only the scenarios you need, and
inside a scenario `media/episodes/` may be skipped unless you run
`action_time`. The data is public in two Hugging Face repositories
(`gimai/RoboChrono-GIM` and `gimai/RoboChrono-Tianji`) — no Hugging Face
account or token is needed; the download
script maps scenarios to repositories on its own, unpacks the media
archives, and finishes with the validation sweep:

```bash
.venvs/tf4/bin/python tools/download_data.py --all        # everything, ~74 GB
.venvs/tf4/bin/python tools/download_data.py make_tea_tianji cap_pen_gim
```

If you fetched by hand instead, validate afterwards:

```bash
.venvs/tf4/bin/python -m robochrono validate-data
```

The command must end with `all checks passed` — the full pool is 39
scenarios and 34,713 questions, every content hash re-verified, no media
missing. If it does not, stop and re-fetch; nothing downstream can repair
missing data.

### 1.4 Download your models' weights

Weights go under `models/<name>` — the exact directory names matter, they
are what `configs/models/local/*.json` declares. The small model the worked
examples use:

```bash
mkdir -p models
.venvs/tf4/bin/hf download Qwen/Qwen3-VL-2B-Instruct \
    --local-dir models/Qwen3-VL-2B-Instruct        # ~4 GB
```

(`hf` ships with the tf4 environment, so the venv path works without any
extra install — the same convention as every other command in this guide.)

Every configured local model, slug → repository → directory. The same
`hf download <repository> --local-dir <directory>` works for every row; the
directory column, not the repository name, is what the configuration
declares — mind the underscores in two of the RynnBrain1.1 directories.
Sizes are approximate download sizes:

| Model slug | Hugging Face repository | `--local-dir` | Size |
| --- | --- | --- | ---: |
| `cosmos3-edge-2b` | `nvidia/Cosmos3-Edge` | `models/Cosmos3-Edge` | ~9 GB |
| `cosmos-reason2-2b` | `nvidia/Cosmos-Reason2-2B` | `models/Cosmos-Reason2-2B` | ~5 GB |
| `internvl3-2b` | `OpenGVLab/InternVL3-2B` | `models/InternVL3-2B` | ~4 GB |
| `mimo-vl-7b-rl` | `XiaomiMiMo/MiMo-VL-7B-RL` | `models/MiMo-VL-7B-RL` | ~16 GB |
| `qwen3-vl-2b-instruct` | `Qwen/Qwen3-VL-2B-Instruct` | `models/Qwen3-VL-2B-Instruct` | ~4 GB |
| `qwen3-vl-8b-instruct` | `Qwen/Qwen3-VL-8B-Instruct` | `models/Qwen3-VL-8B-Instruct` | ~17 GB |
| `qwen3-vl-8b-thinking` | `Qwen/Qwen3-VL-8B-Thinking` | `models/Qwen3-VL-8B-Thinking` | ~17 GB |
| `qwen3-vl-30b-a3b-instruct` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | `models/Qwen3-VL-30B-A3B-Instruct` | ~62 GB |
| `qwen3-vl-32b-instruct` | `Qwen/Qwen3-VL-32B-Instruct` | `models/Qwen3-VL-32B-Instruct` | ~67 GB |
| `qwen3-vl-235b-a22b-instruct` | `Qwen/Qwen3-VL-235B-A22B-Instruct` | `models/Qwen3-VL-235B-A22B-Instruct` | ~471 GB |
| `rynnbrain-2b` | `Alibaba-DAMO-Academy/RynnBrain-2B` | `models/RynnBrain-2B` | ~5 GB |
| `rynnbrain1-1-2b` | `Alibaba-DAMO-Academy/RynnBrain1.1-2B` | `models/RynnBrain1.1-2B` | ~4 GB |
| `rynnbrain1-1-9b` | `Alibaba-DAMO-Academy/RynnBrain1.1-9B` | `models/RynnBrain1_1-9B` | ~19 GB |
| `rynnbrain1-1-122b-a10b` | `Alibaba-DAMO-Academy/RynnBrain1.1-122B-A10B` | `models/RynnBrain1_1-122B-A10B` | ~245 GB |
| `sensenova-si-1-1-internvl3-2b` | `sensenova/SenseNova-SI-1.1-InternVL3-2B` | `models/SenseNova-SI-1_1-InternVL3-2B` | ~4 GB |

None of these repositories needs manual approval, and all but one need
no Hugging Face account at all. The exception is `nvidia/Cosmos-Reason2-2B`,
which is license-gated with automatic approval — log in once
(`.venvs/tf4/bin/hf auth login`) before downloading that row.

A large model is the same command with more patience — the worked examples'
large model:

```bash
.venvs/tf4/bin/hf download Alibaba-DAMO-Academy/RynnBrain1.1-122B-A10B \
    --local-dir models/RynnBrain1_1-122B-A10B      # ~245 GB: hours, plan for it
```

InternVL-family models (`internvl3-2b`, `sensenova-si-1-1-internvl3-2b`)
need one small edit so their code loads from the local directory instead of
reaching for the hub at run time:

```bash
.venvs/tf4/bin/python - << 'PY'
import json
p = "models/InternVL3-2B/config.json"
cfg = json.load(open(p))
cfg["auto_map"] = {k: v.split("--", 1)[-1] for k, v in cfg["auto_map"].items()}
json.dump(cfg, open(p, "w"), indent=2)
print("patched", p)
PY
```

### 1.5 Preflight

```bash
.venvs/tf4/bin/python -m robochrono preflight \
  --models qwen3-vl-2b-instruct rynnbrain1-1-122b-a10b
```

With no `--suite` this checks against the default `official-v1` (all 39
scenarios); scope it with `--suite smoke` while pieces are still arriving.
Every line must be OK or SKIP before you launch. Each FAIL names what is
missing and, for the common cases, the command that fixes it; on a machine
where data or weights are not in place yet those FAILs are expected — work
through them (§1.3 data, §1.4 weights) and rerun. Do not launch past a
FAIL — the whole point of preflight is that a problem found here costs a
minute, and the same problem found mid-run costs hours. The second Python
environment is probed through its own interpreter when a selected model
needs it; a selection that stays in one environment skips that probe with a
visible SKIP line.

### 1.6 Smoke test (about a minute)

One small slice through the real pipeline before committing to the long run:

```bash
.venvs/tf4/bin/python -m robochrono eval \
  --suite smoke --models qwen3-vl-2b-instruct \
  --scenarios pack_airpods_gim --dimensions current_action \
  --limit-items 8 --gpus 4
```

Expected: a `results/<date>_<fingerprint>/` directory appears, the log ends
with a `report.md` line, and the summary inside shows `errors: 0`. The whole
thing is under a minute, most of it model loading.

`--suite` takes the suite's **name** (`smoke`, `official-v1`), not a
filename — passing `smoke.json` looks up a suite called that and fails
with a message listing what exists.

### 1.7 The full run

```bash
nohup .venvs/tf4/bin/python -m robochrono eval \
  --models qwen3-vl-2b-instruct rynnbrain1-1-122b-a10b \
  --gpus 4 > full_run.log 2>&1 &
```

Notes on what happens:

- The default suite is `official-v1`: all 39 scenarios; expect roughly **30 hours**
  for this two-model example — the 122B is the bulk of it (the 2B alone
  finished the full benchmark in 5.7 hours on eight 24 GB cards).
- Models run one at a time; all four GPUs work on the current model. The
  122B needs all 4 GPUs per copy — that is declared in its configuration,
  nothing to set.
- Every answered question is written to disk immediately. Killing the run
  loses at most the questions in flight.

### 1.8 Monitoring and sharing the machine

```bash
tail -f full_run.log                          # everything
grep -E "pool: |ABORTED|error: " full_run.log # the lines that matter
nvidia-smi                                    # GPUs busy?
```

**Sharing the cards with other jobs?** `--gpus` takes either a count
(`--gpus 4`: the first four cards) or a list of exact card indices
(`--gpus 2,5`: those two cards). Check `nvidia-smi`, pick the free ones,
and pass them as a list; add `--gpus-per-worker 2` when the remaining
memory per card is tight.

What the lines mean:

- `[model] pool: N pending unit(s)` — a new model started.
- `[k/N] 12.3 unit/min errors=E` — progress; a nonzero error count is not an
  emergency (see below).
- `ABORTED — circuit breaker` — twenty consecutive failures; something is
  systematically wrong. Read the nearest `error:` lines, and open an issue
  if they point at the harness rather than your setup.

**Do not `git pull` between launching a run and finishing it.** Resuming
finds the run by a fingerprint that includes the code version; updating the
code mid-run means a rerun would start a fresh directory instead of
continuing. Update the repository after your run completes and is packed.

**If the machine reboots or you kill the run**: run the exact same command
again. It finds its directory by configuration fingerprint and continues
where it stopped — already-answered questions are never redone.

**If the finished run shows a few errors** (out-of-memory on the longest
videos is the known kind): rerun with two GPUs per model copy — only the
failed questions are retried. Note the flag goes before the redirection,
not after the trailing `&`:

```bash
nohup .venvs/tf4/bin/python -m robochrono eval \
  --models qwen3-vl-2b-instruct rynnbrain1-1-122b-a10b \
  --gpus 4 --gpus-per-worker 2 > full_run_retry.log 2>&1 &
```

### 1.9 Archive the results

```bash
.venvs/tf4/bin/python -m robochrono report        # prints the table location
.venvs/tf4/bin/python -m robochrono pack          # results/<run_id>.tar.gz, a few hundred KB
.venvs/tf4/bin/python -m robochrono pack --full   # adds per-question records (hundreds of MB)
```

The default pack is what you move between machines or attach to a report.
Keep the full per-question records — they are the raw material for any later
analysis, and what lets a changed metric be recomputed without rerunning.

### 1.10 Flags you have not needed yet

Everything above runs on defaults; these five cover the bigger setups. All
of them appear in `--help` too.

- `--shard i/n` — this machine's share of the matrix. The suite × models
  expansion is split deterministically into `n` disjoint parts by a content
  hash, so the machines need no coordination; machine `i` (counted from 1:
  `--shard 1/2`, `--shard 2/2`) runs the same command plus its shard. Run every shard from the **same commit
  with a clean tree** so all parts carry the same fingerprint, then combine:
  `report <run-dir-A> <run-dir-B> …` merges run directories and refuses
  mismatched datasets.
- `--limit-groups N` — cap model *calls* per (scenario, dimension), where
  `--limit-items` caps *questions*. For every current dimension one call is
  one question, so they agree today; they exist separately so a sampled run
  states which one it means.
- `--label <name>` — a human name for the run, recorded in `run.json`.
  Deliberately outside the fingerprint: two machines running the same
  experiment stay the same experiment however they label it.
- `--fresh` — start a new run directory even though this exact
  configuration has run before (default behaviour is to resume it).
- `--overwrite` — within a run, redo already-answered questions; the old
  rows are kept in a `.bak` file rather than discarded.

`--suite` also accepts a path, so a throwaway subset suite can live outside
the repository during a quick experiment — but commit the suite into
`configs/suites/` for any result you publish, so readers can resolve the
exact question set it names.

---

## Part 2 — API models

No GPU, no weights, no heavy environments — any Linux machine with Python
3.11+ and `ffmpeg` works. Do steps 1.1 and 1.3 above (clone + dataset), then:

### 2.1 Keys

```bash
mkdir -p ~/.config/robochrono
cat > ~/.config/robochrono/keys.env << 'K'
DASHSCOPE_API_KEY=...       # Qwen3.8-Max, both arms, and the 235B
GEMINI_API_KEY=...          # Gemini
ARK_API_KEY=...             # Doubao
K
chmod 600 ~/.config/robochrono/keys.env
```

Keys never go into the repository or into any configuration file; the model
configurations name the environment variable, and you load them per shell:

```bash
set -a; source ~/.config/robochrono/keys.env; set +a
```

If your network cannot reach a provider directly, add a `proxy` field to
that model's `api` configuration (for example in
`configs/models/api/gemini-3-6-flash.json`) pointing at your proxy URL —
the API adapters route their calls through it when it is present.

### 2.2 A minimal environment and a preflight

```bash
UV_PROJECT_ENVIRONMENT=.venvs/api uv venv .venvs/api --python 3.11
uv pip install --python .venvs/api/bin/python requests pillow numpy
.venvs/api/bin/python -m robochrono preflight --only api
```

### 2.3 One-question probe per model (pennies)

```bash
for m in qwen3-8-max qwen3-8-max-thinking gemini-3-6-flash \
         doubao-seed-2-0-lite qwen3-vl-235b-a22b-api; do
  .venvs/api/bin/python -m robochrono eval --models "$m" \
    --scenarios make_tea_tianji --dimensions current_action \
    --limit-items 1 --no-dispatch
done
```

Every probe should answer with `errors=0`. This confirms keys, media upload,
and that each provider's thinking switch behaves as configured.

### 2.4 The run

```bash
nohup .venvs/api/bin/python -m robochrono eval \
  --models qwen3-8-max qwen3-8-max-thinking gemini-3-6-flash \
           doubao-seed-2-0-lite qwen3-vl-235b-a22b-api \
  --no-dispatch --api-concurrency 8 > api_run.log 2>&1 &
```

Two things to know:

- The 235B endpoint rate-limits harder than the rest. If its section of the
  log fills with `429`, let the run finish, then rerun the same command with
  `--api-concurrency 2` — only its failed questions are retried.
- Cost scales with what you select. `--limit-items N` caps questions per
  (scenario, dimension) if a sampled run is wanted instead of the full one.
  Note that `--dry-run` prices the **full selection** and does not account
  for `--limit-items` — use it to see the ceiling, not the sampled cost.

Monitoring, interruption recovery and archiving results are identical to
sections 1.8–1.9.
