# RoboChrono

A temporal visual reasoning benchmark over egocentric robot manipulation video.

---

## What it measures

**39 manipulation scenarios × 7 evaluation dimensions = 34,713 questions,
from 1,816 episodes recorded on three robot platforms and by human hands.**

A *dimension* is a way of asking — the same footage queried seven different
ways. It is not a scoring axis.

| Dimension | Asks | Input | Metric | Questions |
| --- | --- | --- | --- | ---: |
| `current_action` | What is happening right now | clip | accuracy | 5,407 |
| `next_action` | What the robot should do next | clip | accuracy | 4,880 |
| `next_action_with_goal` | What to do next, given the overall goal | clip | accuracy | 4,880 |
| `action_time` | When a named action occurs | full episode | tIoU@0.5 | 5,407 |
| `view_match` | Which image is a given wrist camera's view at this moment | head image + option images | accuracy | 4,835 |
| `frame_match` | Which option image appears in the clip | clip + option images | accuracy | 5,325 |
| `frame_order` | The chronological order of three frames | still frames | accuracy | 3,979 |

`next_action` and `next_action_with_goal` differ only in whether the prompt
names the overall task, which is what makes the pair a controlled comparison.

Names describe **what is asked**, not what an answer proves. A dimension is not
claimed to isolate a capability — see *Reading the results* below for what each
score does and does not support.

Scenarios are named `<task>_<embodiment>`, across four embodiments —
`tianji` (an arm with a parallel gripper), `tianjihand` (the same arm with a
five-finger hand), `gim` (a bimanual platform) and `hand` (the same kinds of
task performed by human hands, no robot):

| Embodiment | Scenarios |
| --- | --- |
| `gim` (15) | `bag_toy_gim` `box_shoes_gim` `brew_teabag_gim` `cap_pen_gim` `make_tea_gim` `pack_aidkit_gim` `pack_airpods_gim` `pack_gift_gim` `slip_tshirt_gim` `sort_cubes_gim` `stack_cubes_gim` `tidy_stationery_gim` `wash_dishes_gim` `wipe_plate_gim` `zip_pouch_gim` |
| `tianji` (14) | `box_shoe_tianji` `brew_teabag_tianji` `make_tea_tianji` `pack_aidkit_tianji` `pack_airpods_tianji` `pack_express_tianji` `pack_gift_tianji` `sort_cubes_tianji` `stack_cubes_tianji` `takeout_trash_tianji` `tidy_stationery_tianji` `wash_dishes_tianji` `wipe_plate_tianji` `zip_pouch_tianji` |
| `tianjihand` (5) | `box_pen_tianjihand` `move_flower_tianjihand` `move_gift_tianjihand` `stack_cubes_tianjihand` `stow_sunglasses_tianjihand` |
| `hand` (5) | `box_pen_hand` `move_flower_hand` `pack_gift_hand` `pack_sunglasses_hand` `stack_cubes_hand` |

Thirteen of the twenty-three tasks appear on more than one embodiment
(`stack_cubes_*` on all four, `pack_gift_*` on three), which allows
like-for-like cross-embodiment comparisons on those tasks. Each scenario's
`manifest.json` records its embodiment, so subsets slice cleanly by platform.

### A question, as a model sees it

From `pack_airpods_gim`, dimension `frame_order`. The model receives three
still frames from one episode —

```
media/frames/pack_airpods_gim/file-000@main@000765.jpg   (shown as Image 1)
media/frames/pack_airpods_gim/file-000@main@000226.jpg   (shown as Image 2)
media/frames/pack_airpods_gim/file-000@main@001381.jpg   (shown as Image 3)
```

— followed by this question, rendered exactly as below:

```
The images above show three moments from the same episode, presented in
random order. Which option lists them in the correct chronological order?
Options:
A. Image 2 -> Image 3 -> Image 1
B. Image 2 -> Image 1 -> Image 3
C. Image 1 -> Image 2 -> Image 3
D. Image 1 -> Image 3 -> Image 2
```

The correct answer is **B**: the robot picks up the charging case (Image 2),
opens it (Image 1), then places an earbud (Image 3). Nothing in any single
frame gives the order away — the model has to know what these manipulations
look like as a sequence.

---

## Quickstart

### 0. What you need

- **Linux.** The pinned video decoder ships no macOS or Windows wheels;
  this holds for the API-only path too.
- **[`uv`](https://docs.astral.sh/uv/)** — it installs Python 3.11 and
  builds the environments below. `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **`git` and `ffmpeg`** (with `ffprobe`) on PATH.
- **For local models**: NVIDIA GPU(s) with a driver for CUDA 12.8. The 2B
  model used in the examples runs on a single 24 GB card; the largest
  models declare their own multi-card needs in `configs/models/`. The
  API-only path needs no GPU at all.
- **A model to evaluate**: local weights downloaded from Hugging Face, or
  an API key for a served model — the [tutorial](docs/TUTORIAL.md) covers
  both (§1.4 weights, §2.1 keys).
- **Disk**: up to ~74 GB for the full dataset (a single scenario is
  0.3–14 GB; per-scenario sizes in docs/DATA_FORMAT.md), ~12 GB for the
  two environments, plus your models' weights
  (4 GB for the 2B example). Results are megabytes. All of it lives under
  the repository directory (`scenarios/`, `.venvs/`, `models/`), so clone
  onto a disk with room for the sum.

### 1. Environments — **two are required**

Model families have mutually exclusive requirements: some load only under
transformers 4.x, others only under 5.x. No single environment runs every model.

There is no PyPI package: clone the repository and build both environments
in place — the project deliberately does not pin a single transformers
version, so one `pip install` could never produce a working setup.

```bash
uv python install 3.11
UV_PROJECT_ENVIRONMENT=.venvs/tf4 uv sync --extra tf4 --python 3.11
UV_PROJECT_ENVIRONMENT=.venvs/tf5 uv sync --extra tf5 --python 3.11
```

Run the two syncs one after the other, not in parallel — they contend for
the same cache lock. The first build downloads ~2.5 GB of torch per
environment — minutes to tens of minutes depending on your link to PyPI;
with a warm cache a rebuild takes a couple of minutes. If `uv`
is configured to use a package mirror, `uv sync` rewrites `uv.lock` to point
every source at the mirror — hundreds of changed lines. Restore the file
afterwards (`git checkout uv.lock`): a modified lock is easy to commit by
accident, and it marks every run you record as `dirty` (see *Reading the
results*). All commands below use the tf4 interpreter explicitly
(`.venvs/tf4/bin/python -m robochrono …`); if you prefer, activate the venv
once (`source .venvs/tf4/bin/activate`) and write `robochrono …` instead.

```bash
.venvs/tf4/bin/python -m robochrono preflight   # checks both environments, prints what to fix
```

Before data and weights are in place, preflight lists them as FAIL — that
is its job; rerun it as the pieces arrive, or scope it to what you are
about to run (`--suite smoke --models <slug>`; without `--suite` it checks
the default `official-v1`, all 39 scenarios). The second environment is
probed — through its own interpreter — only when a selected model needs it;
a selection that runs entirely in the current environment skips the probe
with a visible SKIP line.

### 2. Data

The dataset is a pool of self-contained scenarios under `scenarios/` — one
directory per scenario, each carrying everything it needs:

```
scenarios/<scenario_id>/
├── manifest.json      embodiment, content hash, question counts, media inventory
├── qa/                the seven dimension files, and everything they are rendered from
└── media/             clips, episodes, frames for this scenario
```

Every command reads the pool named by `--data-root` (default: `scenarios/`).
Partial downloads are per scenario: take only the directories you want, and
skip `media/episodes/` inside them if you are not running `action_time` —
episodes are the bulk of the volume and only that dimension reads them.

A suite (`configs/suites/*.json`) is the frozen set a published score cites:
it pins each scenario by the sha256 of its model-visible content, with
`"dimensions": null` meaning all seven. Published suites are never edited —
a revised scenario changes its hash, so it gets a new suite file rather than
silently replacing the old numbers. `official-v1` pins all 39 scenarios,
34,713 questions; `smoke` pins two small ones for a quick end-to-end check.
Preflight verifies every pinned hash against the pool before anything runs;
`report` refuses to merge runs whose shared scenarios carry
different content. The hashes a suite pins are taken
from the pool's manifests, never invented by hand, and anyone can recompute
a scenario's hash with `tools/compute_scenario_hash.py <scenario_dir>
--verify`. The suite *files* are another matter: the published ones are
frozen, but a custom subset suite is yours to write — copy the pinned
entries you want into a new file (docs/DATA_FORMAT.md, "Suites").

The scenarios live in two public Hugging Face repositories
([`gimai/RoboChrono-GIM`](https://huggingface.co/datasets/gimai/RoboChrono-GIM)
and
[`gimai/RoboChrono-Tianji`](https://huggingface.co/datasets/gimai/RoboChrono-Tianji));
the download script finds each scenario on its own and leaves it unpacked,
verified and ready to run:

```bash
.venvs/tf4/bin/python tools/download_data.py --all               # all 39, ~74 GB
.venvs/tf4/bin/python tools/download_data.py \
    pack_airpods_gim move_gift_tianjihand    # the two scenarios `--suite smoke` pins (~1.3 GB)
.venvs/tf4/bin/python tools/download_data.py --all --qa-only     # questions only, ~50 MB
.venvs/tf4/bin/python tools/download_data.py cap_pen_gim --media-types clips,frames
```

A single scenario is 0.3–14 GB depending on its episode count
(per-scenario sizes: docs/DATA_FORMAT.md); the download runs at whatever
your link to Hugging Face sustains. Both repositories are public — no
Hugging Face account or token is needed.

**Only fetched part of the pool?** `eval` refuses to run a suite whose
pinned scenarios are not all present, and `--scenarios` narrows *within* a
suite without relaxing that check — so neither will get a partial pool
running. The supported path is a subset suite that pins exactly what you
have. Say you fetched only `cap_pen_gim`; print its content hash (the same
value sits in `scenarios/cap_pen_gim/manifest.json` under `"hash"`):

```bash
.venvs/tf4/bin/python tools/compute_scenario_hash.py scenarios/cap_pen_gim
# f34b240ce936f3506efed66a198bb037a088e58ad478a4505505155e74e228f4
```

put it in a new file `configs/suites/one-scenario.json`:

```json
{
  "schema": 1,
  "name": "one-scenario",
  "dimensions": null,
  "scenarios": { "cap_pen_gim": "<the 64-hex hash printed above>" }
}
```

and run with `--suite one-scenario` in place of `--suite smoke` below
(`"dimensions": null` means all seven; list a subset to narrow). One
provenance note: until the suite file is committed it is an untracked file,
so runs made with it record `dirty: true` (see *Reading the results*) —
fine for a local check, commit the file before runs you mean to publish.

Interrupted downloads resume with the same command. The script ends by
running validate-data; you can also fetch by hand and validate
yourself:

```bash
.venvs/tf4/bin/hf download gimai/RoboChrono-GIM --repo-type dataset \
    --include "cap_pen_gim/*" --local-dir scenarios
for t in scenarios/*/media/*.tar; do
    tar -xf "$t" -C "$(dirname "$(dirname "$t")")" && rm "$t"
done
.venvs/tf4/bin/python -m robochrono validate-data   # sweeps every scenario present
```

### 3. Run

You need a model first: weights under `models/<name>` for a local model,
or an exported API key (tutorial §2.1). The table in tutorial §1.4 maps
every configured model to its Hugging Face repository and download
directory; for the 2B used below:

```bash
.venvs/tf4/bin/hf download Qwen/Qwen3-VL-2B-Instruct \
    --local-dir models/Qwen3-VL-2B-Instruct        # ~4 GB
```

Then:

```bash
# price a run before starting it: question count, files, gigabytes sent.
# without --models the default is every configured model, paid API ones
# included — name what you actually plan to run
.venvs/tf4/bin/python -m robochrono eval --suite smoke --dry-run \
    --models qwen3-vl-2b-instruct

# a small end-to-end check: --models picks from configs/models/,
# --limit-items caps questions per (scenario, dimension), --gpus says
# how many cards to use
.venvs/tf4/bin/python -m robochrono eval --suite smoke \
    --models qwen3-vl-2b-instruct --limit-items 2 --gpus 2
# other jobs on the machine? pick your free cards with --gpus 2,5 (a list
# names exact card indices); if memory is tight, add --gpus-per-worker 2
# (details: tutorial §1.8)

# the full official benchmark — all 39 scenarios, 34,713 questions
.venvs/tf4/bin/python -m robochrono eval --models qwen3-vl-2b-instruct --gpus 8
```

For scale: the full smoke suite (888 questions) takes the 2B model about
12 minutes on eight 24 GB cards; the `--limit-items 2` check above is about
two minutes; the full benchmark ran in 5.7 hours on the same hardware.
Environment switching is automatic — each model is dispatched to the
interpreter it needs. No manual activation.

### 4. Report

```bash
.venvs/tf4/bin/python -m robochrono report            # the most recent run
.venvs/tf4/bin/python -m robochrono report <run_id>   # a specific one
.venvs/tf4/bin/python -m robochrono pack              # run.json + summaries + report
.venvs/tf4/bin/python -m robochrono pack --full       # adds the per-question records
```

The default archive is a few hundred KB — enough to read, merge and cite
the scores. The per-question records (the third layer under *Reading the
results*) travel only with `--full`; keep them somewhere, they are what
lets a changed metric be recomputed without rerunning anything.

(`preflight`, `validate-data`, `eval`, `report` and `pack` are the five
commands; each takes `--help`.)

Three guides live in `docs/`:

- **[TUTORIAL.md](docs/TUTORIAL.md)** — the full walkthrough from a bare
  machine to a finished run, for both the GPU and the API-only path. Start
  here if the Quickstart above is your first contact with the project.
- [ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md) — putting your own model on
  the roster.
- [DATA_FORMAT.md](docs/DATA_FORMAT.md) — the scenario layout, the question
  schema and the integrity checks.

---

## How it works

```
robochrono/
├── config/        readers for configs/ — protocol, models, suites, environments
├── dataset/       the data contract: manifest, question loading, rendering
├── dimensions/    how each dimension asks, parses and scores
├── parsing.py     extracting an answer from free-form model output
├── results/       run identity, per-question storage, reporting
├── adapters/      one file per model family — the only code importing an inference stack
├── engine.py      runs one (model, scenario, dimension): load → call → parse → score → store
├── media_prep.py  API only: fitting media into a request-size budget
├── orchestrate/   expands suite × models, dispatches to the right environment, pools GPUs
├── preflight.py   pre-run checks: environments, weights, data, configuration
└── cli.py         thin entry points for the five commands
```

One `eval` invocation flows top to bottom:

1. **config** reads the protocol, suite and model files; anything missing is a
   hard error rather than a default.
2. **preflight** cross-checks environments, weights and the dataset
   fingerprint before anything runs.
3. **results/runid** fingerprints the configuration and finds or creates
   `results/<date>_<fingerprint>/`. Re-running the same command resumes in
   place; changing the experiment lands in a new directory.
4. **orchestrate** expands the suite into (model, scenario, dimension)
   combinations and starts each model under the interpreter its environment
   declares — no manual environment switching.
5. **engine** works through each combination one *unit* at a time — a unit
   is one model call, which for every current dimension means one question;
   the `pool: N pending unit(s)` log line counts these. Per unit: the
   loader renders questions, the dimension
   assembles the call, the adapter talks to the model, parsing extracts an
   answer, the dimension scores it, and the store appends one JSONL row per
   question, which is also what makes interrupted runs resumable.
6. **results/report** aggregates summaries into one table — and refuses to
   merge results produced on different datasets.

Three structural rules hold the design together:

- **The dataset is the contract.** Questions, wording and the content hash
  are self-contained under each `scenarios/<id>/qa/`; the loader adapts to
  the data, never the other way around.
- **Results carry their identity.** The configuration fingerprint runs through
  the directory name, `run.json` and the report header, so scores from
  different datasets or protocols cannot silently mix.
- **Orchestration never imports the inference stack.** Only `adapters/` may
  import torch or transformers, and only inside worker processes. This is what
  lets one command drive models with mutually exclusive dependencies.

## Reading the results

A run directory holds three layers, and they answer different questions:

| File | Answers |
| --- | --- |
| `run.json` | what experiment this was — the suite's scenario pins, protocol, models, code |
| `<model>/<scenario>/<dimension>.summary.json` | the metrics for one combination |
| `...<dimension>.jsonl` | every question: full prompt, raw output, parse, score |

`run.json`'s `code` block is the run's provenance: the commit it ran from,
and — when the working tree differed from that commit — `dirty: true` plus a
hash of the diff. A dirty run cannot be reproduced from the commit alone, so
keep the tree clean for runs you intend to publish. Any uncommitted
change sets the flag — including untracked files: the two common accidents
are a mirror-rewritten `uv.lock` (§1) and a custom suite file you have not
committed yet (§2). Commit both before a run you mean to publish.

Before comparing any numbers, read the two flag sections of `report.md`:

- **✗ did not execute properly** — that cell is not a score. The calls failed,
  or the output could not be read at scale. Fix the setup before comparing.
- **⚠ at or below the degenerate floor** — the score is no better than a
  strategy that never watches the video (0.25 for four-way choice; answering
  the whole clip for temporal grounding). It does not say *why*.

Three fields that look alike and are not: `errors` counts calls that
failed outright — out of memory, timeouts, API errors — and the error rows
record why; `answered` counts calls that returned text; `parse_failure_rate`
counts answers no parser could read, *among the calls that succeeded* — a
failed call is never also counted as a parse failure. A run can be 100%
answered, 0 errors, and still measure nothing — unreadable answers score
zero, by the same convention in every dimension.

Generation settings are part of what is measured. Models running with
different thinking settings are different experiments sharing a table; the
report's *Execution settings* section lists each model's settings, and flags
any model that appears under more than one.

---

## Leaderboard

The results table will be published together with the paper.

---

## Known limitations

Dataset-side, know these: transition entropy — in 31 of the 39 scenarios
the action order never branches, so `next_action` there is closer to recall
than prediction (each scenario's `manifest.json` records its
`next_action.conditional_entropy`); some distractor options are authored
rather than observed; `action_time` scores are sensitive to episode length;
and the `gim` footage is upscaled from a lower native resolution.
Evaluation-side:

- **Scores under different thinking settings are not comparable.** Models that
  cannot disable thinking declare it, and the report flags them.
- **`frame_order` sits near the random floor for every model tested so far.**
  A low score there separates nothing yet.
- **Occasional out-of-memory failures are recorded, not hidden.** Rerun the
  same command with `--gpus-per-worker 2` to retry only the failed questions.
- **InternVL-architecture models overflow their context on full episodes.**
  At the protocol's one frame per second, a two-minute episode tokenizes to
  ~31k tokens against a 12,288-token window; the tokenizer warns, inference
  proceeds, and the near-floor `action_time` scores for these models reflect
  that reality. The frame density is not lowered per model — a context window
  that cannot hold the input is a measured limitation, not a harness defect.
- **Answers ship with the dataset.** Scoring needs them, so every question's
  answer is public and there is no held-out split. Results are self-reported:
  there is no submission server and no blind test set. Treat leaderboard-style
  comparisons accordingly.
- **Results are deterministic per stack, not across stacks.** Decoding is
  greedy, and on identical hardware, driver and library versions a full run
  reproduces its raw outputs byte for byte. Across different GPUs or library
  builds, floating-point differences can flip individual answers that sit
  near a decision boundary; expect per-dimension scores to agree within a
  few tenths of a percentage point, not bit-exactly.
- **Platform requirements are narrow.** Linux only (the pinned `decord`
  ships no macOS or Windows wheels), CUDA 12.8 for the pinned torch build,
  Python 3.11. API-only evaluation still needs Linux for the same reason.

---

## Citation

A paper describing this benchmark is in preparation. Citation information
will be added upon publication.

## License

The code is licensed under [Apache-2.0](LICENSE). The dataset is released
under CC-BY-4.0 with the two Hugging Face dataset repositories.
