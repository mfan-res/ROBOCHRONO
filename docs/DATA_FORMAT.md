# Data format

For readers who want to understand the questions, analyse the data, or check
that a download is intact. Everything below is stated against the shipped
data; examples are copied verbatim from the `pack_airpods_gim` scenario.

## The scenario directory

The dataset is a pool of self-contained scenarios. Each one carries
everything it needs — questions, wording, media, and its own manifest:

```
scenarios/<scenario_id>/
├── manifest.json      identity, counts, content hash, media inventory
├── qa/
│   ├── current_action.json          the seven dimension files
│   ├── next_action.json             (always all seven; a dimension the
│   ├── next_action_with_goal.json    scenario cannot ask has an empty
│   ├── frame_match.json              items list, not a missing file)
│   ├── view_match.json
│   ├── frame_order.json
│   ├── action_time.json
│   ├── subtasks.json                this scenario's actions: id -> wording
│   ├── scenarios.json               this scenario's goal and recording facts
│   └── dimensions.json              how each dimension words its question
│                                    (identical in every scenario, checked on load)
└── media/
    ├── clips/<scenario_id>/         one video per action segment
    ├── episodes/<scenario_id>/      full episodes; only action_time reads them,
    │                                and they are most of the volume — skip them
    │                                if you do not run that dimension
    └── frames/<scenario_id>/        still frames for the image dimensions
```

Media paths inside question files are relative to the scenario directory and
repeat the scenario segment (`media/clips/pack_airpods_gim/...`) — that
repetition is deliberate: a path reads the same wherever it appears, and a
scenario directory can be moved or subset without rewriting a byte of qa.
`tools/download_data.py` fetches scenarios into this layout (usage in the
README's Data section).

### What each scenario weighs

Media sizes per scenario, from the manifests — pick scenarios against your
disk with this, and remember `media/episodes/` (the right column) can be
skipped entirely unless you run `action_time`. The qa of all 39 scenarios
together is ~50 MB (`tools/download_data.py --all --qa-only`).

| Scenario | Questions | Media | of which episodes |
| --- | ---: | ---: | ---: |
| `bag_toy_gim` | 697 | 1.2 GB | 0.8 GB |
| `box_pen_hand` | 680 | 0.8 GB | 0.5 GB |
| `box_pen_tianjihand` | 1000 | 0.5 GB | 0.3 GB |
| `box_shoe_tianji` | 671 | 0.9 GB | 0.4 GB |
| `box_shoes_gim` | 833 | 1.0 GB | 0.7 GB |
| `brew_teabag_gim` | 924 | 3.8 GB | 2.9 GB |
| `brew_teabag_tianji` | 1092 | 5.3 GB | 3.8 GB |
| `cap_pen_gim` | 1200 | 2.5 GB | 1.4 GB |
| `make_tea_gim` | 1092 | 4.1 GB | 3.5 GB |
| `make_tea_tianji` | 819 | 3.0 GB | 1.7 GB |
| `move_flower_hand` | 160 | 0.4 GB | 0.2 GB |
| `move_flower_tianjihand` | 510 | 0.4 GB | 0.3 GB |
| `move_gift_tianjihand` | 510 | 0.5 GB | 0.3 GB |
| `pack_aidkit_gim` | 1050 | 14.2 GB | 7.9 GB |
| `pack_aidkit_tianji` | 840 | 4.0 GB | 2.2 GB |
| `pack_airpods_gim` | 378 | 0.8 GB | 0.4 GB |
| `pack_airpods_tianji` | 840 | 1.0 GB | 0.9 GB |
| `pack_express_tianji` | 1000 | 1.0 GB | 0.6 GB |
| `pack_gift_gim` | 1050 | 0.9 GB | 0.7 GB |
| `pack_gift_hand` | 720 | 0.3 GB | 0.3 GB |
| `pack_gift_tianji` | 861 | 1.4 GB | 1.0 GB |
| `pack_sunglasses_hand` | 738 | 1.2 GB | 0.7 GB |
| `slip_tshirt_gim` | 960 | 1.8 GB | 1.3 GB |
| `sort_cubes_gim` | 1239 | 1.1 GB | 0.8 GB |
| `sort_cubes_tianji` | 861 | 0.8 GB | 0.6 GB |
| `stack_cubes_gim` | 1050 | 2.4 GB | 1.3 GB |
| `stack_cubes_hand` | 731 | 0.8 GB | 0.4 GB |
| `stack_cubes_tianji` | 1000 | 1.0 GB | 0.9 GB |
| `stack_cubes_tianjihand` | 1000 | 0.6 GB | 0.3 GB |
| `stow_sunglasses_tianjihand` | 850 | 0.5 GB | 0.3 GB |
| `takeout_trash_tianji` | 819 | 1.9 GB | 1.8 GB |
| `tidy_stationery_gim` | 1836 | 1.9 GB | 1.0 GB |
| `tidy_stationery_tianji` | 1260 | 1.0 GB | 0.7 GB |
| `wash_dishes_gim` | 1029 | 3.2 GB | 2.8 GB |
| `wash_dishes_tianji` | 840 | 1.9 GB | 1.6 GB |
| `wipe_plate_gim` | 1050 | 2.3 GB | 1.8 GB |
| `wipe_plate_tianji` | 840 | 1.5 GB | 1.2 GB |
| `zip_pouch_gim` | 816 | 1.3 GB | 0.8 GB |
| `zip_pouch_tianji` | 867 | 0.8 GB | 0.6 GB |

Total: 74 GB.

## manifest.json

```jsonc
{
  "schema": 1,
  "scenario_id": "pack_airpods_gim",   // must equal the directory name
  "embodiment": "gim",                 // see the four values below
  "hash": "c0d5867f3b56…",             // sha256 of the model-visible content (64 hex)
  "questions": 378,                    // sum over the dimensions table
  "dimensions": {                      // question count per dimension; 0 is a
    "current_action": 54, /* … */      // structural gap, kept honest rather
    "action_time": 54                  // than dropped
  },
  "episodes": 18,
  "camera": {                          // recording facts, informational
    "resolution": [736, 416], "fps": 50.0,
    "views": ["main", "wrist_left", "wrist_right"], /* … */
  },
  "counts": {"episodes": 18, "segments": 108, "subtasks": 6},
  "next_action": {                     // transition statistics: when
    "transitions": 90,                 // conditional_entropy is 0.0 the action
    "unique_successor_share": 1.0,     // order never branches and next_action
    "conditional_entropy": 0.0         // is closer to recall than prediction
  },
  "media": {                           // the weak check: inventory, not hashes
    "files": 509, "bytes": 798769762,
    "entries": [ {"path": "media/clips/…", "bytes": 2766686}, /* … */ ]
  }
}
```

`embodiment` takes exactly four values: `gim` (a bimanual platform),
`tianji` (an arm with a parallel gripper), `tianjihand` (the same arm with a
five-finger hand), and `hand` (the same kinds of task performed by human
hands, no robot). It always equals the scenario name's suffix; the loader
refuses a manifest where the two disagree.

Manifests are written by `tools/compute_scenario_hash.py --write`, never by
hand.

## Question files

A stored question names *ids*, not sentences. The sentence a model reads is
rendered at load time from the three render files beside it: option wording
from `subtasks.json` (`id` → `text`), the goal from `scenarios.json`, the
question stem from the templates in `dimensions.json`. Ids are stored
instead of prose so a rewording is one edit, not a hunt for scattered
copies. Consequences worth knowing: raw items have **no** `question` field
and choice options have no `text` field — both appear only after rendering
(`robochrono.dataset.loader.load_questions` returns rendered items; result
rows record the full rendered prompt).

Fields shared by every item: `id` (globally unique,
`<scenario>/<episode>/s<segment>@<dimension>` — this is also the *unit key*
the engine resumes by and the replay adapter's table is keyed on, see
docs/ADDING_A_MODEL.md), `video_id`, `type` (the
dimension), `segment`, `subtask` and `next_subtask` (action ids scoped to
this scenario), and `input` (the media it shows).

### The six choice dimensions

Four options lettered `A`–`D`, `answer` is the correct letter. Scored as
accuracy; random guessing floors at 0.25. What varies is what the options
*are* and what media the model sees.

**`current_action`** — which action is happening in the clip. Options name
actions; the model sees the segment's clip.

```json
{"id": "pack_airpods_gim/file-000/s00@current_action",
 "video_id": "pack_airpods_gim/file-000", "type": "current_action",
 "segment": 0, "subtask": "pick_up_charging_case",
 "next_subtask": "open_charging_case",
 "input": {"clip_path": "media/clips/pack_airpods_gim/file-000@main@000000-000680.mp4"},
 "options": [{"id": "A", "subtask": "close_charging_case"},
             {"id": "B", "subtask": "pick_up_charging_case"},
             {"id": "C", "subtask": "put_left_earbud_in_case"},
             {"id": "D", "subtask": "put_down_charging_case"}],
 "answer": "B"}
```

**`next_action`** — same shape, but the question asks what comes next and
the correct option is the following segment's action.
**`next_action_with_goal`** — identical to `next_action` except the rendered
stem also states the scenario goal; the pair is a controlled comparison of
how much the goal helps.

**`frame_match`** — which of four still frames appears in the clip. Options
are images, not text:

```json
 "options": [{"id": "A", "image_path": "media/frames/pack_airpods_gim/file-000@main@003986.jpg"}, …],
 "answer": "D"
```

**`view_match`** — the model sees one head-camera frame (`input.image_path`)
and must pick the frame that a named wrist camera (`target_camera`, rendered
as "left"/"right" in the stem) recorded at the same moment. Distractor
images are the other wrist and other moments.

**`frame_order`** — three still frames (`input.image_paths`, presented in a
scrambled order) and four candidate chronological orderings; an option's
`order` lists image numbers, rendered as `"Image 2 -> Image 1 -> Image 3"`:

```json
 "options": [{"id": "A", "order": [2, 3, 1]}, {"id": "B", "order": [2, 1, 3]}, …],
 "answer": "B"
```

### `action_time` (temporal grounding)

No options. The model watches the full episode (`input.video_path`) and must
say when the named action happens. The ground truth is an interval:

```json
{"id": "pack_airpods_gim/file-000/s00@action_time",
 "input": {"video_path": "media/episodes/pack_airpods_gim/file-000@main.mp4"},
 "answer": "00:00:00.000-00:00:13.620",
 "answer_seconds": {"start": 0.0, "end": 13.62}}
```

Scored as temporal IoU; the headline metric is tIoU@0.5 because mean tIoU
has a ~0.13 floor reachable by always answering "the whole video". Scoring
uses `answer_seconds` (`start < end`, in seconds); the `answer` string is
the same interval for human readers.

## What the benchmark asks, in one paragraph

The same footage, queried seven ways: recognise the current action, predict
the next one (with and without knowing the goal), tie a still frame to a
clip, tie a wrist view to a head view, order moments in time, and localise
an action inside a full episode. The six choice dimensions test whether the
model read the video at all; `action_time` tests whether it can place events
on a timeline. Nothing claims a dimension isolates a capability — the names
describe what is asked.

## Hashes and integrity checking

Each scenario's `hash` is a sha256 over its **model-visible content**: the
seven dimension files' items, the subtask wordings (`id` → `text`), the
goal, and the question templates — canonicalised JSON, so formatting never
moves it. Deliberately outside the hash: media bytes, recording metadata
(`camera`, `counts`, transition statistics), and provenance fields.

Check a download:

```bash
.venvs/tf4/bin/python tools/compute_scenario_hash.py scenarios/pack_airpods_gim --verify
# ok: pack_airpods_gim c0d5867f3b56 (378 questions)

.venvs/tf4/bin/python -m robochrono validate-data   # the full sweep, every scenario
```

Know the boundary of each check. `--verify` passing means the questions,
answers and wording are exactly what the manifest (and any suite pinning
this hash) declare — and that the media inventory (paths and byte sizes)
matches the disk. It does **not** cryptographically prove the media bytes
are the original release: media integrity is a weak check by design, sized
for catching missing or truncated files, not tampering. `validate-data`
additionally loads and renders every question the way the evaluation does,
checks each item's shape (options lettered A–D with the answer among them,
`answer_seconds` a real interval), re-checks every count against the
manifest, confirms every referenced media file exists, and confirms the
inventory carries no dead weight (references = shipped files).

## Suites

A suite is the frozen set a published score cites —
`configs/suites/<name>.json`:

```jsonc
{
  "schema": 1,
  "name": "RoboChrono-full",
  "dimensions": null,          // null = all seven; or a subset list
  "scenarios": {               // scenario -> the full content hash it pins
    "bag_toy_gim": "5e38a2dd606a…",
    /* … */
  }
}
```

`official-v1.json` pins all 39 scenarios (34,713 questions);
`smoke.json` pins two small full-coverage scenarios (888 questions) for
quick end-to-end checks. Preflight recomputes every pinned hash from the qa
on disk before a run starts, so a revised or tampered scenario fails loudly
by name; `robochrono report` refuses to merge runs whose shared scenarios
carry different hashes.

To evaluate a subset, write your own suite file: copy the entries you want
from an existing suite (or read the hashes from the scenarios' manifests),
give it a name, and pass `--suite <name>`. Narrowing `dimensions` works the
same way — `["action_time"]` runs one dimension over the pinned scenarios.
Published numbers should cite suites, never ad-hoc `--scenarios` selections:
the suite file is what makes a score reproducible and comparable.

A suite file you write is part of the run's provenance: while it sits
untracked, every run made with it records `dirty: true` in `run.json`
(README, *Reading the results*). Commit the suite before runs you intend
to publish — which is also what lets anyone else reproduce them.
