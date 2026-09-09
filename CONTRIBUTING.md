# Contributing

## Running the tests

The tests are standalone scripts — no pytest, no fixtures to install. Run
them with the tf4 interpreter (built by `uv sync --extra tf4`, see the
README's Quickstart):

```bash
for t in tests/test_*.py; do echo "== $t"; .venvs/tf4/bin/python "$t"; done
```

Every test must end with `all checks passed` or an explicit `skip:` line
that names what is missing. Tests that need the dataset skip cleanly when
`scenarios/` is absent; `test_runtime_equivalence` skips unless a private
reference implementation is present — on your machine that skip is expected.

## Pull requests

- One concern per PR, with the reasoning in the commit message — this
  repository's history explains *why*, not just *what*.
- Match the surrounding code: comment density, naming, standard library
  first. Adapters are the only modules that may import an inference stack,
  and only lazily inside methods.
- Every value copied from a model's documentation carries an evidence grade
  (`configs/README.md`). Fill it honestly; `source: "none"` is a valid and
  common answer.

## Adding a model

See [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md) — most models are one
configuration file and no code. It ends with the verification checklist a
model PR is expected to have walked through.

## Changes that need extra verification

Some parts of this codebase carry results-affecting weight. If your change
touches any of the following, say so in the PR and run the full test suite,
not just the tests nearest your change:

- **the data layer** (`robochrono/dataset/`, the scenario-root resolution in
  the loader) — a path mistake here reads the wrong files silently;
- **run identity** (`robochrono/results/runid.py`, `config/suites.py`) —
  the fingerprint decides when runs resume and when results may merge;
- **parsing and scoring** (`robochrono/parsing.py`, `robochrono/dimensions/`)
  — any change here changes what published numbers mean, and existing runs
  would need re-scoring (`tools/rescore.py`) rather than silent comparison;
- **the protocol** (`configs/protocol.json`) — results produced under a
  different protocol are not comparable, by design.

Note that *any* commit changes the code version inside the run fingerprint:
in-flight evaluations should finish before a change lands, and a rerun after
it starts a fresh run directory on purpose.
