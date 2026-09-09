# configs/

Configuration is split by **who changes it**:

> Does changing this alter what the same model returns for the same question?
> Yes → `protocol.json`. No → `runtime.json`.

| File | Holds | Effect of changing it |
| --- | --- | --- |
| `protocol.json` | Temperature, generation budget, frame sampling, parsing, degenerate floors, system prompt | Results produced under a different protocol are not comparable |
| `models/local/*.json`<br>`models/api/*.json` | How one model is adapted; what its documentation states | That model only |
| `suites/*.json` | Which scenarios and dimensions a published number covers | Frozen; add a new suite rather than editing one |
| `environments.json` | Which Python environment each model needs | None on results |
| `runtime.json` | Concurrency, proxying, cache location | None on results; **not recorded in results** |

What the dataset contains — question counts, the content hash, the media
inventory — lives in each scenario's `scenarios/<id>/manifest.json`. Each
fact is declared in exactly one place.

---

## Evidence grades

Every value in a model configuration that comes from that model's official
documentation carries a `source` grade. **It is required**:
`tests/test_config_consistency.py` fails if one is missing.

| Grade | Meaning | Example |
| --- | --- | --- |
| **L1** | Stated as a recommendation | A model card's recommended output length |
| **L2** | Used in an official example | A dtype chosen in the card's sample code |
| **L3** | Default in an official function signature | A frame count defaulted in a loader |
| **L4** | Default of an official companion library | The sampling rate a vendor's utility uses |
| **L5** | Inherited from the same architecture | Following the base model's conventions |
| **L6** | Our own decision | Disabling a model's reasoning mode |
| **none** | The documentation does not say | A card that states no library version |

**L5, L6 and `none` are not statements by the model's authors** and are reported
as such.

### Why the grade is recorded

**Three different things otherwise look identical.** One model's generation
budget is taken verbatim from its card; several models' cards state no budget at
all. Written as bare numbers they are indistinguishable, and a reader comparing
models cannot tell which figure carries authority.

**Deviations become discussable rather than invisible.** One model's card
recommends a sampling rate the protocol does not use, a dtype the runs do not
use, and an output format the protocol disables. Each is recorded next to what
the card says, with its grade. Under a single grade for the whole block, all
three would disappear behind one summary line.

**It resists false authority.** A configuration file reads as authoritative — a
number sitting in one is assumed to have a provenance. Marking a value `L6` says
plainly that it is a choice made here, not a recommendation received.

### What it does not do

- **It does not make the value right.** The grade records where a value came
  from, not whether it should have been adopted. A recommended sampling
  temperature is `L1` and still not used, because reproducibility matters more
  here than matching interactive defaults.
- **It does not check that the grade is accurate.** A missing `source` fails;
  an `L6` mislabelled as `L1` does not. There is no machine-checkable standard
  for the latter, and enforcing one would freeze today's judgement into a test.
- **It depends on being filled in honestly.** Filled carelessly, it degrades
  into decoration.

Its value is that it forces the question *where did this number come from* to be
answered at all.
