# Few-Shot Learning MLOps on Vertex AI

A complete, gated MLOps cycle for a few-shot image classifier - from frozen,
checksummed data to a promoted production artifact - built rung by rung on
Google Cloud Vertex AI (Pipelines, Custom Jobs, Model Registry, Artifact
Registry, Experiments).

The model is deliberately simple. The point of this repository is everything
around it: **reproducibility, statistical gating, artifact verification,
promotion mechanics, and decision-level explainability** - the parts that make
an ML system auditable rather than just accurate.

---

## What is being trained

**Prototypical Networks (ProtoNet)** with a Conv4 encoder, on **Omniglot**
(1,623 handwritten characters from 50 alphabets, 20 samples each - the
classic few-shot benchmark).

Few-shot classification answers a different question than ordinary
classification. Instead of *"which of the classes I trained on is this?"*, it
answers: *"given only K examples each of N brand-new classes, which one is
this?"*. Operationally this is the "new category, five examples, recognize it
from tomorrow" scenario - new document type, new signature, new target class.

Training is **episodic**: every iteration samples an episode - N=5 random
classes, K=5 *support* images per class (the "examples you're given") and 15
*query* images to classify. The encoder embeds all images; each class's
support embeddings are averaged into a **prototype**; queries are classified
by **nearest prototype** (Euclidean distance). The encoder is trained so that
this nearest-prototype rule works for classes it has *never seen* - test
episodes use a disjoint set of characters.

Two properties of this setup matter for the rest of the repo:

- **The decision is a distance comparison.** That makes explanations
  first-class: "class B, because it is closest to *these five examples*" is
  the actual mechanism, not a post-hoc approximation (see Explainability).
- **Per-episode accuracy is a distribution, not a number.** Evaluation runs
  1,000 test episodes; the gates reason about the mean, its 95% confidence
  interval, and the spread (std) - not a single point estimate.

Current result on the frozen data: ~98% mean accuracy, 5-way 5-shot,
comfortably above the promotion bar (lower CI bound >= 0.90, std <= 0.05).

---

## The pipeline

```
verify -> train[:candidate image, Custom Job] -> LIGHT GATE
  |
  +-- [PASS] -> REGISTER (new version in Model Registry, no alias)
  |               |
  |               v
  |             HEAVY EVAL (container; verifies the SAVED artifact; HTML report)
  |               |
  |               +-- [PASS] -> PROMOTE MODEL  (alias "production" on the version)
  |               |             PROMOTE IMAGE  (retag :candidate -> :production, by digest)
  |               |
  |               +-- [FAIL] -> version stays registered, unaliased (diagnosable)
  |               |
  |               +-- EXPLAIN (always, parallel to promotion) -> HTML report
  |
  +-- [FAIL] -> REJECTION REPORT (HTML gauges: which criterion failed, by how much)
  |             EXPLAIN (model_version="not-registered") -> HTML report
  |
  (LOG to Vertex Experiments: always, both endings, including gate verdict)
```

Every run ends in exactly one of two explicit outcomes: **promotion** or an
**explained rejection**. Silence is not an outcome.

---

## Components

Only steps that need the few-shot logic (`fsl` package) run inside the
project's Docker image; everything else is a lightweight component (numbers
and API calls). This is the containerization rule for the whole repo.

| # | Component | Kind | What it does |
|---|-----------|------|--------------|
| 1 | `verify_frozen_component` | light | Recomputes and checks the SHA-256 of the frozen dataset archive in GCS against its manifest. The pipeline refuses to train on data that isn't exactly the frozen bytes. |
| 2 | `train_container` | **container** + Custom Job | Runs `scripts/train_pipeline_entry.py` inside the `:candidate` image (dedicated machine via `create_custom_training_job_op_from_component`). The wrapper calls the same `fsl.training.loop.train()` the inner loop uses, saves `model.pt` + `architecture.json` (incl. the seed) to GCS, and writes 6 outputs (metrics, model dir, data sha) through KFP OutputPath files. |
| 3 | `evaluate_gate_component` (light gate) | light | Cheap first filter on the training-time numbers. Two criteria: **lower CI bound** (`mean - ci95 >= accuracy_threshold`) and **stability** (`std <= max_std_threshold`). Both thresholds live in `configs/pipeline_config.json`. |
| 4 | `register_model_component` | light | Uploads the model dir as a new **version** of `fsl-protonet-omniglot` in Model Registry (versioning via `parent_model`), labeled with seed/framework and described with accuracy + data sha. Returns `(resource_name, version_id)`. Registration is *cataloguing*, not promotion. |
| 5 | `heavy_eval_container` | **container** | The decisive gate. Downloads the **registered artifact** from GCS, rebuilds the exact architecture from `architecture.json`, `model.eval()`, rebuilds the **same class split** from the saved seed, and recomputes accuracy over fresh test episodes. Three checks: **consistency with training** (statistical, not bitwise: `|acc_h - acc_t| <= ci95_t + ci95_h` - episodes are re-sampled, so exact equality is not expected; a large gap means broken serialization), **lower CI on the recomputed numbers**, **stability**. Renders a self-contained HTML report (pure-SVG histogram). |
| 6 | `promote_model_component` | light | `ModelRegistry.add_version_aliases(["production"], version=...)` on exactly the version heavy eval judged. The alias is unique per model and *moves* on the next promotion; consumers fetch `Model("...@production")`. Rollback = re-pointing the alias. |
| 7 | `promote_image_component` | light | Retags the image **by digest**: reads the version the `candidate` tag points at and points `production` at the *same* version (create-or-update; no pull/push). Promoted bytes == tested bytes, by construction. |
| 8 | `rejection_report_component` | light | The failure ending of the light gate. HTML with two gauge visualizations - a scale, the threshold line, and a dot showing where the model landed (red zone = the reason). Makes "rejected" legible: the mean can look fine while the lower-CI bound fails. |
| 9 | `explain_container` | **container** | Case-based reasoning report; runs in **both** branches (it describes, never decides). See below. |
| 10 | `log_experiment_component` | light | Unconditional. Logs params + metrics + the gate verdict (`gate_passed`, `gate_reason`) to Vertex Experiments - rejected runs leave a record too. |

### Explainability (component 9, in detail)

ProtoNet's decision *is* a distance comparison, so the report shows the actual
mechanism on concrete examples:

- a **"How to read this"** primer (the mechanism in three sentences),
- three **case studies** picked automatically from real test episodes - the
  most *confident* decision (largest margin), the most *borderline* (smallest
  margin), and a *misclassification* if one occurs - each rendered as
  *query image | distance bars to all five prototypes | the support images of
  the chosen class* (on a miss, an orange outline marks the true class's bar),
- a 2D **embedding map** of one episode (prototypes as diamonds, queries as
  dots; PCA via `torch.svd`, labeled as illustrative),
- a **margin histogram** across all sampled episodes, with the share of
  decisions made "by a whisker" (margin < 0.5).

Honest scope, stated in the report itself: this is **decision-level**
explainability (distances - the auditable mechanism), not representation-level
(why the encoder embeds two images nearby stays inside the network).

No plotting or imaging dependencies were added for any report: images are
rendered via PIL (already present through torchvision), all charts are
hand-rolled SVG, PCA uses `torch.svd`.

---

## Design decisions worth stealing

**Registration != promotion.** Every candidate that clears the cheap filter is
*catalogued* (a version with metrics and lineage); only heavy-verified
versions get the `production` alias. The registry keeps rejected candidates
visible - "evaluated, rejected, kept for the record" is a feature, not noise.

**Two-stage gating, cheap-then-expensive.** The light gate (arithmetic on
training outputs, seconds) filters obvious failures; the heavy gate
(container, loads data, recomputes) runs only for plausible candidates - and
it judges the **saved artifact**, not the in-memory model, because that is
what gets promoted. This ordering caught a real bug class in development:
serialization/BatchNorm-eval mismatches are invisible to the light gate.

**Test what you promote.** Promotion never builds anything. The image is built
*first* (`:candidate`), tested as-is, and promoted by **retagging the digest**;
the model version evaluated by heavy eval is the version aliased. There is no
window in which the promoted artifact could differ from the tested one.

**Pointers over copies.** "Production" is a moving pointer (model alias, image
tag) on an immutable, verified artifact. Rollback is re-pointing, not
rebuilding or retraining.

**Containers only where the domain logic lives.** Training, heavy eval and
explainability import `fsl` from the image; the gate, registration, promotion
and logging are plain API calls and stay lightweight. One image, three entry
points (`scripts/*_pipeline_entry.py`), zero duplicated model code.

**Reproducibility as a chain.** Data frozen in GCS with a SHA-256 manifest
(verified at the start of every run) -> seed stored inside the model artifact
-> the same seed rebuilds the same class split at evaluation time -> the data
sha travels into the model description and the experiment log.

**Statistical honesty.** Gates use the lower CI bound (confidence penalized by
spread), not the mean; the artifact-consistency check compares within combined
confidence intervals instead of demanding bitwise equality of re-sampled
evaluations; the rejection report visualizes exactly which criterion failed
and by how much.

**Both endings are first-class.** A rejected run produces a rejection report,
an explainability report, and an experiment record - never silence.

---

## Repository layout

```
fsl-projekt/
├── Dockerfile                     # one image: src/fsl + scripts/ + deps
├── pyproject.toml
├── configs/
│   └── pipeline_config.json       # project/region/bucket, image URIs,
│                                  # gate thresholds, training params, machine type
├── scripts/
│   ├── train.py                   # inner-loop CLI (Vertex Experiments logging)
│   ├── train_pipeline_entry.py    # container entry: train + save + KFP outputs
│   ├── evaluate_pipeline_entry.py # container entry: heavy eval + HTML report
│   ├── explain_pipeline_entry.py  # container entry: explainability report
│   ├── build_image.ipynb          # one-run image build+push (default tag :candidate)
│   ├── publish_template.py        # auto-incremented pipeline template versions
│   ├── bootstrap_fsl_data.py      # freeze dataset in GCS + SHA-256 manifest
│   └── verify_frozen.py
├── src/fsl/                       # the single home of all model logic
│   ├── config.py                  # TrainConfig dataclass (a run == its config)
│   ├── data/omniglot.py           # frozen-data loading, task samplers
│   ├── models/protonet.py         # Conv4, fast_adapt (episode step)
│   └── training/loop.py           # train() -> metrics + model; evaluate()
└── notebooks (rungs)              # one notebook per rung, each verified by
                                   # compilation before first run
```

---

## The working cycle

1. Change code in `src/fsl/`.
2. Rebuild the image: run `scripts/build_image.ipynb` (config-driven, defaults
   to `:candidate`; Docker layer cache makes code-only rebuilds fast).
   **Any new/changed file in `scripts/` requires a rebuild** - the pipeline
   runs what is in the image, not what is on disk.
3. Run the test pipeline (latest rung notebook): it trains **using the
   `:candidate` image**, gates, registers, verifies, and - if everything
   passes - promotes both the model version and the image.
4. Consumers use `Model("...@production")` and the `:production` image tag.

To watch the rejection path work, temporarily set
`evaluation.accuracy_threshold: 0.99` in the config and run once.

---

## Requirements & gotchas

- GCP project with Vertex AI, GCS, Artifact Registry enabled; a Workbench (or
  any environment) with Docker for image builds.
- The pipeline service account needs: GCS read/write on the data bucket,
  `aiplatform` model upload/update (aliasing), and
  `artifactregistry.tags.create/update` on the image repo.
- `gcloud ai models list-versions` may not exist in your gcloud version; list
  aliases via the Python SDK (`ModelRegistry.list_versions()`).
- The image historically used `ENTRYPOINT ["python", "scripts/train.py"]`;
  KFP's `command` overrides it, but for manual `docker run` diagnostics use
  `--entrypoint` (e.g. `docker run --rm --entrypoint ls IMAGE scripts/`).
  Recommended: switch to `CMD` on the next rebuild.
- Serving is intentionally out of scope: few-shot inference needs a support
  set per request, so a real endpoint requires a custom serving container
  (a separate project). "Deploy" in this repo means *moving the alias*.

## Status & roadmap

Done and verified live: the complete test pipeline above (both endings),
promotion confirmed in both registries, explainability reports on real
episodes.

Next: the **production pipeline** (same components, `:production` image,
weekly scheduler trigger, alias-move as deploy), then optional rungs:
hyperparameter tuning via Vertex Vizier (the training wrapper is already
parameterized for it), and an active-learning loop that would use this
pipeline as its inner engine.
