# Challenge documentation — Software Engineer (ML & LLMs)

This document records the decisions, fixes and rationale behind operationalizing
the Data Scientist's flight-delay model. It is organized by the four challenge
parts, followed by an **Architecture Decision Record (ADR)** log.

## Working method

- **GitFlow** branching: feature branches `feature/lt-mle-0XX-<slug>` are merged
  via Pull/Merge Requests; development branches are preserved (never deleted).
- Work is tracked as sequential tickets (`LT-MLE-001` … `LT-MLE-007`), each with
  a branch, an MR and acceptance criteria.
- Tooling: **uv** (env + lockfile), **ruff** (lint/format), **Python 3.10**.
- The provided `requirements*.txt` are preserved as the canonical install used by
  the Makefile targets and the Docker image.

---

## Part I — Model

### Transcription & bug fixes
The notebook (`exploration.ipynb`) was transcribed into `challenge/model.py` as a
`DelayModel` class with `preprocess` / `fit` / `predict`. Fixes and good
practices applied:

- **Skeleton bug**: the provided return annotation `Union(Tuple[...], pd.DataFrame)`
  (a runtime `TypeError`) was corrected to valid typing.
- **Feature construction is deterministic**: features are one-hot encoded from
  `OPERA`, `TIPOVUELO`, `MES` and then **reindexed to the fixed top-10 feature
  set** (`fill_value=0`). The same code path therefore works for training (full
  dataset) and for serving a single flight that only activates a couple of
  categories — avoiding the classic "missing dummy column at inference" bug.
- **Target** is computed as `delay = 1 if min_diff > 15 else 0`, returned as a
  one-column DataFrame when `target_column` is provided.
- **Persistence**: the fitted estimator is saved with `joblib` and loaded in
  `__init__`, so `predict` works on a fresh instance (and the API serves
  immediately at start-up).

> Notebook bugs noted: the `sns.barplot` calls pass positional `x, y` arrays
> (deprecated/raises in modern seaborn) and the EDA `get_rate_from_column`
> computes `total/delays` (inverted ratio). These are EDA-only cells and do not
> affect the productive model, so they were left in the notebook and simply not
> carried over.

### Model selection (data-driven)
The DS proposed XGBoost and LogisticRegression and concluded the choice between
them was open. Instead of guessing, we **reproduced the comparison under MLflow
tracking** (`challenge/train.py`): both algorithms, on the **top-10 features**,
**with and without class balancing**, using the same `train_test_split`
(`test_size=0.33`, `random_state=42`). No hyper-parameter tuning was performed —
the challenge explicitly does not ask for model improvements.

**Validation results (top-10 features):**

| run | recall (delay·1) | f1 (delay·1) | ROC-AUC | PR-AUC | model size | fit time |
|---|---|---|---|---|---|---|
| `logreg-balanced-top10`  | **0.688** | 0.364 | 0.640 | 0.283 | **0.96 KB** | 0.07 s |
| `xgboost-balanced-top10` | 0.688 | 0.366 | 0.643 | 0.286 | 250 KB | 0.29 s |
| `logreg-plain-top10`     | 0.013 | 0.025 | 0.640 | 0.282 | 0.95 KB | 0.06 s |
| `xgboost-plain-top10`    | 0.006 | 0.012 | 0.642 | 0.287 | 241 KB | 0.21 s |

![Model comparison](assets/model_selection/model_comparison.png)

MLflow tracking UI for the `scl-flight-delay--model-selection` experiment:

![MLflow runs](assets/model_selection/mlflow_runs.png)

**Findings (which confirm the DS's conclusions):**
- Without balancing, both models collapse to predicting "on-time" almost always
  (recall on the delay class ≈ 0) — useless for the airport team's purpose.
- With balancing, both lift delay-recall to **0.688**.
- **LogisticRegression ≈ XGBoost** on every quality metric — including the
  threshold-independent **ROC-AUC (0.640 vs 0.643)** and **PR-AUC (0.283 vs
  0.286)** — so the comparison does not hinge on a particular decision threshold.

**Decision:** the two balanced models are statistically indistinguishable in
quality, but LogisticRegression is **~260× smaller** (0.96 KB vs 250 KB) and
**~4× faster to train**. We therefore pick the lighter, fully interpretable
`LogisticRegression(class_weight='balanced')`
(see [ADR-001](#adr-001--production-model-choice)).

### Experiment tracking & model versioning
- Runs, params, per-class metrics, confusion matrices and the fitted models are
  logged to a **local MLflow** store (`sqlite:///mlflow.db`), experiment
  **`scl-flight-delay--model-selection`**.
- The champion is **registered** following the MLOps naming convention
  `{domain}_{model_type}` → **`flight_delay_logreg`**, with the lifecycle stage
  applied as an **alias** (`@champion`) rather than baked into the name, the
  **semantic version `1.0.0`** as a tag, and traceability tags (`git_commit`,
  `dataset_md5`, `dataset_rows`, `feature_set`). It is exported to
  `challenge/model.joblib` (the exact artifact the API serves). See
  [ADR-006](#adr-006--model-naming--registry-convention).
- Reproduce / inspect:
  ```bash
  uv run python -m challenge.train
  uvx --from "mlflow==2.22.5" mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```

### Tests
`make model-test` → **4 passed**, coverage 92%.

---

## Part II — API

The model is served with **FastAPI** (`challenge/api.py`):

- `GET /health` → `{"status": "OK"}` (liveness probe).
- `POST /predict` accepts a batch:
  ```json
  { "flights": [ { "OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 3 } ] }
  ```
  and returns `{ "predict": [0] }`.

**Input validation** uses Pydantic models with field validators:
- `MES` ∈ 1..12,
- `TIPOVUELO` ∈ {`I`, `N`},
- `OPERA` ∈ the 23 airlines seen in training (`KNOWN_OPERA`).

Any invalid field returns **HTTP 400**. FastAPI/Pydantic raise `422` by default, so
a `RequestValidationError` handler maps validation failures to `400`, as the tests
require.

The `DelayModel` is instantiated once at start-up and loads `challenge/model.joblib`,
so the API serves immediately without retraining. Requests are turned into the
fixed 10-feature frame via `DelayModel.preprocess` and scored with `predict`.

**Tests:** `make api-test` → **4 passed** (the valid case plus the three 400 cases).

## Part III — Cloud deployment

The API is containerized and deployed to **GCP Cloud Run** via **Terraform**
(`infrastructure/`), with the actual rollout driven by the CD pipeline (Part IV).

### Container
- `Dockerfile`: `python:3.10-slim`, installs **only** `requirements.txt` (no
  mlflow/xgboost/ruff) for a lean image and fast cold-starts, runs as a non-root
  user, and binds uvicorn to Cloud Run's `$PORT`. A `.dockerignore` keeps the
  notebook, training script, tests and docs out of the image.
- Verified locally: `docker build` + `docker run` → `/health` and `/predict`
  return 200 with correct predictions.

### Rate limiting & safety
- Per-IP rate limiting via **slowapi**, configured by the `RATE_LIMIT` env var
  (e.g. `120/minute`). It is **disabled by default** so `make stress-test` can
  saturate the API; the real overload/cost guardrail is Cloud Run
  `max_instances` (plus `min_instances = 0` to scale to zero when idle).

### Infrastructure as code (`infrastructure/`)
Terraform provisions: Artifact Registry (Docker), a least-privilege **runtime
service account**, a **deployer service account**, and **Workload Identity
Federation** so GitHub Actions authenticates **keylessly** (no SA-key secret).
Remote state lives in GCS. The Cloud Run service is public
(`allUsers` → `roles/run.invoker`) so the airport team and the stress test can
reach it. See `infrastructure/README.md` for the one-time bootstrap and the list
of GitHub secrets. See [ADR-002](#adr-002--cloud-platform).

### Stress test
Once deployed, the Cloud Run URL goes into `Makefile` line 26 (`STRESS_URL`) and
`make stress-test` runs against the live service.

## Part IV — CI/CD
_To be completed in LT-MLE-006._

---

## Architecture Decision Record (ADR) log

### ADR-001 — Production model choice
**Status:** Accepted · **Context:** Part I model selection.
The DS left the XGBoost-vs-LogisticRegression choice open. An MLflow experiment
(`challenge/train.py`) shows the two **balanced** models are equivalent on the
validation split (delay-class f1 0.364 vs 0.366; recall 0.688 vs 0.688), while
the unbalanced variants are unusable.
**Decision:** use **LogisticRegression(class_weight='balanced')** on the top-10
features.
**Why:** equal predictive performance, but lighter (no XGBoost in the runtime
image → faster Cloud Run cold-starts), fully interpretable via coefficients, and
fewer dependencies to secure/maintain.
**Consequences:** XGBoost remains a dev-only dependency for the comparison; if
future data shifts favor it, the registry + `train.py` make swapping the champion
a one-line change.

### ADR-002 — Cloud platform
**Status:** Accepted.
**Decision:** deploy on **GCP Cloud Run**, provisioned with **Terraform**.
**Why:** pay-per-use with **scale-to-zero** is the cheapest option for a
low-traffic API that must stay live ~1 week (a Compute Engine VM bills 24/7 even
when idle); it provides managed HTTPS and autoscaling out of the box. Terraform
makes the whole environment reproducible and reviewable, and the same config is
reused by CD.
**Consequences:** cold-starts on the first request after idle — mitigated by the
lean image; `max_instances` caps cost; `min_instances` can be raised if cold
starts matter during the demo week.

### ADR-003 — Secrets & CD authentication
**Status:** Accepted.
**Decision:** GitHub Actions authenticates to GCP with **Workload Identity
Federation** (OIDC) impersonating a least-privilege **deployer** service account;
the WIF provider is restricted to this repository. **No service-account key** is
stored. Non-secret config (project id, region, WIF provider name, deployer SA
email, state bucket) is provided as GitHub Actions secrets/variables; nothing is
committed.
**Why:** eliminates long-lived credentials — the highest-value secret to avoid
leaking. A SA-key JSON (`GCP_SA_KEY`) remains a documented fallback only if WIF
is unavailable.
**Consequences:** a one-time bootstrap `terraform apply` (by a project owner)
must create the WIF pool/provider before CD can authenticate.

### ADR-004 — Toolchain: Python 3.10 + uv + ruff
**Status:** Accepted. The provided dependency pins (pandas 1.3.5, numpy 1.22.4,
scikit-learn 1.3.0, pydantic v1) are compatible with **Python 3.10**, which we
pin via `.python-version`. **uv** provides a reproducible env + `uv.lock`;
**ruff** provides linting/formatting. The `requirements*.txt` files are preserved
as the canonical install. `anyio` is pinned to the 3.x line because the 4.x
pytest plugin is incompatible with the pinned `pytest 6.2.5`.

### ADR-005 — Experiment tracking: local MLflow + semantic versioning
**Status:** Accepted. A **file/sqlite-backed local MLflow** store (no hosted
server) is enough for one model: it provides run comparison, plots and a model
registry with **semantic versions** and aliases. MLflow is a **dev-only**
dependency and is not shipped in the serving image.

### ADR-006 — Model naming & registry convention
**Status:** Accepted.
**Decision:** name models `{domain}_{model_type}_v{MAJOR}_{MINOR}_{PATCH}_{stage}`
(e.g. `flight_delay_logreg_v1_0_0_champion`). In the registry the name is
**decoupled from the stage**: register `flight_delay_logreg`, carry the SemVer as
a `semantic_version` tag, and apply the lifecycle stage as an **alias**
(`@champion`, `@challenger`). Traceability metadata (git commit, dataset hash,
row count, feature set) is stored as tags, never hardcoded into filenames.
**Why:** names communicate family/version/status at a glance, prevent ambiguity
during A/B testing and rollback, and let CI/CD auto-increment the patch on
retraining without renaming artifacts.
**Consequences:** the serving artifact (`challenge/model.joblib`) tracks whichever
version holds the `@champion` alias; CD (LT-MLE-006) can bump the patch
automatically. This convention is also captured as a reusable project skill
(`.claude/skills/mlops-model-naming`).
