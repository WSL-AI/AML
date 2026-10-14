# AML: System Design

Design for taking the current research pipeline (CSV → ETL → notebooks → model) to an operated
system that scores transactions, produces an analyst alert queue, learns from investigator
outcomes and is monitored. It builds on what exists in this repo; section 2 lists what does not.

Status: proposal. Written against commit `4b584c5` plus the uncommitted ETL rewrite on branch `ETL`.

---

## 1. Requirements

### Functional
| # | Requirement |
|---|---|
| F1 | Ingest transactions in time-ordered batches and validate them (schema, ranges, formats). |
| F2 | Compute features for each transaction using only strictly earlier history (no leakage). |
| F3 | Score each transaction with the champion model. |
| F4 | Turn scores into a bounded alert queue for analysts. |
| F5 | Let analysts disposition alerts; store the outcome as a label. |
| F6 | Retrain on a schedule or on drift, evaluate against the champion, promote or reject. |
| F7 | Dashboard: queue, alert volume, precision/recall, drift, model version. |
| F8 | Reproduce any past score: which data, feature state, model and threshold produced it. |

### Non-functional (assumptions, to be confirmed)
| Dimension | Target | Basis |
|---|---|---|
| Volume | ~0.5 M tx/day, ~6 tx/s mean, 60 tx/s peak (10x) | HI-Small: 5.08 M rows over ~10 days |
| Positive rate | ~0.1% | Dataset; real-world rate is likely lower |
| Latency | Micro-batch, alert within 15 min of the transaction | Assumption. No source requirement for real-time blocking exists |
| Alert capacity | Fixed daily budget (e.g. 0.1% ≈ 500/day) | Matches the backtest policy |
| Availability | 99% for scoring; analyst UI may be down independently | Post-hoc monitoring, not payment authorisation |
| Auditability | Every alert traceable to model, features, data hash | AML is regulated; model-risk review expects this |
| Cost | Single machine or small VM; no cluster | Team size and data volume |

### Constraints
- Python 3.13, pandas/pyarrow/joblib for data; XGBoost/LightGBM/CatBoost for models; `uv` lockfile.
- Small team; the pipeline is currently notebook-driven.
- Synthetic data with no transaction ID, no arrival time and no label-arrival time.

**Out of scope:** payment blocking (hard real-time), graph/network-level detection, SAR filing workflow,
multi-tenant bank isolation.

---

## 2. Where the repo is today

| Stage | State | Evidence |
|---|---|---|
| Extract / validate / split | **Solid.** pyarrow reader, hash+row count in manifest, quality report, calendar-day splits, sparse-tail exclusion | `src/aml/data/` |
| Feature pipeline | **Solid.** Stateful `FeaturePipeline` with `fit_transform` / `transform` / `transform_update`; tested against naive recompute and random batching | `features/pipeline.py`, `historical.py` |
| Run publishing | **Good pattern.** Unique run dir, `SUCCESS` marker, atomic `active.json` | `runs.py`, `data/load.py` |
| Categorical encoding | **Gap.** OTE exists (`features/encoding.py`) but ETL does not run it; notebook 02 does | README |
| Training / threshold / evaluation | **Gap.** Notebooks only; `model/train.py` deleted. Only metrics and backtest windows are library code | `model/` |
| Scoring service | **Gap.** `transform_update` exists, but nothing wraps it with encoder + model + threshold | – |
| Alerts, labels, case feedback | **Gap.** Absent | – |
| Monitoring | **Partial.** Drift is a notebook (05) | – |
| Dashboard | **Prototype.** Static `aml_dashboard.html` | untracked |

Current model quality (test, XGBoost): AP 0.374, precision 67.6%, recall 32.0% at 355 alerts. Backtest at a
0.1% daily budget: AP 0.24, recall ~31%. The system around the model must treat it as a **prioritisation
aid with ~⅓ coverage**, not a detector.

---

## 3. High-level design

```
                        ┌────────────────────────── batch plane ──────────────────────────┐
 transactions           │                                                                  │
 (CSV/Parquet/   ─►  [1 Ingest] ─► [2 Validate] ─► [3 Feature pipeline] ─► [4 Scorer] ─► [5 Alert policy]
  core banking)         │  raw Parquet     quality gate     stateful, single      champion        threshold +
                        │  + manifest      + dead letter    writer, checkpointed  bundle          daily cap
                        └──────────────────────┬──────────────────┬─────────────────────┬──────────────┘
                                               │                  │                     │
                                      ┌────────▼────────┐  ┌──────▼───────┐     ┌───────▼────────┐
                                      │ Object store     │  │ State store  │     │ Postgres        │
                                      │ raw/, scores/    │  │ pipeline     │     │ alerts, labels, │
                                      │ (Parquet)        │  │ snapshots    │     │ score_log, runs │
                                      └────────┬────────┘  └──────────────┘     └───────┬────────┘
                                               │                                         │
        ┌─────────────── learning plane ───────▼───────────────┐              ┌─────────▼─────────┐
        │ [6 Label join] ─► [7 Train + encode] ─► [8 Evaluate]  │              │ [9 Analyst UI /   │
        │  (matured labels)   OTE fit on train     backtest +   │◄─ labels ────│  dashboard API]   │
        │                     Optuna, XGBoost      champion gate│              └───────────────────┘
        │                              │                         │
        │                       [10 Model registry: bundles]     │      [11 Monitoring: drift, alert rate,
        └────────────────────────────────────────────────────────┘           score dist., delayed AP] → alerts to team
```

Two planes with different cadences:

- **Batch plane** (every 15 min, or daily to begin with): ingest → score → alert. Deterministic, idempotent.
- **Learning plane** (weekly or on trigger): rebuild datasets from raw + matured labels, train, gate, promote.

Both planes use the **same** `FeaturePipeline` code. That is the main defence against training/serving skew
and it already exists.

### Component responsibilities

| # | Component | Responsibility | Exists |
|---|---|---|---|
| 1 | Ingest | Land source file/batch as immutable raw Parquet; record path, SHA256, row count, `batch_id` | `data/extract.py` (extend) |
| 2 | Validate | Reject or quarantine bad rows; emit quality report; enforce time-order and watermark | `data/validate.py` (extend) |
| 3 | Feature pipeline | `transform_update` on the batch; snapshot state after success | `features/pipeline.py` |
| 4 | Scorer | Apply encoder → model from the champion bundle; write scores | new |
| 5 | Alert policy | Threshold plus per-day cap; dedupe; assign priority | new |
| 6 | Label join | Join analyst dispositions to transactions; apply maturity delay | new |
| 7–8 | Train / evaluate | Port notebooks 02–04 and 06 into `aml-train`, `aml-evaluate` | notebooks → library |
| 9 | Analyst UI | Queue, disposition buttons, explanations | new (prototype exists) |
| 10 | Registry | Versioned bundles + `champion` pointer | pattern in `runs.py` |
| 11 | Monitoring | Drift, volumes, alert rate, delayed metrics | notebook 05 → job |

---

## 4. Deep dive

### 4.1 The model bundle (unit of deployment)

A scorer must never combine a model with the wrong encoder or feature state. Package them together:

```
artifacts/bundles/<bundle_id>/
├── feature_pipeline.joblib     # currency rates + account history up to cutoff T
├── encoder.joblib              # OTE fitted on train only
├── model.json                  # XGBoost
├── feature_schema.json         # ordered names + dtypes; its hash is checked at load time
├── policy.json                 # threshold, daily cap, selection rule, period used
├── lineage.json                # source SHA256s, ETL run_id, training run_id, code git SHA, lockfile hash
├── metrics.json                # validation, threshold-period, test, backtest
└── SUCCESS
artifacts/bundles/champion.json # atomic pointer, same mechanism as active.json
```

Rules:
- Scorer refuses to start if `feature_schema.json` hash ≠ the encoder's output schema or the model's expected inputs.
- `joblib` is pickle: loading is arbitrary code execution. Bundles are only loaded from the registry location,
  and `lineage.json` stores a SHA256 of every file that is verified before load.
- The champion's feature state is advanced by the scorer after each batch; the **bundle itself is immutable**.
  State lives in a separate checkpoint (4.3), keyed by `bundle_id`.

### 4.2 Data model

Parquet (object store, append-only, partitioned by `date`):

| Dataset | Key columns |
|---|---|
| `raw/` | `batch_id`, `row_in_batch`, original 11 columns |
| `scores/` | `tx_id`, `batch_id`, `bundle_id`, `score`, `scored_at` |
| `features/` (optional, for audit/retrain) | `tx_id`, feature columns (or recomputed from raw) |

Postgres (small, transactional):

```sql
transaction_ref(tx_id PK, batch_id, row_in_batch, ts, from_bank, from_acct, to_bank, to_acct, amount_usd)
alert(alert_id PK, tx_id FK UNIQUE, bundle_id, score, rank_in_day, status, created_at)
  -- status: new | assigned | confirmed | false_positive | escalated
disposition(disposition_id PK, alert_id FK, analyst, outcome, reason, decided_at)  -- append-only
batch_run(batch_id PK, source_sha256, row_count, state_in, state_out, bundle_id, status, started, finished)
model_bundle(bundle_id PK, trained_at, cutoff_ts, status, metrics_json)  -- status: candidate|champion|retired
```

**Transaction identity.** The source has no transaction ID, and duplicates are legitimate. Define
`tx_id = sha256(batch_id, row_in_batch)` (deterministic for a fixed source), never a content hash, so genuine
duplicate payments stay distinct. Real core-banking feeds should supply their own ID; keep it in the same column.

**Entity key.** An account is `(bank, account)`, not `account` (`data/keys.py`). All joins and shards use this pair.

### 4.3 Stateful scoring: ordering, checkpointing, recovery

The history builder is a **single-writer, time-ordered state machine** (≈91 MB on HI-Small). Its correctness
guarantee ("only strictly earlier events") is what the test-suite verifies. The scorer must preserve it:

1. **Watermark.** Accept a batch only if `min(ts) ≥ state.max_ts`. Late data goes to a quarantine dataset and is
   applied at the next retrain (it changes history, so it cannot be silently folded in).
2. **Sort** the batch by timestamp (stable) – already done by `validate_raw`.
3. **Process:** `transform_update` → encode → score → apply policy → write outputs.
4. **Commit order:** write scores/alerts → write `state_out` snapshot to a new path → insert `batch_run` row as
   `success` → atomically move `state/current.json`. A crash before the last step leaves the old pointer; the
   next run re-processes the same `batch_id` and produces identical output (idempotent upsert on `tx_id`).
5. **Recovery:** if a snapshot is lost or corrupt, rebuild by replaying `raw/` from the bundle's cutoff. Replay is
   cheap (≈30 s per 5 M rows) – this is why raw Parquet is the source of truth and state is a cache.

Snapshots: keep the last N (e.g. 7) plus one per day; ~100 MB each.

### 4.4 Alert policy

Two policies exist in the repo and they are not equivalent:

| Policy | Where | Behaviour |
|---|---|---|
| Max-F1 threshold (0.2988) | notebook 04 | Variable alert count; 355 alerts on 2 test days |
| Daily top-k at 0.1% | backtest | Fixed count; needs the whole day, so **not computable online** |

Production design: **threshold with a cap**.
- Threshold `τ` is set from the threshold period so that the expected alerts/day ≈ analyst capacity, not to maximise F1.
- A daily cap `K` bounds overflow: after K alerts, only transactions with `score ≥ τ_high` are added.
- Alerts are ranked by score within the day; the UI shows the queue in that order.
- `τ` is re-estimated at every retrain and recorded in `policy.json`. Drift in the score distribution
  (monitoring, 4.7) can shift alert volume without any model change, so alert rate is an operational metric
  with its own alarm.

Optional later: group alerts per account/day so an analyst reviews one account, not 30 payments.

### 4.5 Labels and the feedback loop (the biggest risk)

The offline experiments assume labels are known immediately (`README`: "Предположение OTE о немедленной доступности
меток"). In operation:

- Only **alerted** transactions receive a disposition. Non-alerted positives are never discovered unless an
  external event (SAR, law-enforcement, customer complaint) reveals them. Training on dispositions alone teaches
  the model its own past decisions (selection bias).
- Dispositions take days to weeks (**label lag**).

Design:
1. **Label maturity.** A transaction is trainable as negative only after `L` days (e.g. 30) without a positive
   outcome; positives enter as they are confirmed.
2. **Encoder uses matured labels only.** Ordered target encoding at serving time must see only labels that
   would have been known at that moment. The training pipeline must replay with the same lag, otherwise offline
   metrics are optimistic. This is a required change to the current OTE assumption.
3. **Exploration sample.** Reserve a small random slice of the queue (e.g. 5–10% of K, drawn from below-threshold
   transactions) for review. It is the only unbiased sample of what the model misses, and the only honest way to
   estimate recall in production.
4. Store `alert_source ∈ {model, exploration}` so training and evaluation can reweight.

Without (3) the system can report precision, but never recall.

### 4.6 Training and promotion

`aml-train` (new CLI; code lifted from notebooks 02, 03):

```
raw + matured labels ─► split (existing calendar-day logic) ─► FeaturePipeline.fit_transform
  ─► OTE fit on train ─► Optuna search (bounded trials) ─► XGBoost/LightGBM/CatBoost ─► choose by val AP
  ─► threshold period: set τ for target alerts/day ─► test + rolling backtest ─► candidate bundle
```

Promotion gate (all must hold, recorded in `metrics.json`):
- test AP ≥ champion AP on the **same** window, with a stated tolerance (the backtest shows AP std of 0.007–0.05 over
  3 days, so a few-point gain is not significant; use a minimum effect size and, ideally, paired bootstrap).
- Recall at the production alert budget does not regress.
- Schema, lineage and `SUCCESS` checks pass; unit and equivalence tests green.
- Score-distribution shift against the champion on the same data is within limits (alert volume predictable).

Then **shadow** the candidate for ≥ 3 days (score everything, no alerts), compare, and flip `champion.json`.
Rollback = flip the pointer back; state for the old bundle is retained for N days.

### 4.7 Monitoring

| Signal | Source | Alarm |
|---|---|---|
| Batch freshness / failure | `batch_run` | no successful batch in 2× cadence |
| Data quality | validation report | rejected rows > 0.5%, schema change, unknown payment format |
| Late-data rate | watermark rejects | > 0.1% of rows |
| Feature drift | PSI/KS per feature vs train (notebook 05 logic) | PSI > 0.2 on top-importance features |
| Score distribution | scores/ daily | alert rate outside ±30% of expected |
| Currency rates | converter | pair rate vs fitted median deviates beyond tolerance |
| Analyst outcomes | `disposition` | precision of the queue (7-day rolling), time-to-disposition |
| Delayed performance | matured labels | AP/recall on exploration sample vs bundle metrics |
| Resource | process | peak RSS (6 GB on HI-Small ETL), batch duration |

Retrain triggers: schedule (weekly), feature drift alarm, or delayed AP below floor. A trigger creates a candidate;
it never auto-promotes.

### 4.8 Interfaces

CLI (first), then HTTP for the UI:

```
aml-etl      --source … --output …                  # exists
aml-train    --etl-run … --out bundles/             # new
aml-score    --batch raw/2026-10-14.parquet         # new: runs 1–5, idempotent on batch_id
aml-promote  --bundle <id> [--rollback]             # new
aml-monitor  --date 2026-10-14                      # new
```

Analyst API (read-heavy, tiny):

```
GET  /alerts?status=new&date=…&limit=…          queue, ordered by rank
GET  /alerts/{id}                                transaction, score, top features, account history summary
POST /alerts/{id}/disposition {outcome, reason}  append-only; 409 if already final
GET  /metrics/summary?from=…&to=…                dashboard tiles
```

Per-alert explanations: SHAP for tree models, computed **only for alerted rows** (≈500/day), so cost is negligible.
Store top-5 contributions in `alert`.

---

## 5. Scale and reliability

### Load estimate
- Mean 6 tx/s; peak 60 tx/s. One daily batch of 0.5 M rows runs in seconds; the history update and scoring are
  vectorised, so a 15-minute micro-batch (~5k rows) is dominated by state load/save unless state stays resident.
- **Keep the scorer as a long-lived process** holding the state in memory; snapshot after each batch (≈100 MB,
  sub-second to local SSD). Do not `joblib.load` per batch.
- Memory: state is bounded by accounts + last-24 h events. ~100 MB at this scale; 100× volume ≈ 10 GB.
- ETL/training peak is ~6 GB, driven by pandas string columns (35 M Python strings), not features. Cast ID columns to
  `category` or dictionary-encoded Arrow strings before it matters.

### Scaling path (only when needed)
| Trigger | Step |
|---|---|
| Training > 1 h | Pre-encode IDs to integers in ingest; move split/feature joins to Polars or DuckDB; GPU XGBoost |
| State > RAM of one box | Shard by account **key** – but sender features also read the *receiver's* history, so a transaction touches two shards. Options: replicate receiver-side aggregates, or two-hop lookup. This is the hardest scaling step; avoid by staying single-node as long as possible |
| < 1 min latency | Replace file batches with a queue (Kafka/Redpanda), keep the same `transform_update` core behind a consumer |

### Failure modes
| Failure | Effect | Handling |
|---|---|---|
| Scorer crash mid-batch | Partial outputs | Commit order in 4.3; re-run is idempotent |
| State snapshot corrupt | Features wrong silently | SHA256 in `batch_run`; replay from raw |
| Late/out-of-order events | Leakage invariant broken | Watermark + quarantine |
| Bad currency rate (new currency) | Run stops (by design) | Quarantine rows with unknown currency; alert on-call; do not default to rate 1 |
| Source re-delivered | Double scoring | `batch_id` + source SHA256 uniqueness |
| Model/feature mismatch | Garbage scores | Schema-hash check at load; bundle is atomic |
| Alert flood | Analysts overwhelmed | Daily cap `K`; alert-rate alarm |
| DB down | UI unavailable | Scoring writes Parquet first; alerts back-filled to DB when it returns |

### Redundancy
Single node with daily backup of `raw/`, `scores/`, Postgres and the latest state snapshot. Scoring is stateless
given raw + a snapshot, so a cold standby can take over after a replay. Active/active is unjustified at this scale.

---

## 6. Security and compliance
- PII: real account numbers must be tokenised at ingest; models and dashboards use the token. The synthetic
  dataset needs none, but design the join key so that it can be swapped.
- Access: analysts see alerts and account history; only the pipeline role writes `scores/` and `raw/`; promotion
  requires a second person.
- Artifacts: signed/hashed bundles (see 4.1); no network-loaded pickles.
- Audit: `score_log` + `lineage.json` answer "why was this flagged / not flagged on date X". Retain per policy.
- Fairness/model-risk: document features used; exclude any proxy for protected attributes if real data is used.

---

## 7. Trade-off analysis

| Decision | Chosen | Alternative | Why | Cost of choice |
|---|---|---|---|---|
| Micro-batch vs streaming | Micro-batch, 15 min | Kafka + streaming consumer | No blocking requirement; matches daily alert budget; reuses tested batch code | Detection latency up to 15 min; revisit if blocking is wanted |
| Single-writer state | One scorer process | Sharded feature store (Redis/Feast) | 100 MB state; correctness already tested; no infra | Vertical scaling only; no hot standby without replay |
| State as cache, raw as truth | Replay from Parquet | Persist state in DB | Rebuild takes seconds-minutes; removes a consistency problem | Slow recovery if raw grows large (add periodic full snapshots) |
| Parquet + Postgres | Two stores | Parquet only / warehouse | Analytics volume in Parquet, transactional alert workflow in Postgres | Two systems to back up |
| Threshold + cap | Fixed capacity | Pure max-F1 | Capacity is the real constraint; F1 ignores analyst cost | Threshold re-calibration on each retrain |
| Exploration slice | 5–10% of queue | None | Only way to measure recall and break selection bias | Spends analyst time on likely negatives |
| XGBoost as champion | Keep | LightGBM / ensemble | Beats LightGBM on all 3 backtest windows | Evidence is 3 correlated dev days, not independent |
| Notebooks → library + CLI | Port | Keep notebooks as pipeline | Reproducibility, CI, scheduling | Upfront refactor; keep notebooks for exploration only |
| Pickle bundles | `joblib` + hashes | ONNX / native model formats for the model, JSON for state | Fast to ship; state is a Python object | Pickle risk, Python-version pinning. Model alone can already be exported as `xgboost` JSON |
| Orchestration | cron/systemd (or Prefect later) | Airflow | Two or three jobs; no DAG complexity | Manual retry visibility; adopt a scheduler when jobs > ~5 |

---

## 8. Roadmap

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 – finish the offline loop | Move training (notebooks 02–04) into `aml.model.train`, `aml-train` CLI; ETL emits the encoder; restore `model/train.py` as a real module; commit the ETL rewrite | `aml-etl` → `aml-train` → bundle with no notebook, covered by an end-to-end test |
| 1 – scoring | `aml-score` on a bundle, watermark, idempotent batches, state snapshots, schema-hash checks | Replay of the test period equals offline test scores bit-for-bit |
| 2 – alerts | Postgres schema, alert policy with cap, queue API | Daily queue ≤ K; same input gives same alerts |
| 3 – feedback | Disposition capture, exploration slice, label-lag-aware OTE and training | Offline metrics re-run with lagged labels; documented drop vs. immediate labels |
| 4 – operations | Monitoring job, drift alarms, shadow mode, promotion gate, rollback | A drill: bad candidate is rejected; rollback restored in < 5 min |
| 5 – scale/UX | Account-level grouping, explanations, dashboard from live data | Analyst usability review |

Phases 0 and 1 are the highest value per effort: they remove the notebook dependency and the training/serving gap.
Phase 3 is the one most likely to change the model's reported numbers.

---

## 9. What to revisit as the system grows

1. **Streaming and latency.** If the business wants to hold or block payments, micro-batch is wrong; move to a
   consumer with a hard latency budget and a precomputed per-account feature cache.
2. **State sharding.** Receiver-side features make account sharding non-trivial. Decide the replication strategy
   before the state exceeds one machine's RAM.
3. **Graph features.** Laundering is structural (fan-out/fan-in, cycles, chains). Current features are
   per-account windows; multi-hop or GNN features would likely raise recall but need a graph store and a different
   freshness story.
4. **Label quality.** Re-estimate everything once real dispositions exist; the synthetic generator's labels,
   including its end-of-simulation positive tail, are not representative.
5. **Separate models.** One model for all payment formats/currencies may be inferior to segment-specific models or
   rules plus ML. Look at `reports/full/error_analysis` by segment.
6. **Registry and orchestration.** Replace directory pointers with MLflow (or similar) and a scheduler once there
   are multiple models, environments or teams.
7. **Evaluation statistics.** The three-window backtest cannot separate models at the 0.01–0.05 AP level. Longer
   history (HI-Medium/Large) and bootstrap intervals are needed before using small gains to decide promotions.

## 10. Open questions
- Is the real use case alerting after the fact (assumed) or authorising/blocking payments?
- What is analyst capacity (alerts/day) and cost per review? This sets `K`, and the economically right threshold.
- Which fields exist at scoring time? The dataset has no arrival time, so availability of `Amount Received` and
  `Receiving Currency` at decision time is assumed (already noted in `docs/ETL.md`).
- Are dispositions available in a system we can read, and with what delay?
- Retention and privacy rules for transaction data and scores.
