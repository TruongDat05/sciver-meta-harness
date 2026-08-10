# Meta-Harness requirement trace

Authoritative design source: [`pipeline_meta_harness.md`](../pipeline_meta_harness.md).
This trace covers the implemented runtime requirements; phase prompts and
historical proposed file layouts are design guidance rather than executable
interfaces.

| Pipeline requirement | Implementation | Offline verification |
| --- | --- | --- |
| Prompt-only search; no tuning, LoRA, weight/model/evidence/parser changes | `meta_harness/schemas.py`, `meta_harness/prompt_family.py`; declarative candidate JSON only | `test_candidate_schema.py`, `test_prompt_family.py`, `test_end_to_end_offline.py` |
| Exactly four templates | `TEMPLATE_KEYS` and exact-key validation | `test_candidate_schema.py`, `test_prompt_family.py` |
| Preserve template placeholders and yes/no parser contract | `REQUIRED_PLACEHOLDERS`, candidate static validation | `test_candidate_schema.py`, `test_answer_parser.py` |
| Preserve original `cot` bytes | `utils/constant.py` remains canonical; `meta_harness/baseline.py` validates canonical export | `test_baseline_equivalence.py` |
| `baseline_cot.json` exactly matches `COT_PROMPT` | canonical byte comparison in `validate_baseline_snapshot` | `test_baseline_equivalence.py` |
| Separate prompt variant from reasoning method | `utils/prompt_registry.py`, result `prompt_variant`, unchanged four reasoning methods | `test_prompt_family.py`, `test_finalize.py`, `test_metrics.py` |
| Fixed Qwen search model and omitted legacy generation overrides | `MetaHarnessConfig`, orchestrator/evaluator validation | `test_config.py`, `test_orchestrator.py`, `test_evaluator.py` |
| Paper-level deterministic 20/20/60 split | `split_manager.py`; verified `paper_id` or complete paper JSON SHA-256 | `test_split_manager.py` |
| No split overlap and immutable hashes | manifest verification, atomic create-only save | `test_split_manager.py`, `test_full_pipeline_offline.py` |
| Search receives exactly validation records | run CLI loader and orchestrator exact-membership checks | `test_orchestrator.py`, `test_full_pipeline_offline.py` |
| Final-test records/IDs never reach proposal or search | validation-only materialization; aggregate-only proposer envelope | `test_split_manager.py`, `test_codex_cli.py`, `test_orchestrator.py` |
| Codex CLI is the only runtime subprocess | `meta_harness/proposer/codex_cli.py` is the only Meta-Harness subprocess import | `test_codex_cli.py`; repository search audit |
| Codex cannot edit repository code | isolated temporary cwd, read-only sandbox, ignored user config/rules, structured output | `test_codex_cli.py` plus local `codex exec --help` |
| Solver credentials do not enter Codex | minimal allowlisted child environment; explicit `API_KEY`/`API_URL` removal | `test_codex_cli.py` |
| Proposer does not need solver/API access | proposer uses cached Codex login and receives aggregate sanitized history only | `test_codex_cli.py`, `test_orchestrator.py` |
| Proposer output is structured and candidate hashes are trusted | JSON Schema validates text/metadata; Python derives SHA-256 after parsing without changing text | `test_codex_cli.py`, `test_candidate_schema.py` |
| Candidate IDs/parents/content are valid and unique | schema and store validation | `test_candidate_schema.py`, `test_candidate_store.py`, `test_codex_cli.py` |
| Candidate storage immutable, hashed, atomic | canonical create-once candidate files, registry hashes, atomic links/replaces | `test_candidate_store.py` |
| Candidate locks recover after process exit | advisory `flock`; stale marker files do not block | `test_candidate_store.py` |
| Existing request construction and image order | evaluator calls `prepare_remote_requests`; only prompt family differs | `test_baseline_equivalence.py`, `test_evaluator.py`, `test_end_to_end_offline.py` |
| Existing parser, metrics, and result writer | evaluator imports production functions directly | `test_evaluator.py`, `test_metrics.py`, `test_result_writer.py` |
| Gold-only fields absent from solver messages | evaluator detaches label/rationale fields before request preparation | `test_evaluator.py`, `test_end_to_end_offline.py` |
| API, parse, and invalid-input failures remain distinct | evaluator/result writer/metrics failure statuses | `test_evaluator.py`, `test_metrics.py`, `test_result_writer.py` |
| No network in tests | suite autouse socket/HTTP denial and fake boundaries | `tests/conftest.py`, `test_full_pipeline_offline.py` |
| Hard budgets | `BudgetLimits` and transition-boundary accounting | `test_orchestrator.py` |
| Deterministic Pareto frontier | Macro-F1/solver-call/token/latency objectives plus candidate-ID stability | `test_orchestrator.py` |
| Early stopping ignores failed/partial iterations | completed-iteration Macro-F1 delta policy | `test_orchestrator.py` |
| Failure isolation and recovery | per-candidate failure records; durable proposal/evaluation checkpoints | `test_orchestrator.py`, `test_full_pipeline_offline.py` |
| Atomic resume without duplicate completed work | result identities, state checkpoints, config/split hashes | `test_orchestrator.py`, `test_full_pipeline_offline.py` |
| Lock one run writer | nonblocking advisory run lock | `test_orchestrator.py` |
| Rank only full-coverage, zero-API-failure candidates | evaluator eligibility plus orchestrator/finalizer independent rechecks | `test_evaluator.py`, `test_orchestrator.py`, `test_finalize.py` |
| Global winner across all iterations | finalizer scans all evaluated rankable candidates | `test_finalize.py` |
| Winner order: Macro-F1, accuracy, tokens, deterministic ID | `select_global_best` and frozen `selection_rule` | `test_finalize.py` |
| Freeze before any final-test use | final CLI freezes before loading final records; artifact hashes code/config/split/prompt | `test_finalize.py` |
| Frozen run cannot evolve | orchestrator rejects resume when `frozen_winner.json` exists | `test_finalize.py` |
| Final test once per prompt/model | immutable manifest and completion receipt for each `cot`/`meta_cot` identity | `test_finalize.py` |
| Qwen final compares unchanged `cot` and frozen `meta_cot` | `execute_final_test_pair`; final CLI uses it | `test_finalize.py` |
| Exact frozen transfer to three Gemma models | `execute_transfer_matrix`; transfer CLI has fixed model tuple | `test_finalize.py`, `test_full_pipeline_offline.py` |
| No per-model prompt adaptation or search feedback | transfer loads one frozen artifact and has no proposer boundary | `test_finalize.py` |
| Additive `meta_cot` without overwriting `cot` | explicit verified registration and immutable export | `test_finalize.py`, `test_prompt_family.py` |
| Live access explicitly gated | `--live-api`; final paths additionally require `--confirm-final-test` | `test_orchestrator.py`, `test_finalize.py`, `test_cli.py` |
| Secrets/base64 sanitized from persisted diagnostics | candidate rejection, proposer/orchestrator redaction, result writer recursion | `test_codex_cli.py`, `test_candidate_store.py`, `test_result_writer.py` |
| Backward-compatible existing benchmark paths | legacy routing remains default; new prompt variant and provider flags additive | `test_cli.py`, `test_benchmark_workflow.py`, `test_baseline_equivalence.py` |

## Deliberately unresolved design detail

The pipeline proposes a smoke/screen/full-validation funnel but gives only an
example screening threshold and does not fix an exploration quota. The current
implementation evaluates every valid candidate on the complete fixed
validation split. This is more expensive but does not change winner data,
metrics, or leakage boundaries. A staged funnel should not be enabled until
the threshold and exploration quota are pre-registered explicitly; inventing
them during this audit would change the experiment design.
