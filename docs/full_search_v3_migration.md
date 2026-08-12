# Full-Search v3 migration note

Milestone 7 intentionally removes the obsolete staged Meta-Harness workflow.
Archived v1/v2/staged run layouts are no longer loadable or resumable by this
checkout.

Removed entry points include `prepare_meta_harness_data.py`,
`run_meta_harness.py`, `finalize_meta_harness.py`,
`retry_meta_harness_failures.py`, `reparse_meta_harness.py`,
`smoke_meta_harness_proposer.py`, `export_meta_cot.py`, and
`run_meta_cot_transfer.py`. Their staged configuration files, candidate-batch
schema, compatibility loaders, tests, and historical design documents were
removed with them.

Use `notebooks/sciver_full_search_v3_server.ipynb` as the canonical launcher.
For terminal operation, use `scripts/run_full_search_v3_server.py`; it is a
thin wrapper over `meta_harness.full_search_v3_server`. The supported stages
are deterministic preparation, SEARCH preflight, isolated SMOKE with a
compatible create-once receipt, production SEARCH execution/status, immutable
winner freeze, and separately authorized paired FINAL preflight/execution/
status. Existing user datasets and run directories are not migrated or
deleted by this cleanup.
