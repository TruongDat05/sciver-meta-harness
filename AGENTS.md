# AGENTS.md

These rules apply to the entire repository. Keep all work aligned with the project goal and treat the privacy, security, and API-safety rules below as hard requirements. When requirements appear to conflict, prefer the interpretation that prevents disclosure, network access, prompt leakage, and compatibility regressions.

## Project goal

Extend SciVer with a generic OpenAI-compatible remote API provider while preserving the existing system. The implementation must support these evaluation datasets:

- SciVer
- SciAtomicBench
- MuSciClaims
- SciClaimEval

It must support these models:

- Qwen2.5-VL-7B-Instruct
- gemma-4-31B-it
- gemma-4-26B-A4B-it
- gemma-3-27b-it

## Privacy and security

- Never include a company or remote provider name in source code, documentation, file names, class names, CLI arguments, logs, or tests. Compatibility-protocol and model identifiers listed in this file describe technical interfaces and supported models; they must not be used to identify or advertise a remote provider.
- Use only generic, provider-neutral names such as `remote`, `remote_api`, and `api_provider`.
- Read credentials only from the `API_KEY` environment variable.
- Read the endpoint only from the `API_URL` environment variable.
- Never hardcode an API key or endpoint.
- Never log, print, serialize into diagnostics, or expose the API key, an `Authorization` header, or image base64 data.
- Never use realistic-looking secrets in tests, fixtures, examples, or documentation. Use unmistakably fake placeholders.

## API safety

- Unit and integration tests must never access the network. Mock every HTTP call at the request boundary.
- Live requests are opt-in only and require an explicit, provider-neutral CLI flag. The default path must remain offline.
- Importing any module must never trigger an API request or other remote side effect.
- Never download model weights. Remote-model support must not introduce code paths that fetch weights implicitly or explicitly.
- Tests for live-request gating must mock the request and verify that no call occurs without the explicit flag.

## Compatibility and data integrity

- Preserve all existing providers and existing CLI behavior. New options must be additive, provider-neutral, and backward compatible.
- Preserve the original SciVer prompts and reasoning methods. Do not rewrite, simplify, or silently substitute them when adding remote support.
- Preserve image ordering end to end, including dataset loading, message construction, serialization, and request dispatch.
- Never guess dataset field mappings. Inspect representative samples or a schema report before implementing a mapping; return a clear error when the available schema is insufficient.
- Ground-truth labels must never be included in model prompts, message payloads, demonstrations, or other model-visible input.

## Development rules

- Keep each change narrowly scoped to the requested behavior and avoid unrelated cleanup.
- Add or update tests for every new behavior, including error paths and safety gates.
- Reuse established repository patterns before introducing new abstractions or dependencies.
- Return clear, actionable errors for invalid configuration and invalid input. Do not include secrets or sensitive payload data in error messages.
- Do not commit or push on behalf of the user.
- Do not modify unrelated files.

## Verification

Run these commands before declaring implementation work complete:

```text
python -m compileall -q .
pytest -q
git diff --check
```

Also inspect the test setup and results to confirm that all HTTP calls were mocked and no test made a live request. Review the final diff for secret material, provider identity, image base64 data, accidental prompt changes, image reordering, ground-truth leakage, implicit model downloads, and unrelated file changes.

## Definition of done

Work is complete only when all of the following are true:

- Relevant tests pass.
- No test performs a live request.
- No secret or remote provider identity is exposed.
- Existing providers, CLI behavior, prompts, reasoning methods, and other existing behavior remain compatible.
- No unrelated files are changed.
- `AGENTS.md` has been reviewed for conflicting, vague, or provider-specific instructions.

## Meta-Harness prompt search

The canonical detailed plan is `docs/IMPLEMENTATION_PLAN.md`. For the
`sciver_full_search_v3` protocol, the following are permanent invariants:

The earlier `sciver_full_search_v2` 750/1250 design was abandoned before any
official live experiment and is not an active or compatible protocol.

- The repository is the complete reusable experiment engine. It owns
  deterministic preparation, solver/API calls, caching, retries, concurrency,
  SEARCH evaluation and orchestration, proposing, ranking, patience,
  checkpoint/resume, winner freezing, paired FINAL execution, persistence,
  and summaries. A final server Jupyter Notebook is only a thin launcher and
  operator layer; it must call supported repository entry points and must not
  reimplement any experiment logic.
- A notebook may supply `API_URL` and `API_KEY` only at runtime, including from
  the server environment or hidden interactive input for the key. Never commit
  or persist those values in a notebook, repository artifact, or result.
- Implementation milestones, tests, dry runs, and notebook development do not
  authorize a real experiment. Full live SEARCH and paired FINAL execution may
  occur only after the repository and notebook are finalized and the user has
  explicitly authorized that live scope.
- Meta-Harness is prompt-only optimization. A candidate has exactly the
  `direct`, `analytical`, `parallel`, and `sequential` templates; only their
  text may change. Codex proposes prompt text only; trusted Python validates,
  evaluates, scores, checkpoints, and selects.
- Keep prompt variant (`cot`, `meta_cot`, or candidate ID) distinct from
  reasoning method. Canonical `cot`, placeholder interfaces, answer format,
  parser, labels, claims, context, captions, image number/order, and solver
  generation semantics are frozen. Expose a winner only as additive
  `meta_cot`.
- The fixed solver is Qwen/Qwen3.5-35B-A3B through the generic compatible HTTP
  boundary. Read its endpoint only from `API_URL` and credentials only from
  `API_KEY`; never persist either.
- Select exactly 2,000 SciVer samples with seed 42: exactly 1000 paper-disjoint
  SEARCH samples and exactly 1,000 paper-disjoint FINAL samples. Membership is
  deterministic and independent of labels, predictions, difficulty, and
  baseline errors. Fail if exact counts and paper exclusivity cannot be met.
- Evaluate canonical P0 and every valid candidate on the same complete 1000
  SEARCH records. There is no hard subset/objective, smoke scoring gate,
  promotion, or protected validation. Representative SEARCH errors may be
  summarized only as proposer feedback, never as an evaluation subset.
- Propose one candidate per iteration, with 15 minimum and 40 maximum
  iterations, patience 8, and at most three attempts for an invalid or
  duplicate proposal. Rank P0 and candidates by SEARCH Macro-F1 descending,
  SEARCH Accuracy descending, prompt SHA-256 ascending, then candidate ID
  ascending. A hash/ID-only tie does not reset patience.
- FINAL is isolated from proposal, SEARCH ranking, patience, and prompt
  modification. Freeze the offline winner before an explicit FINAL execution;
  then evaluate frozen P0 and frozen P* exactly once each on the identical
  1,000 FINAL IDs. FINAL results never modify SEARCH state or the winner.
- Support deterministic checkpoint/resume, retry handling, concurrency, and a
  safe SEARCH cache. Tests must stay offline and mock both HTTP and Codex
  subprocess boundaries; tests must never recursively invoke Codex.
- Legacy hard-search, staged, protected-validation, retry/reparse, and
  transfer modules were removed in the authorized Milestone 7 cleanup. Do not
  reintroduce or route current workflows through those designs.
- Maximum logical workload is 41,000 SEARCH evaluations (1,000 P0 plus 40,000
  candidate evaluations), 2,000 FINAL evaluations (1,000 each for P0 and P*),
  and 43,000 overall. Transport retries are excluded from these counts.

## SciVer Meta-Harness skill

For one prompt-only evolution iteration, explicitly invoke `$meta-harness`.
The skill is a read-only proposer: it never calls a solver, benchmark, network
service, or another Codex process; it never reads FINAL artifacts; and it never
modifies files. When generic skill guidance describes hard-search, promotion,
or protected validation, the `sciver_full_search_v3` contract above controls
this repository.
