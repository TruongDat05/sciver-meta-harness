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

- Meta-Harness work is prompt optimization only: a candidate contains exactly the `direct`, `analytical`, `parallel`, and `sequential` templates, and only their text may change.
- Keep prompt variant (`cot`, `meta_cot`, or candidate ID) separate from reasoning method; preserve `cot` exactly and expose a frozen winner only through the additive `meta_cot` variant.
- Preserve each template's existing placeholder interface and preserve claims, context, captions, image order, parser behavior, solver configuration, and labels outside candidate control.
- For SciVer, split by verified paper identity or SHA-256 of the referenced paper JSON, never by claim; never expose gold labels or any final-test information to the proposer.
- Keep proposer and solver tests offline behind fake subprocess and HTTP boundaries. Live solver requests remain explicitly gated, and tests must never invoke Codex recursively.

## SciVer Meta-Harness skill

For one prompt-only evolution iteration, explicitly invoke `$meta-harness`.

Authoritative skill:
`.agents/skills/meta-harness/SKILL.md`

The skill is a read-only proposer:
- it analyzes search and validation history;
- it returns candidate JSON matching the configured output schema;
- it never runs benchmarks or calls the solver;
- it never reads final-test artifacts;
- it never modifies benchmark source code;
- the Python outer loop validates and persists candidate artifacts.

Do not use this skill for implementing Meta-Harness phases or general repository
changes. Use the phase-specific implementation prompts for those tasks.
