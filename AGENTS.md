# AGENTS.md

Operating guide for autonomously maintaining **FastText.Net** with agentic workflows.
This file defines what an agent may do, how it verifies its work, and how releases are
cut. Humans gate publishing; everything up to that point can be agent-driven.

## What this project is

A managed, dependency-free C# port of fastText **inference** (model loading + `predict`).
Unofficial, not affiliated with Meta. Training is out of scope — do not add it.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/FastText.Net/` | The library (the only packable project). |
| `tests/FastText.Net.Tests/` | xUnit correctness tests. **This is the behavioral contract.** |
| `bench/FastText.Net.Benchmarks/` | BenchmarkDotNet performance suite. |
| `.github/workflows/ci.yml` | Build, test, pack, and tag-gated publish. |
| `global.json` | SDK floor (rolls forward, prerelease allowed). |

## Verification (how an agent knows "green = correct")

- `dotnet test FastText.Net.sln -c Release` must pass with **0 skipped**.
- The `DownloadFastTextModel` target in the test project fetches `lid.176.ftz` and the
  `LanguageIdentificationTests` assert byte-exact output against `oracle_lid.json`. If
  these tests *skip*, the model failed to download — treat that as a failure, not a pass.
- `SyntheticModelTests` exercise the quantized / hierarchical-softmax paths with a
  checked-in synthetic model and need no network.
- **The C# tests are the textual contract for behavior.** Do not add a separate public
  API snapshot tool; express behavioral expectations as tests. When changing observable
  behavior, update the tests in the same PR.

## Issue intake → PR loop

An agent may pick up an issue when it can:

1. **Reproduce** it as a failing test under `tests/FastText.Net.Tests/`.
2. **Fix** the library so that test passes, without regressing the existing suite.
3. **Describe** the user-facing impact clearly in the PR title/description, which feeds
   GitHub's auto-generated release notes.

Rules:

- Every fix or feature ships with a test that fails before and passes after.
- Agents **open PRs**; they do **not** merge their own PRs and do **not** publish.
- Keep changes surgical; do not introduce training, native deps, or P/Invoke.
- Comments describe the final state of the code, not the diff.

## Dependency hygiene

- Dependabot opens grouped weekly PRs for NuGet and GitHub Actions.
- A dependency PR is acceptable when `dotnet test -c Release` is green (0 skipped).
- For `System.Numerics.Tensors` and SDK bumps, also sanity-check performance: run
  `dotnet run -c Release --project bench/FastText.Net.Benchmarks` and compare against
  `bench/results.md`. Flag (do not silently merge) a clear regression.

## Release runbook

Releases are cut by pushing a `vX.Y.Z` tag, which triggers the publish step in
`ci.yml`. An agent may prepare a release PR; a human approves and tags/publishes.

To prepare a release:

1. Decide the version per SemVer (public API changes ⇒ minor/major).
2. In `src/FastText.Net/FastText.Net.csproj`:
   - Set `<Version>` to the new version.
   - Set `<PackageValidationBaselineVersion>` to the **previous** released version so
     package validation diffs against it.
3. Open the release PR. After human review and merge, a human pushes the `vX.Y.Z` tag and
   publishes a GitHub release. Release notes are GitHub's auto-generated notes from merged
   PRs, so keep PR titles descriptive and user-facing. CI builds, validates the package,
   and publishes to NuGet.org.

## Guardrails (hard limits — never cross)

- **Never** modify, print, exfiltrate, or commit `eng/Open.snk` (the signing key) or the
  `NUGET_API_KEY` secret.
- **Never** push directly to `main` or force-push; **never** rewrite published history.
- **Never** publish to NuGet.org or push tags — that is a human action.
- **Never** create, close, or edit releases/PRs outside the agent's own working PR.
