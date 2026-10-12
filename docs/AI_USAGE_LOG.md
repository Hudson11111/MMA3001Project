# AI Usage Log

Raw material for the report's AI-use reflection (brief §7). The brief asks for:
tools used, what for, approximate contribution, why, how output was checked,
which decisions stayed yours, AI errors/unhelpful suggestions, and the effect on
your understanding.

**How to use this file:** one row per working session. Claude fills in the factual
columns (what it did). **You** fill in "Your steering" and "Your verification" —
the rubric rewards evidence that you were driving the process, so record the
times you redirected, rejected or corrected the AI.

## Tools
| Tool | Used for |
|---|---|
| Claude (Anthropic, Opus 5.5) via claude.ai with code/GitHub access | Planning, code, tests, documentation drafts |
| Google Colab | Running notebooks on GPU (no AI features used unless noted) |

## Session log

| # | Date | Your request / steering | What Claude did | Your verification | AI errors or rejected suggestions |
|---|---|---|---|---|---|
| 1 | 2026-10-12 | Asked for a plan, skeleton repo, list of needs and key decisions; supplied brief, LO list, project sheet, dataset link; instructed Claude to ask rather than assume | Read brief/rubric/LOs; checked dataset page (classes, split sizes); found Roboflow unreachable from its workspace (so data work runs in Colab); wrongly told you GitHub pushes were blocked based on a tool warning before trying — the push then worked; drafted plan, decision log, LO map, I/O spec; scaffolded package, tests, CI, Colab setup notebook | *(to fill)* | Claude's `.gitignore` line `data/` also ignored the package folder `src/porkqc/data/`, so the first push was missing code. Tests passed locally (files existed on disk) but **CI failed on a fresh clone**; fixed by anchoring the pattern to `/data/`. Lesson: local tests alone did not verify what was actually committed. *(add your own)* |

| 2 | 2026-10-12 | Confirmed framing; chose AGPL-3.0; set deadline 22 Oct and free Colab; delegated line-speed choice; forked dataset on Roboflow | Re-planned to a day-by-day schedule; set line-speed requirement from published equipment speeds; added licence; built resumable training wrapper + notebook 02; CPU smoke tests incl. simulated disconnect; opened PRs for review instead of merging itself | *(to fill — e.g. reviewed and merged PRs #1–#2)* | Claude's first resume logic would have silently retrained a finished run (found by its own smoke test, see VERIFICATION_LOG V3). Claude's attempt to self-merge PRs was blocked; PRs now wait for your review |

## Running notes for the reflection
- *(e.g. "Claude proposed X; I chose Y because …")*
