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
| 3 | 2026-10-12 | (continued while you ran notebook 00) | Built WP2 metrics (verified exactly against Ultralytics) and WP1 data-audit notebook with leakage check | *(to fill — review PRs #3, #4)* | **Claude's local dry run pushed fake audit results to a `colab-results` branch of your repo** (VERIFICATION_LOG V8). Claude added a guard and a regression test, but could not delete the branch — you need to delete it on GitHub |
| 4 | 2026-10-12 | Reported the download error from Colab; sent the development sample; asked how to merge PRs | Found and fixed the download bug (V9); from the sample found frame-number leakage, same-named different frames and inconsistent labels (V10); rewrote notebook 01 around frame numbers; added time-blocked re-split and `SPLIT` option in notebook 02; explained PR merging | *(to fill — e.g. your null-image review)* | Claude's planned image-hash leakage check would not have worked on this data; Claude's own dry run on the real sample showed it |
| 5 | 2026-10-14 | Sent run 1–3 and cross-evaluation results; coded the 26 unlabelled frames (25 E, 1 M) and explained the M frame; chose to re-code two "G" frames as edge cases | Evaluated runs; found that comparing runs on different validation sets was confounded (V12) and built cross-evaluation + paired bootstrap; quantified leakage (+0.27 mAP50) | *(to fill)* | Claude's first reading of run 2 (packaging-error "improved") was confounded by different validation sets — caught by Claude, fixed by cross-evaluation |
| 6 | 2026-10-17 | **Chose YOLO11n for the one-shot TEST** (Claude had recommended it; decision D-025); ran notebook 04 | Froze decision thresholds before TEST; built notebook 04 (TEST, speed, robustness, baseline features) and the AP-integral study; analysed TEST, baseline, roofline, robustness | *(to fill)* | Two wrong expectations in Claude's own unit tests (V15) |
| 7 | 2026-10-17 | Asked Claude to act as a marker and audit the repository against the brief and rubric | Audit (V16): rewrote README, completed docstrings + documentation test, finalised decision statuses, corrected LO-map claims, added failure-mode analysis, regenerated test report | *(to fill)* | Several of Claude's own earlier documents had become stale or over-claimed (e.g. a "debugger session" that never happened) |

## Running notes for the reflection
- *(e.g. "Claude proposed X; I chose Y because …")*
