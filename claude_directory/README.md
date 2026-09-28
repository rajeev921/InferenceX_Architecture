# claude_directory — InferenceX architecture study notes

Notes produced with Claude from a read of [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) at commit `45aa3a24` (26 Sep 2026), with the `utils/aiperf` submodule at `754356e9`.

| File | Contents |
| --- | --- |
| [01_architecture_and_study_plan.md](01_architecture_and_study_plan.md) | What InferenceX is, repo map, component deep-dive, how components connect (9-step pipeline + handoff contracts), external dependencies, 6-phase study plan, extraction guide |
| [02_runtime_deep_dives.md](02_runtime_deep_dives.md) | Line-referenced walkthroughs: `benchmark_lib.sh`, srt hook scripts, `infx/bench_serving`, `infx/evals`, `infx/golden_al_distribution`, `infx/datasets`, `infx/results`, the aiperf fork, and `AGENTS.md` / skills / commands. Every `Lnnn` link is pinned to the commit above |
| [pipeline_diagram.svg](pipeline_diagram.svg) | The 9-step config → dashboard pipeline diagram |
| [agentx_extraction/](agentx_extraction/) | AgentX extraction task: extraction guide (what to pull out of InferenceX and how to run it standalone), code flow diagram, and onboarding-session transcripts (internal names redacted) |
