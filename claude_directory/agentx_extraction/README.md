# agentx_extraction

Second set of notes (see `../README.md` for the first), drafted with Claude for the task of extracting InferenceX's agentic (AgentX) benchmark into a separate repository. Snapshot of 27 Sep 2026, pinned to InferenceX commit `88349ad` and SemiAnalysis AIPerf fork commit `754356e`.

| File | What it is |
| --- | --- |
| [inferencex-agentx-extraction-guide.md](inferencex-agentx-extraction-guide.md) | The main guide: where AgentX lives in InferenceX (with file and line links), what to extract or skip, how to run one point standalone, how to compare with published numbers, and whether `vllm bench serve` can act as a proxy |
| [agentx-code-flow.svg](agentx-code-flow.svg) / [.png](agentx-code-flow.png) | Code flow graph of one AgentX point: fleet orchestration → AgentX client → post-processing |
| [transcripts/onboarding-session-summary.md](transcripts/onboarding-session-summary.md) | What the onboarding call discussed and what was asked |
| [transcripts/onboarding-session-transcript.md](transcripts/onboarding-session-transcript.md) | Cleaned, speaker-labelled transcript of both recordings |
| [transcripts/raw/](transcripts/raw/) | Timestamped Whisper (medium) output for the meeting portions of each recording |

Colleague names and internal repository/team names are redacted. Speaker labels in the transcript are inferred from context. Server flags in the run example are translated from the InferenceX recipe and have not been validated on hardware yet.
