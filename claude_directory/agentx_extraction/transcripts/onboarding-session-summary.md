# Onboarding session summary: InferenceX / AgentX benchmarking

Source: two recordings of one onboarding call (`InferenceX1.m4a`, 13.5 min, is part 1; `inferenceX.m4a` is part 2, of which only the first ~2.5 min is meeting audio). About 16 minutes of speech in total, with a gap between the two recordings. A senior colleague (the lead) handed over a benchmarking task. Names of colleagues and internal repositories are redacted.

## Where things stand: release gates

- Current release gates benchmark a sweep of fixed input lengths, output lengths and concurrencies.
- Each result is divided by an NVIDIA baseline run, and the gate value is the geometric mean of those ratios, checked against thresholds.
- The lead considers this outdated for today's workloads: the agentic case (a model in a harness, many turns, growing context) matters more.
- SemiAnalysis's InferenceX has considerable influence on what gets attention, so its agentic results matter.

## What InferenceX's agentic test ("AgentX") is

- NVIDIA's AIPerf benchmark client, run on a public dataset of agent traces recorded in a harness.
- Each point on the chart is one pass through the dataset at a fixed concurrency.
- It differs from fixed input/output-length benchmarks: many turns per request, heavy KV-cache reuse. A config can look bad at one input/output length and still be fine in aggregate.
- The exact command is public (around line 302 of the lib at the time, in the build-replay command function). It ends as a single AIPerf command.

## The task

1. Find the minimum core: AIPerf + the public dataset, configured with InferenceX's parameters, run from our own internal stack. Check whether it reproduces the published numbers without the rest of the InferenceX setup.
2. North star: a test for the agentic use case, built from InferenceX's working core. The lead had started this but was pulled onto other work.
3. Part 2 question: our tooling is built around `vllm bench serve`. Can a vLLM-bench setup be a good-enough proxy for AIPerf AgentX, so that optimizing on one gives good results on the other? It won't be a 1:1 replacement. If not, move to AIPerf.

## Points raised by me

- InferenceX does more than run the benchmark: trace generation, metrics, pinned versions. We need to decide which of these we need for our own visualization.
- InferenceX appears to use a separate internal harness for the agentic runs, which is worth looking at.

## Actions and blockers

- Me: find the minimal AgentX core, check it reproduces InferenceX's numbers from our stack, and assess vLLM bench as a proxy.
- Blocker: no access yet to [internal perf repo], which has an AIPerf AgentX example under `examples/`, or to several [internal repos].
- Fix: [colleague] needs to add me to [internal dev team]; they were on holiday that day.

See `../inferencex-agentx-extraction-guide.md` for the technical follow-up.
