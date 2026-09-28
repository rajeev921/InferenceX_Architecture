# InferenceX / AgentX benchmarking – onboarding session transcript

Source: `InferenceX1.m4a` (part 1, 13:28) and `inferenceX.m4a` (part 2, 29:54).
Names of colleagues and internal repositories are redacted as [colleague], [internal perf repo], [internal dev team], [internal repo(s)], [internal stack].

Auto-transcribed (Whisper medium), lightly cleaned for filler words. Speaker labels are inferred from context: **Lead** = the colleague walking you through the work, **Rajeev** = you. The recordings don't overlap; there is a gap between the end of part 1 and the start of part 2.

Part 2 contains only ~2.5 min of meeting audio. From ~02:25 to ~28:30 it is near-silence/background, and the last ~90 s (28:30–29:54) is unrelated background audio (not part of the meeting), so both are omitted.

---

## Part 1 — `InferenceX1.m4a`

**[00:00] Lead:** And what these release gates are is really just benchmarking across different input and output lengths and concurrencies. Then we compare against NVIDIA, take an average, and see if the performance is competitive. There are some thresholds that we use. Let me just open one case here, so it's something concrete.

**[00:36] Rajeev:** Can I start recording, so that I have it?

**[00:42] Lead:** Yeah, absolutely, go ahead.

**[00:50] Lead:** So let's look at, for example — do they have some baselines here? Let's take some other model where we have V4 Flash in. I'll take a random combination here, so this is going to be meaningless, but just to show it. So we take two runs: the baseline would be some NVIDIA run, and this would be our run. We calculate these ratios, and then we calculate the final gate value, which is the geometric mean of these ratios.

**[02:18] Lead:** The thing is that this is now sort of outdated for today's use cases. Yes, there are still some short-sequence cases, but more and more it's the agentic scenario — you have a model in a harness, and it's doing multiple turns with longer and longer context. That's becoming more important. And for better or worse, InferenceX and SemiAnalysis have a considerable amount of influence on what gets attention and what doesn't — and they are good at what they do as well.

**[03:19] Lead:** So this AgentX, or agentic scenario — what it is, essentially, is a run with AIPerf, which is the benchmarking client from NVIDIA, together with their own dataset. This dataset contains traces of some kind of agent in a harness. So instead of having multiple input/output lengths and concurrencies in isolation, there's this dataset, and the model runs through it at a fixed concurrency — and that's essentially one point here. Each point is a fixed concurrency and one run through this agentic dataset. This is all public. If you dig through it, you can find the actual command that they run. I think I linked it…

**[04:56] Rajeev:** Yes, you had line 302 or so. What was the name of the function — was it the build-replay command?

**[05:06] Lead:** Yeah. So this is just an AIPerf command at the end, and of course there are a lot of things involved around it. But what I wanted to understand first is: can we take just the minimum core of this — AIPerf with the public dataset — configure it with the same parameters, and get the same or similar results to what they are publishing? So we don't need the whole InferenceX apparatus; we just run the core and get similar results to what they show.

**[06:08] Rajeev:** Let me reiterate to check I understand correctly. You want to take the benchmark shell — not the complete thing, the minimal part — integrate it with our existing internal setup, and see whether, if we pass the same input parameters from our internal stack, we produce the same result?

**[06:33] Lead:** Yes. For starters I want to understand what the minimum component is that we need to extract.

**[06:48] Rajeev:** OK. Because when I was going through it, I saw they do a lot more. The benchmark also reads whatever they replay — how they generate the traces, what the different metrics are, what the different pinned versions are. We also have to think about what they generate, and what we do or don't need for our visualization.

**[07:21] Lead:** Right. I don't want to replicate this whole thing. The north star is that we want a test for this agentic use case. They basically have something that works, so we could extract the core bits and use them ourselves. I started doing some work along those lines, but then a bunch of other stuff came along.

**[08:17] Rajeev:** Just one question. When you say "agentic", they do three or four different kinds of work separately. Are we also trying to capture how their agentic workflow handles everything internally, as well as the benchmarking results — both of these?

**[08:45] Lead:** The benchmarking *is* the simulation of the agentic workload. When I say agentic workflow, I mean it's not the traditional benchmarking setup where we fix an input length, fix an output length, fix a concurrency, and repeat that many times. In an LLM-plus-harness setup, you have multiple turns on the same request, and a lot of KV-cache hits. The model might end up generating a very long output, but over the course of multiple turns. So, for example, the metrics might not look very good on some particular ISL/OSL combination, but in aggregate — for the real use case — that doesn't matter much. So, based on AIPerf and their own dataset, they simulate how a model performs as if it were in a real harness.

**[10:12] Rajeev:** Yes, understood. I also saw they use a separate harness for this, which they call internally. That will be interesting to look at.

**[10:30] Lead:** Yeah. Good. So if you go to our repository, [internal perf repo] — here in `examples` I think there's an AIPerf AgentX example.

**[10:55] Rajeev:** Do I have access to [internal perf repo]?

**[11:00] Lead:** You should.

**[11:01] Rajeev:** I have access to [internal repo], but I don't…

**[11:10] Lead:** Maybe not — we need to ask, maybe [colleague]. If I go here and look at the members, I should see you here… *(checks)* … No. So I think the next step is that [colleague] adds you.

**[12:17] Rajeev:** On these repositories I can't see [internal repos] — I don't see anything.

**[12:30] Lead:** We probably need to add you to the [internal dev team], or one of these teams, and then you'd be able to see them. Let me just check one thing. [colleague] is on holiday today, right?

**[12:56] Rajeev:** Yeah, I think he had to take…

**[13:00] Lead:** I'm trying to look for it, because I asked… *(recording ends)*

---

## Part 2 — `inferenceX.m4a` (picks up mid-conversation)

**[00:00] Lead:** …the question is whether optimizing for that setup is the same as optimizing for the AIPerf AgentX use case. The reason is that we have everything set up to use the vLLM bench client, and it's a lot easier if the setup is based on vLLM bench. But it might not be possible to emulate what AIPerf is doing, so we might really need to go towards AIPerf. Or maybe we don't. That's also a question I want to explore.

**[00:49] Rajeev:** OK. So it's whether we can remove AIPerf entirely, use vLLM bench instead, and capture the same information it does.

**[01:01] Lead:** Yeah. I think it will never replace it one-to-one.

**[01:10] Rajeev:** Yes, definitely.

**[01:12] Lead:** But at the end of the day, maybe it gives us a good proxy. As in: if we use this setup with `vllm bench serve`, it's a good proxy for what AgentX is testing. So if we optimize on one, we know we'll get good results on the other. At the end of the day, that's what we care about.

**[01:40] Rajeev:** Sure. I'll look into it from that perspective.

**[01:50] Lead:** All right, good. Any last questions?

**[01:57] Rajeev:** No, I think I already asked a lot. Thanks for the session.

**[02:05] Lead:** No worries. Let's hope you get all this access sorted out and can start doing some stuff soon. Have a good rest of the day.

**[02:20] Rajeev:** You too. Bye.
