# Runtime deep-dives

Each section below takes one row of the repo map and walks its code in reading order. Every `file:Lnnn` link opens GitHub at InferenceX commit `45aa3a2` (aiperf at `754356e`), so the line numbers stay correct even after `main` moves on. Each section starts with what the component is for, then gives a reading-order table (location · symbol · what it does), then lists the behaviours that are easy to miss. To read the fork locally, run `git submodule update --init utils/aiperf`.

## 1. `benchmarks/benchmark_lib.sh` — the shared runtime toolbox

Every hook script sources this 3,566-line bash library. It owns GPU telemetry, server readiness, the fixed-seq client call, all five eval frameworks and the AgentX replay. Sourcing it with `--validation-only` stops at L132 and loads only the helpers above that line. Launchers use that mode so they get `check_env_vars` without the side effects of a full load.

| Where | Symbol | What it does |
| --- | --- | --- |
| [L4](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L4) | `check_env_vars` | Checks the named variables by indirection (`${!var}`), then calls `exit 1`, not `return`, and so kills the caller |
| [L37](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L37) | `stop_background_process_groups` | Sends TERM to each process group, waits a grace period, sends KILL, waits again. Refuses PGID ≤ 1 and its own group |
| [L111](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L111) | `run_amd_multinode_after_preflight` | `srun` runs a per-node preflight script first, then the server containers only if every node passes |
| [L132](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L132) | `--validation-only` return | Stopping point for launchers; everything below needs the full load |
| [L154](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L154) | `select_available_server_port` | Tries to bind `$PORT`; if it is taken, binds port 0 and re-exports `PORT` |
| [L174–193](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L174) | `agentic_kv_offload_enabled`, `require_agentic_kv_offload_*` | Enforce the `KV_OFFLOADING` = none \| dram contract and a positive `TOTAL_CPU_DRAM_GB` |
| [L228](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L228) | `apply_chat_template_kwargs_shim` | Inline Python patch to vLLM: adds `--chat-template-kwargs` (the vllm#44244 equivalent) for SPEED-Bench runs |
| [L366](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L366) | `disable_trtllm_detailed_perf_metrics` | `sed` on TRT-LLM `py_executor.py` that forces per-request perf metrics off |
| [L384–417](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L384) | top-level agentic guard | If `IS_AGENTIC=1` or `SCENARIO_TYPE=agentic-coding`: `unset MAX_MODEL_LEN` and validate the offload env |
| [L448](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L448) | `start_gpu_monitor` | NVIDIA: `nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory -l 1` plus an identity CSV. AMD: `amd-smi metric -p -c -t -u -w 1 --csv` plus energy-accumulator start/end files and `amd-smi static --json` |
| [L495](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L495) | `stop_gpu_monitor` | Stops the sampler, drops a truncated last row, appends one final sample so the benchmark window is bracketed |
| [L632](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L632) | `wait_for_ready` | Tails the server log and polls an endpoint with `curl`; exits if the server PID dies or the timeout passes |
| [L726](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L726) | `wait_for_server_ready` | Waits on `/health`, then `python3 -m infx.bench_serving.server_watch capture --pid` snapshots the engine workers |
| [L760](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L760) | `run_server_client` | Wraps any client in `server_watch run` so it is killed if the engine dies mid-run |
| [L785](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L785) | `run_benchmark_serving` | Builds the `infx.bench_serving.benchmark_serving` call. Flags: `--dataset-name random`, `--request-rate inf`, `--ignore-eos`, `--num-warmups 2×conc`, `--percentile-metrics ttft,tpot,itl,e2el`. `PROFILE=1` adds `--profile` |
| [L1025–1061](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L1025) | `_find_latest_profile_trace`, `move_profile_trace_for_relay` | Finds the newest torch-profiler trace and copies it to `/workspace/profile_<RF>.trace.json.gz` for Perfetto |
| [L1122](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L1122) | `_install_lm_eval_deps` | `pip install lm-eval[api]`, then force-reinstalls lm-evaluation-harness at commit `b315ef3b` |
| [L1141](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L1141) | `_prepare_vendor_verifier_python` | Chooses or builds a Python ≥ 3.12 venv, using uv 0.11.33 when needed, for the vendor verifiers |
| [L1339–1439](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L1339) | `_run_kimi_tool_call_schema_eval`, `run_kimi_vendor_eval` | Kimi-Vendor-Verifier pinned at `3dad65a7`; archive SHA-256 is checked |
| [L1593–1717](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L1593) | `_run_bfcl_suite_eval`, `run_bfcl_eval` | BFCL suites: smoke (4 threads, 900 s), MiniMax (8, 7200 s), Kimi (16, 14400 s) |
| [L1755–1961](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L1755) | `_run_minimax_m3_*`, `run_minimax_vendor_eval` | MiniMax provider verifier, smoke (1 case) and full (102 cases) |
| [L1991–2039](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L1991) | `get_native_max_context_length`, `compute_eval_context_length` | Reads `max_position_embeddings` via `AutoConfig`; eval context = min(native, requested), 16384 fallback |
| [L2044](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L2044) | `run_lm_eval` | `python3 -m lm_eval --model local-chat-completions --apply_chat_template --tasks infx/evals/gsm8k.yaml --log_samples`; max output = ctx − 4096, capped at 16384 |
| [L2255](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L2255) | `_write_lm_eval_meta_json` | Writes `meta_env.json`, the identity record (hw, topology, conc, prefix…) that the collectors and the app key eval rows on |
| [L2417](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L2417) | `stage_eval_artifacts` | Copies `results*.json`, `sample*.jsonl`, SWE-bench preds and reports, and trajectories into the upload dir |
| [L2514](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L2514) | `_run_swebench_agentic_generation` | `mini-extra swebench --subset lite --environment-class swerex_modal`, with a watchdog on the `preds.json` count |
| [L2680](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L2680) | `run_swebench_eval` | Generation, then scoring with `python3 -m infx.evals.swebench_score --modal` |
| [L2795](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L2795) | `_wait_for_openai_chat_route` | Ready = model listed in `/v1/models` and the chat route answers GET with 401/403/405, or health has been stable |
| [L2893](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L2893) | `run_eval` | Dispatcher on `EVAL_FRAMEWORK` (lm-eval \| swebench \| kimi-vendor \| minimax-vendor \| bfcl). A space-separated `EVAL_CONCURRENT_REQUESTS` gives batched per-conc lm-eval |
| [L3116](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L3116) | `install_agentic_deps` | `uv venv`, then `uv pip install -e utils/aiperf` plus numpy, pandas, transformers, datasets, `huggingface_hub[cli]` |
| [L3165](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L3165) | `resolve_trace_source` | Picks the `semianalysis_cc_traces_weka_062126` loader (or `_256k` for other models), then `hf download --repo-type dataset` |
| [L3236](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L3236) | `build_replay_cmd` | Assembles `aiperf profile --scenario inferencex-agentx-mvp ...` (flags in section 8) |
| [L3368](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L3368) | `write_agentic_result_json` | `python -m infx.results.agentic.process_agentic_result`, then `generate_aiperf_plots` |
| [L3412](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L3412) | `run_agentic_replay_and_write_outputs` | Runs in a subshell. Power monitor or window, then replay via `run_server_client`, aggregate, power adapter, distributions, and `validate_agentic_result` (10% failure cap) |

Call order in a fixed-seq run: `check_env_vars` → `start_gpu_monitor` → `run_benchmark_serving` → EXIT trap `stop_gpu_monitor`; then, as post\_eval, `run_eval` → `compute_eval_context_length` → `run_lm_eval` → `append_lm_eval_summary` → `stage_eval_artifacts`. In an agentic run: `resolve_trace_source` → `install_agentic_deps` → per concurrency, `build_replay_cmd` → `run_agentic_replay_and_write_outputs`.

Easy to miss:

- `check_env_vars`, `wait_for_ready` and `resolve_trace_source` call `exit`, not `return`. The whole sourcing shell dies on failure.
- `AGENTS.md` bans `${VAR:-default}` fallbacks in shared bash. All configuration must come from the caller and be validated, so a missing variable fails loudly rather than silently using a default.
- The EXIT trap on `stop_gpu_monitor` is what guarantees a closing power sample. Without it, J/token cannot be integrated to the end of the window.

## 2. srt hook scripts — what srt-slurm runs once the server is up

srt-slurm starts the engine, then runs the recipe's `benchmark.command` (`benchmark.type: custom`) inside the container, and after it the `post_eval.command`. Five short scripts fill those two slots. They get their inputs from three places:

- **srt-slurm** exports `SRT_FRONTEND_HOST`, `SRT_FRONTEND_PORT`, `SRT_*_ENDPOINTS`, `MODEL_NAME` and `SRT_MEASUREMENT_WINDOW_DIR`. This is inferred from the call sites; the srt-slurm source was not read.
- **The recipe's** `benchmark.env` supplies `MODEL`, `ISL`, `OSL`, `RANDOM_RANGE_RATIO` and `USE_CHAT_TEMPLATE`.
- **InferenceX** injects `--set` overrides through [`infx/srt_slurm/single_node.py` L163–198](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/srt_slurm/single_node.py#L163): `CONC`, `RESULT_FILENAME`, `RUN_EVAL`, `EVAL_ONLY`, `FRAMEWORK`, and `RESULT_DIR=/logs`.

### Wiring

| Where | What |
| --- | --- |
| [`qwen3.5/trtllm/b200-fp4/8k1k.yaml` L53](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/single_node/srt-slurm-recipes/qwen3.5/trtllm/b200-fp4/8k1k.yaml#L53) | Example single-node recipe: `command: bash /infmax-workspace/benchmarks/single_node/srt_fixed_sequence.sh` |
| [`minimaxm3/vllm/gb200-fp4/agentx/agg-dep4.yaml` L93](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/multi_node/srt-slurm-recipes/minimaxm3/vllm/gb200-fp4/agentx/agg-dep4.yaml#L93) | Example AgentX recipe calling `srt_agentic.sh` |
| [`runners/slurm_utils.sh` L7–9](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/runners/slurm_utils.sh#L7) | Multi-node `post_eval.command = [bash, .../multi_node/srt_eval.sh, {endpoint}, {infmax_workspace}]` |
| [`slurm_utils.sh` L43–56](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/runners/slurm_utils.sh#L43) | `SRT_EVAL_PASSTHROUGH`: the env list forwarded to post\_eval (EVAL\_*, SWEBENCH\_*, MODAL\_\*, TP, EP\_SIZE…) |
| [`slurm_utils.sh` L179–182](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/runners/slurm_utils.sh#L179) | Single-node override: `post_eval.command = [bash, .../single_node/srt_eval.sh, {endpoint}, /logs/infx-eval-exit-code]` |
| [`slurm_utils.sh` L246–252](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/runners/slurm_utils.sh#L246) | Launcher fails the job unless the eval exit-code file says `0` and the result JSON is non-empty |
| [`single_node.py` L117](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/srt_slurm/single_node.py#L117) | Validates that `benchmark.command` ends in `srt_agentic.sh` exactly when the run is agentic |

### [`single_node/srt_fixed_sequence.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/single_node/srt_fixed_sequence.sh) (64 lines)

| Lines | What happens |
| --- | --- |
| L1–7 | Sources the lib with `--validation-only`, then `check_env_vars MODEL CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME RESULT_DIR SRT_FRONTEND_HOST SRT_FRONTEND_PORT RUN_EVAL EVAL_ONLY GPU_MONITOR_INTERVAL USE_CHAT_TEMPLATE FRAMEWORK` |
| L14–18 | Maps the framework to a client backend: sglang or atom → `vllm` (OpenAI completions); trt → `openai`. Any other framework is an error, so there are no single-node vLLM fixed-seq recipes |
| L27–43 | Optional `--use-chat-template`. Validates integers and that `RESULT_DIR` exists |
| L45–47 | Full lib load; `pip3 install --break-system-packages sentencepiece datasets pandas` |
| L49–50 | `start_gpu_monitor --output $RESULT_DIR/gpu_metrics.csv`, plus a `trap ... stop_gpu_monitor ... EXIT` |
| L52–64 | `run_benchmark_serving --base-url http://$SRT_FRONTEND_HOST:$SRT_FRONTEND_PORT --num-prompts CONC×10 --max-concurrency CONC` writes `/logs/$RESULT_FILENAME.json` |

### [`single_node/srt_eval.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/single_node/srt_eval.sh) (41 lines)

| Lines | What happens |
| --- | --- |
| L1–8 | Arguments are `endpoint status-file`. An EXIT trap always writes the return code to the status file |
| L12–24 | Full lib load. Fixed-seq runs use `--framework lm-eval`. Agentic runs keep the workflow's `EVAL_FRAMEWORK` (swebench); GLM 5.2 also gets `SWEBENCH_AGENT_STEP_LIMIT=150` |
| L25–33 | Parses `PORT` from the endpoint, refuses multinode, and uses `/model` as `MODEL_PATH` if that directory exists |
| L35–41 | `run_eval --port $PORT`, then `append_lm_eval_summary`, which writes `meta_env.json` and the staged `results*.json` and `samples*.jsonl` |

### [`multi_node/srt_fixed_sequence.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/multi_node/srt_fixed_sequence.sh) (57 lines)

| Lines | What happens |
| --- | --- |
| L1–7 | Loads the lib with `--validation-only` only, so there is no GPU monitor. Power comes from the DCGM exporter instead |
| L10–20 | Backend `openai` (`/v1/completions`) by default, or `openai-chat`; chat template on by default |
| L22–25 | Model name = first `id` from `curl .../v1/models`, not the workflow `MODEL` |
| L26–29 | `ctx = PREFILL_NUM_WORKERS×PREFILL_TP`, `gen = DECODE_NUM_WORKERS×DECODE_TP`; output dir `/logs/sa-bench_isl_X_osl_Y` |
| L30–52 | Loops over `CONC_LIST` with `python3 -P -m infx.bench_serving.benchmark_serving ... --num-prompts 10c --num-warmups 2c` and writes `results_concurrency_{c}_gpus_{ctx+gen}_ctx_{ctx}_gen_{gen}.json` |
| L53–56 | If `SRT_MEASUREMENT_WINDOW_DIR` is set, `python3 -m infx.results.power.window` records the measurement window for the DCGM power package |

### [`multi_node/srt_eval.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/multi_node/srt_eval.sh) (61 lines)

| Lines | What happens |
| --- | --- |
| L10–26 | Parses host and port from the endpoint, `cd`s to the workspace, full lib load |
| L30–41 | Uses prefill topology for eval metadata: `TP=$PREFILL_TP`, `EP_SIZE=$PREFILL_EP`, `EVAL_CONCURRENT_REQUESTS=$EVAL_CONC` |
| L44–60 | `run_eval`, then `append_lm_eval_summary`, then copies artifacts to `/logs/eval_results` and propagates the return code |

### [`benchmarks/srt_agentic.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/srt_agentic.sh) (184 lines)

| Lines | What happens |
| --- | --- |
| L12–19 | Finds the lib; defaults `IS_MULTINODE=false` and `PORT=8000`; full load |
| L21–50 | Target URL is `AIPERF_SERVER_URL`, else `http://$SRT_FRONTEND_HOST:$SRT_FRONTEND_PORT`, else localhost. `AIPERF_MAX_CONTEXT_LENGTH` is restored as `MAX_MODEL_LEN`, because the lib unsets it for agentic runs |
| L53–59 | Builds worker `/metrics` URLs from `SRT_AGG_ENDPOINTS` or `SRT_PREFILL_ENDPOINTS` + `SRT_DECODE_ENDPOINTS` (skipped for the Dynamo frontend) so aiperf can scrape KV and cache counters |
| L61–83 | Concurrency list from `CONC_LIST` or `CONC`; then `resolve_trace_source`, `install_agentic_deps`, and a chat-route wait when `EVAL_ONLY` |
| L85–151 | `wait_for_agentic_servers_idle`: needs three consecutive idle polls of `dynamo_frontend_active_requests`, `vllm:num_requests_{running,waiting}` or `trtllm_num_requests_*` |
| L158–184 | Per concurrency: `RESULT_FILENAME=<base>_conc<c>`, `build_replay_cmd`, `run_agentic_replay_and_write_outputs`, then drain before the next point |

Easy to miss: the multi-node fixed-seq script names the model from `/v1/models`, and multi-node evals record prefill TP/EP as the "TP" of the row. Keep both in mind when joining eval rows to throughput rows.

## 3. `infx/bench_serving` — the fixed-seq load generator

This is a trimmed fork of vLLM's `benchmarks/benchmark_serving.py` and `backend_request_func.py`. It keeps the SPDX header, the `ASYNC_REQUEST_FUNCS` table, the goodput code and even the `"request_goodput:"` key typo. Only `--dataset-name random` survives, and you must pass it: the argparse default is still `sharegpt`, which then fails with "Unknown dataset".

### `benchmark_serving.py` (1,300 lines)

| Where | Symbol | What it does |
| --- | --- | --- |
| [L106](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/benchmark_serving.py#L106) | `_load_tokenizer` | Uses vLLM's tokenizer for `--tokenizer-mode deepseek_v4`; otherwise the local HF `get_tokenizer`, with an `AutoTokenizer` fallback |
| [L167](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/benchmark_serving.py#L167) | `_apply_chat_template` | `--dsv4` uses `encoding_dsv4.encode_messages(thinking_mode="thinking")`; otherwise HF `apply_chat_template(add_generation_prompt=True)` |
| [L235](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/benchmark_serving.py#L235) | `sample_random_requests` | Prompt synthesis; algorithm below |
| [L377](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/benchmark_serving.py#L377) | `get_request` | Arrivals. With `rate = inf` there is no sleep, so every request is released at once and the semaphore sets the real load. Otherwise gamma gaps with mean 1/rate (Poisson when `burstiness=1`) |
| [L404](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/benchmark_serving.py#L404) | `calculate_metrics` | Metric math, formulas below |
| [L508](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/benchmark_serving.py#L508) | `benchmark` | Warmups (own semaphore) → optional `/start_profile` → clock start → one task per request under `Semaphore(max_concurrency)` → `gather` → `/stop_profile` → duration → result dict (L695) |
| [L826](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/benchmark_serving.py#L826) | `main` | Seeds RNGs, `gc.freeze()`, runs, computes the `benchmark_outcome`, saves JSON, raises `SystemExit("FAIL")` if more than 5% of requests failed |

How `sample_random_requests` hits an exact ISL:

1. It draws lengths uniformly in \[⌊L·r⌋, L\] with r = `--random-range-ratio` (0.8 in InferenceX), separately for input and output.
2. With a chat template, it measures the template's token overhead once and subtracts it from each target, so the ISL includes the template.
3. Token ids for request i are `(offset_i + i + j) mod vocab`, a consecutive-id run from a random offset, so prompts share no prefix.
4. It then decodes and re-encodes up to 10 times, padding with random ids or truncating until the round-trip length equals the target. This is why the exact model tokenizer is required.
5. The work runs in parallel over `multiprocessing.Pool` (up to 8 workers). The seeds come from a copy of the global RNG, so results do not change with the worker count.

Metric definitions (all reported in ms, then converted in section 7):

```latex
\mathrm{TPOT}_i = \frac{\mathrm{latency}_i - \mathrm{TTFT}_i}{\mathrm{out}_i - 1}, \qquad \mathrm{total\ tok/s} = \frac{\sum \mathrm{in}_i + \sum \mathrm{out}_i}{\mathrm{duration}}
```

ITL is the gap between consecutive SSE chunks, flattened across all requests. E2EL is the request latency. Output length comes from the server's `usage.completion_tokens`, and is retokenised only when that field is `None`. The percentiles default to 90/99/99.9; InferenceX asks for ttft, tpot, itl and e2el.

### `backend_request_func.py` (539 lines)

| Where | Symbol | What it does |
| --- | --- | --- |
| [L20–35](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/backend_request_func.py#L20) | `RequestFuncInput` / `RequestFuncOutput` | Per-request in and out records; `output_tokens` defaults to 0 |
| [L224](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/backend_request_func.py#L224) | `async_request_openai_completions` | Streams `/v1/completions` with `temperature 0`, `stream_options.include_usage`, `ignore_eos`. TTFT = first chunk that has `choices`; ITL appended per chunk; latency = last token chunk − start |
| [L313](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/backend_request_func.py#L313) | `async_request_openai_chat_completions` | Same for chat. Latency here includes the final usage-only chunk |
| [L414](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/backend_request_func.py#L414) | `_fix_tokenizer_for_sglang` | transformers v5 rebuilds some fast tokenizers differently from the server. This swaps in the raw `pre_tokenizer`/`decoder` and fixes BOS/EOS so client token counts match SGLang's |
| [L529](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/backend_request_func.py#L529) | `ASYNC_REQUEST_FUNCS` | vllm, sglang, openai, lmdeploy, scalellm → completions; `openai-chat`; tgi; tensorrt-llm; deepspeed-mii |

### Helpers

| File | What it does |
| --- | --- |
| [`benchmark_outcome.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/benchmark_outcome.py#L18) | `benchmark_outcome(requested, completed)` returns `failed` if more than 5% of requests failed. `infx.results` re-checks it |
| [`server_watch.py` L35](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/server_watch.py#L35) | `snapshot`: root PID plus descendants matching `sglang::scheduler\|EngineCore\|VllmWorker\|TPWorker\|trtllm-worker`, keyed by `/proc/<pid>/stat` start time (guards against PID reuse) |
| [`server_watch.py` L105](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/server_watch.py#L105) | `run`: starts the client in its own session and checks health every 2 s. If a worker dies or turns zombie, it sends SIGTERM and then SIGKILL to the client group and returns 1 |
| [`encoding_dsv4.py` L475](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/encoding_dsv4.py#L475) | `encode_messages`: self-contained DeepSeek-V4 template (DSML tool markup, `<think>` handling). The benchmark only produces `BOS<｜User｜>prompt<｜Assistant｜><think>` |
| [`benchmark_utils.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/bench_serving/benchmark_utils.py#L8) | Optional PyTorch-OSS benchmark record, only when `SAVE_TO_PYTORCH_BENCHMARK_FORMAT` is set |

Easy to miss:

- ITL is per SSE chunk, not per token. With MTP or any engine that sends several tokens per chunk, ITL and TPOT diverge, and TPOT is what feeds interactivity.
- If the server never sends a usage chunk, `output_tokens` stays 0 and is not retokenised. Those requests then drop out of TPOT.
- Duration is measured after `/stop_profile`, so profiled runs slightly under-report throughput.

## 4. `infx/evals` — accuracy guardrails against the live server

Evals exist to catch a fast config that returns wrong answers. They run against the same OpenAI-compatible endpoint the benchmark used, at chosen concurrency points (single-node 8k1k at the highest and median conc; multi-node at the highest per topology). Every adapter writes lm-eval-shaped `results_*.json`, so one validator and one collector handle them all. Start with [`EVALS.md`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/EVALS.md). Its headings are Selection (L8), How (L108), Metrics (L583), Environment variables (L603) and Adding a new eval task (L620).

| Framework (`EVAL_FRAMEWORK`) | Task / suite files | Adapter code | Shell entry | Scoring |
| --- | --- | --- | --- | --- |
| lm-eval (default) | [`gsm8k.yaml`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/gsm8k.yaml) (5-shot, `#### N`, strict and flexible regex), [`gpqa_diamond.yaml`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/gpqa_diamond.yaml) + [`utils.process_docs`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/utils.py#L8) (answers shuffled, 2 repeats, seed 3407) | lm-evaluation-harness `local-chat-completions` | `run_lm_eval` L2044 | `exact_match` |
| swebench | [`swebench_lite.yaml`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/swebench_lite.yaml) | mini-swe-agent 2.4.5 in Modal sandboxes, then [`swebench_score.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/swebench_score.py#L244) | `run_swebench_eval` L2680 | resolved / total (`exact_match,resolved`) |
| kimi-vendor | `kimi_tool_call_schema` (2 records), `_full` (408) | [`kimi_vendor_eval.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/kimi_vendor_eval.py#L40) runs upstream pytest; [`_kimi_verifier_archive.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/_kimi_verifier_archive.py#L25) fetches it SHA-verified | `run_kimi_vendor_eval` L1439 | passed / total |
| minimax-vendor | `minimax_m3_smoke` ([fixture](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/minimax_m3_smoke.json)), `minimax_m3_full` (102 cases) | [`minimax_provider_eval.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/minimax_provider_eval.py#L222), [`minimax_m3_full_eval.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/minimax_m3_full_eval.py#L331) (upstream pinned at `c899f95e`) | `run_minimax_vendor_eval` L1961 | tool-call match rate |
| bfcl | `bfcl_smoke`, `bfcl_vllm_minimax_m3`, `bfcl_vllm_kimi` | [`bfcl_adapter.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/bfcl_adapter.py#L576) (`bfcl-eval==2026.3.23`) | `run_bfcl_eval` L1717 | `acc` per category |

The common output format (`inferencex-eval-v1`, built in [`kimi_vendor_eval.py` L237](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/kimi_vendor_eval.py#L237)): `{result_format, eval_adapter, model_name, results: {task: {"exact_match,strict-match": score}}, configs, "n-samples": {task: {original, effective}}, integration_error?}`. A failed integration still writes this file, with score 0 and an `integration_error`, so a missing result is never mistaken for a pass.

### Gating: [`thresholds.yaml`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/thresholds.yaml) + [`validate_scores.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/validate_scores.py#L219)

- Default thresholds: GSM8K 0.90, GPQA-diamond 0.30, SWE-bench Lite 0.50. Per-model GSM8K overrides: dsr1 0.91, dsv4 0.91, glm5 0.94, glm5.1 0.93, qwen3.5 0.94. Vendor and BFCL suites are 0.0, which makes them diagnostic only.
- [`resolve_threshold` L63](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/validate_scores.py#L63) looks in order at `models[prefix][task]`, then `default[task]`, then `--min-score` (0.85).
- The validator exits 1 on any of: an `integration_error`, an invalid `n-samples.effective`, a primary metric below its threshold, zero results checked, or a batch-manifest mismatch ([L119](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/validate_scores.py#L119): `eval_concs` must equal `--expected-concs`).
- It runs from the workflow, not the shell lib: [`benchmark-tmpl.yml` L485](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.github/workflows/benchmark-tmpl.yml#L485) and [`benchmark-multinode-tmpl.yml` L510](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.github/workflows/benchmark-multinode-tmpl.yml#L510).

### Runtime patches ([`patches/`](https://github.com/SemiAnalysisAI/InferenceX/tree/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/evals/patches))

| Patch | Target | Change |
| --- | --- | --- |
| `lm_eval_sitecustomize.py` | lm-eval 0.4.9.2 | Import-time monkeypatch. `parse_generations` falls back to `reasoning_content` when `content` is empty, so thinking models are not scored 0. Installed as `sitecustomize.py` on `PYTHONPATH` |
| `patch_swebench_agent.py` | mini-swe-agent 2.4.5, swe-rex 1.4.0 | Anchor-checked source edits: a git-diff submission fallback, guaranteed sandbox teardown, and a `</dev/null` stdin fix (SWE-ReX #281) |
| `patch_swebench_scoring.py` | swebench 4.1.0 | Modal sandbox CPU from `$SWEBENCH_EVAL_SANDBOX_CPU` (default 2), plus a `finally: sandbox.terminate()` |

Easy to miss: the anchor patches refuse to apply if an anchor string is missing or appears twice, so upgrading a pinned tool breaks loudly rather than silently. `EVALS.md` describes the patches as atomic, but they write in place with `write_text`.

## 5. `infx/golden_al_distribution` — making spec-decode results comparable

Speculative decoding throughput depends on how many draft tokens the target accepts, and that depends on the draft head and the prompts. Synthetic traces would make acceptance unrealistic, and a tuned draft head would win on draft quality rather than serving quality. InferenceX avoids both. It measures one golden acceptance length (AL) per model, thinking mode and draft length on real coding prompts, then forces every agentic submission's engine to accept exactly that many tokens per step. A submitter still chooses the draft length; the AL at that length is fixed.

### Where the numbers come from

| Where | What |
| --- | --- |
| [`.github/workflows/speedbench-al.yml`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.github/workflows/speedbench-al.yml) | Manual workflow: B300, TP8, fp4, vLLM; sweeps `num_speculative_tokens` 1–8 × thinking on/off; can open a PR with the YAML |
| [`benchmarks/single_node/speedbench/dsv4_fp4_b300_vllm.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/single_node/speedbench/dsv4_fp4_b300_vllm.sh#L115) | `run_cell`: a fresh `vllm serve --speculative-config {method: mtp, num_speculative_tokens: N}` for each cell, then `vllm bench serve --dataset-name speed_bench --speed-bench-category coding --temperature 1.0 --max-concurrency 1` |
| same file L160–188 | Reads `vllm:spec_decode_num_accepted_tokens_total` and `..._num_drafts_total` from `/metrics` before and after; AL = 1 + Δaccepted / Δdrafts, 2 decimals |
| [`README.md`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/golden_al_distribution/README.md) | Method (L75–94), the fairness rule (L25), per-engine knobs (L27–71), curve table (L124) |

Curve file format, [`dsv4_mtp.yaml`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/golden_al_distribution/dsv4_mtp.yaml) (keys = draft tokens, values = AL):

```yaml
deepseek-v4-pro:
  thinking_on:  {1: 1.79, 2: 2.27, 3: 2.49, 4: 2.54, 5: 2.55, 6: 2.55, 7: 2.54, 8: 2.58}
  thinking_off: {1: 1.92, 2: 2.61, 3: 2.97, 4: 3.00, 5: 3.10, 6: 3.16, 7: 3.00, 8: 3.11}
```

### Lookup API

| Where | Symbol | What it does |
| --- | --- | --- |
| [`curves.py` L19](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/golden_al_distribution/curves.py#L19) | `Curve` | Validates 1 ≤ AL ≤ tokens + 1 and that the value is finite |
| [`curves.py` L41](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/golden_al_distribution/curves.py#L41) | `curve_name(model, spec)` | Maps (model prefix, spec method, draft model, sample method) to a file: eagle/nextn becomes `eagle3` for MiniMax, else `mtp`; the dspark and GQA variants get their own files |
| [`curves.py` L82](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/golden_al_distribution/curves.py#L82) | `golden_length(model, spec, thinking)` | The value the injector uses |
| [`__main__.py` L109](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/golden_al_distribution/__main__.py#L109) | CLI | `python -m infx.golden_al_distribution list \| show \| lookup` |

### Injection: [`infx/srt_slurm/synthetic_acceptance.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/srt_slurm/synthetic_acceptance.py)

| Where | What |
| --- | --- |
| [L21](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/srt_slurm/synthetic_acceptance.py#L21) | `ENGINES`: framework → engine (dynamo-sglang → sglang, dynamo-trt → trtllm, …) |
| [L40](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/srt_slurm/synthetic_acceptance.py#L40) | `spec_parameters`: reads the spec method, draft length and draft model from each engine's own recipe args (`speculative-config`, `speculative-num-steps`, `speculative_config.max_draft_len`, …) |
| [L93](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/srt_slurm/synthetic_acceptance.py#L93) | `build_overrides`: active only when agentic, not eval-only, and `SPEC_DECODING != none`. Emits per-role `srtctl --set`: vLLM `speculative-config.rejection_sample_method=synthetic` + `synthetic_acceptance_length=AL`; SGLang env `SGLANG_SIMULATE_ACC_LEN=AL`, `..._METHOD=match-expected`, `..._TOKEN_MODE=real-draft-token`; TRT-LLM env `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=AL−1`; ATOM `spec-decode-acceptance-length=AL` |
| [L201](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/srt_slurm/synthetic_acceptance.py#L201) | `plan_commands`: expands recipe variants with `srtctl.core.config.generate_override_configs`, merges the caller's `--set` flags and rejects conflicts, then emits one `srtctl apply` per variant |

Easy to miss: fixed-seq runs (random tokens) and evals are not forced; synthetic acceptance applies only to AgentX. `AGENTS.md` L77–83 forbids hard-coding an AL anywhere else.

## 6. `infx/datasets` — building the AgentX trace datasets (offline)

AgentX replays real coding-agent sessions. SemiAnalysis logs its own Claude Code traffic through a proxy into Postgres (Neon). This package samples sessions from that log and converts each one into a "weka" trace: token counts, timings and KV-block hash IDs, with no text. It then publishes the result to Hugging Face as `semianalysisai/cc-traces-weka-*`. None of this runs in CI; benchmarks only download the published dataset. You need it only to build your own trace set.

Pipeline: `sample_proxy_traces.py` → per-session JSONL → `proxy_to_weka.py` → per-trace JSON → `build_weka_hf_dataset.py` → `traces.jsonl` + plots + README → `HfApi.upload_folder`.

### [`sample_proxy_traces.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/sample_proxy_traces.py) (987 lines, psycopg 3, `AGENTIC_PROXY_DB_URL`)

| Where | Symbol | What it does |
| --- | --- | --- |
| [L61–121](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/sample_proxy_traces.py#L61) | SQL fragments | Rows after 2026-04-16 only (subagent labels begin then); `model LIKE 'claude-%'`; thread and agent IDs from request headers; excludes classifier calls (`max_tokens ≤ 64`, no tools) and security-monitor subagents |
| [L393](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/sample_proxy_traces.py#L393) | `CANDIDATES_PHASE1_SQL` | Per-session aggregates (request count, main turns, subagent requests, span, CLI version) with threshold filters |
| [L451](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/sample_proxy_traces.py#L451) | `SUBAGENT_CONCURRENCY_SQL` | Sweep line over subagent spans to get peak parallel subagents per session |
| [L517](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/sample_proxy_traces.py#L517) | `find_sessions` | Filter → drop image sessions → subagent concurrency → sort (top, recent, or seeded random by md5) → limit |
| [L694](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/sample_proxy_traces.py#L694) | `REQUESTS_SQL` | Row dump per session: `t_sec`, in/out/cache tokens, `hash_ids`, `hash_token_count`, ttft, duration, labels |
| [L761](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/sample_proxy_traces.py#L761) | `is_dynamic_workflow_bug` | Drops sessions from older CLI versions whose unlabeled parallel agents would replay incorrectly |
| [L837](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/sample_proxy_traces.py#L837) | `main` | Writes `<session_id>.jsonl` files plus a `manifest.json` of filters and stats |

### [`proxy_to_weka.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/proxy_to_weka.py) (449 lines, stdlib)

| Where | Symbol | What it does |
| --- | --- | --- |
| [L129](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/proxy_to_weka.py#L129) | `remap_hash` | Global 24-hex block hashes → per-session ints 0, 1, 2… (`hash_id_scope: "local"`). Shared IDs mark shared KV prefix |
| [L135–143](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/proxy_to_weka.py#L135) | `infer_block_size`, `effective_input_length` | Block size is always 64. Input length = `hash_token_count`, else blocks × 64, else in + cache read + cache write |
| [L162–178](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/proxy_to_weka.py#L162) | `build_normal_request`, `build_top_request` | `{t, type: n\|s, model, in, out, hash_ids, api_time, think_time?, ttft?}` |
| [L200](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/proxy_to_weka.py#L200) | `compute_think_times` | Think time = start of this request − end of the previous one, clamped at 0 |
| [L269](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/proxy_to_weka.py#L269) | `build_subagent_entry` | Groups one subagent's requests into `{type: subagent, agent_id, duration_ms, total_tokens, requests: [...]}` |
| [L307](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/proxy_to_weka.py#L307) | `session_to_weka` | Walks rows in time order: unlabeled → main turn; an agent-ID group is emitted whole at first appearance. Returns `{id, models, block_size: 64, hash_id_scope: local, requests}` |

The schema consumer is aiperf's [`weka_trace_models.py`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/dataset/loader/weka_trace_models.py#L142) (`WekaTrace`, L142). Shape of one trace (fields per the converter; values illustrative):

```json
{"id": "<session>", "models": ["claude-..."], "block_size": 64, "hash_id_scope": "local",
 "requests": [
  {"t": 0.0, "type": "s", "model": "claude-...", "in": 41216, "out": 812, "hash_ids": [0, 1, 2], "api_time": 14.2, "ttft": 2.1},
  {"t": 19.8, "type": "s", "in": 42304, "out": 230, "hash_ids": [0, 1, 2, 660], "api_time": 6.0, "think_time": 3.6},
  {"t": 26.1, "type": "subagent", "agent_id": "subagent_001_ab12cd34", "duration_ms": 48210, "requests": [ ... ]}
 ]}
```

### [`build_weka_hf_dataset.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/build_weka_hf_dataset.py#L688) (752 lines)

`main` (L688) runs sample (L151) → convert (L188) → optional ISL filter (L617) → payload (L451: `traces.jsonl`, `stats.txt`, plots, README) → upload (L633). The `--repo-256k` variant ([`_filter_trace` L513](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/datasets/build_weka_hf_dataset.py#L513)) drops any request with in + out > 256k tokens. It then re-bases the timestamps and recomputes subagent durations, so models with shorter context can replay the same sessions. `plot_weka_distributions.py` and `plot_subagent_distributions.py` draw ISL, OSL, think time, cache-hit and fan-out histograms.

Easy to miss: the prompt text is never stored. aiperf rebuilds prompts from `hash_ids` (section 8), so the prefix-cache hit rate is reproduced faithfully even though the content is synthetic.

## 7. `infx/results` — raw output to aggregate rows

This is where raw client JSON becomes a dashboard row. Each format has a pure builder that takes loaded data plus an env mapping and does no I/O, and a thin CLI that finds files and writes `agg_*.json`. Nearly all of it is stdlib-only, which makes it the easiest code in the repo to reuse.

### Fixed sequence: [`fixed_sequence.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/fixed_sequence.py)

| Where | Symbol | What it does |
| --- | --- | --- |
| [L45](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/fixed_sequence.py#L45) | `build_result(benchmark, env)` | Identity fields: `hw` = `RUNNER_TYPE`, `conc` = `max_concurrency`, image, model, prefix, framework, precision, spec, disagg, fingerprint, isl, osl. Re-checks `benchmark_outcome` |
| L90–170 | multinode branch | GPUs = prefill + decode. `tput_per_gpu` = total tok/s ÷ all GPUs; `output_tput_per_gpu` = output tok/s ÷ decode GPUs; `input_tput_per_gpu` = input tok/s ÷ prefill GPUs. Aggregate (non-disagg) mode uses `AGGREGATE_GPUS` |
| L171–201 | single-node branch | GPUs = tp × pp × pcp ([`topology.py` L20](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/topology.py#L20)); EP and DCP share those GPUs |
| L203–207 | conversion loop | Every `*_ms` key becomes seconds with `_ms` dropped. Every tpot key also produces `*_intvty` = 1000 / tpot\_ms, i.e. tokens/s per user, the x-axis of the dashboard Pareto |
| [L261](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/fixed_sequence.py#L261) | `aggregate_power_result` | Chooses single-node CSV integration, the multinode DCGM package, or native per-node SMI traces |
| [L345](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/fixed_sequence.py#L345) | `process_result` | Reads `<RF>.json`, writes `agg_<RF>.json`, merges the power audit |
| [L374](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/fixed_sequence.py#L374) | `process_multinode_results` (`--all`) | Parses `_conc{N}_gpus_{T}_ctx_{P}_gen_{D}` from filenames, checks them against `CONC_LIST`, and writes `result_processing_<RF>.json` |

Support modules: [`topology.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/topology.py#L12) (`Parallelism`), [`metadata.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/metadata.py#L8) (router and offload `{name, version}`), [`result_filename.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/result_filename.py#L22) (stems of ≤ 120 bytes with a hash suffix), [`artifacts.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/artifacts.py#L43) (identity keys used to validate reused sweeps).

### AgentX: [`results/agentic/`](https://github.com/SemiAnalysisAI/InferenceX/tree/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/agentic)

| Where | What it does |
| --- | --- |
| [`__init__.py` L141](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/agentic/__init__.py#L141) `build_result` | Identity + topology (`num_gpus`, prefill/decode shape) + `request_metrics` + `server_metrics` + `kv_cache_pool_tokens`; per-GPU throughput only when GPUs > 0 |
| [`process_agentic_result.py` L24](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/agentic/process_agentic_result.py#L24) | CLI: reads `aiperf_artifacts/{profile_export.jsonl, profile_export_aiperf.json, server_metrics_export.json}`, server logs and cached traces, and writes `<RF>.json` rounded to 5 dp |
| [`artifacts.py` L33](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/agentic/artifacts.py#L33) | Drops warmup and error records, keeping counts by category |
| [`request_metrics.py` L115](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/agentic/request_metrics.py#L115) | TTFT, E2EL, ITL in seconds; `intvty` = 1 / ITL; `e2e_norm_intvty` = OSL / E2EL; QPS from 1 s sliding windows (L161); throughput over max(end) − min(start) (L214) |
| [`server_metrics.py` L17](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/agentic/server_metrics.py#L17) + [`backends/`](https://github.com/SemiAnalysisAI/InferenceX/tree/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/agentic/backends) | Tries TRT-LLM, Dynamo-vLLM, ATOM, SGLang, vLLM in order. Maps each engine's Prometheus names to one schema: GPU/CPU/external cache hit rate, KV usage, offload bytes and bandwidth. KV pool capacity comes from logs or gauges (e.g. vLLM "GPU KV cache size: N tokens", TRT `max_blocks × tokens_per_block`) |
| [`validate_agentic_result.py` L48](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/agentic/validate_agentic_result.py#L48) | Fails if the error rate exceeds the threshold (10% in CI) or nothing completed |
| [`power_adapter.py` L78](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/agentic/power_adapter.py#L78) | Converts aiperf start and end times into a power window, then calls the power engines |

### Power: [`results/power/`](https://github.com/SemiAnalysisAI/InferenceX/tree/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/power)

| Where | What it does |
| --- | --- |
| [`__init__.py` L15–39](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/power/__init__.py#L15) | Metric keys (`avg_power_w`, `total_gpu_energy_j`, `joules_per_{input,output,total}_token`, role energies); `with_power_metrics` sets `power_valid` |
| [`single_node.py` L264](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/power/single_node.py#L264) `integrate_power` | Auto-detects the time, power and GPU-index columns. Keeps samples within ±3 s of the window; each GPU needs samples on both sides of the window and no gap over 3 s. Trapezoid integration with interpolated endpoints ([`common.py` L56](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/power/common.py#L56)); the energies are summed |
| [`single_node.py` L553](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/power/single_node.py#L553) | Advisory ±5% cross-check against the AMD energy accumulator |
| [`multinode.py` L920](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/power/multinode.py#L920) | Validates the srt-slurm `dcgm-power` package: pinned producer SHA, recomputed validity, and an exact match between window and benchmark times. Integrates energy per role |
| [`native_multinode.py` L133](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/power/native_multinode.py#L133), [`window.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/power/window.py#L32), [`audit.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/power/audit.py#L11) | Per-node SMI traces with stable device UUIDs, window files for srt-slurm, and the `power_audit` summary |

### Evals and collectors

| Where | What it does |
| --- | --- |
| [`evals.py` L141 / L305](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/evals.py#L141) | `extract_metrics`, `build_rows`: score priority is strict → accuracy → flexible; the latest attempt per concurrency wins |
| [`collect_results.py` L6](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/collect_results.py#L6) | Concatenates every `bmk_*` JSON into `agg_bmk.json` |
| [`collect_eval_results.py` L136](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/collect_eval_results.py#L136) | Finds `meta_env.json` directories and writes `agg_eval_all.json` plus markdown tables |
| [`compare_results.py` L99](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/compare_results.py#L99) | `BASELINE_QUERY` joins `benchmark_results` ⨝ `configs` ⨝ `workflow_runs` on hw, framework, prefix, precision, spec, topology, isl/osl and conc, taking the latest main run. A PR's step summary shows the deltas |
| [`generate_aiperf_plots.py` L765](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/infx/results/generate_aiperf_plots.py#L765) | 6×2 time-series panel of KV usage, queue, prefix hit, throughput, offload and TTFT |

Easy to miss: fixed-seq interactivity is 1000 / TPOT, but AgentX interactivity is 1 / ITL. The two scenarios measure it differently, so compare them only within a scenario.

## 8. `utils/aiperf` — the AgentX replay engine (SemiAnalysis fork)

[SemiAnalysisAI/aiperf](https://github.com/SemiAnalysisAI/aiperf/tree/754356e9a39acc6cc6afb242d123bb57c3fb6f75) (v0.12.0) is NVIDIA's aiperf plus an AgentX layer: a locked benchmark scenario, loaders for the weka traces, prompt synthesis from `hash_ids`, and an agentic replay timing strategy. The shallow submodule clone cannot show the exact diff against upstream, but the AgentX surface is identifiable by name. Start with its docs: [`docs/tutorials/agentx-mvp.md`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/docs/tutorials/agentx-mvp.md), [`docs/benchmark-modes/semianalysis-agentx-faq.md`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/docs/benchmark-modes/semianalysis-agentx-faq.md), and [`docs/architecture.md`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/docs/architecture.md).

Architecture in brief: `aiperf profile` ([`cli_commands/profile.py`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/cli_commands/profile.py#L14)) starts about 10 services that talk over ZMQ. The TimingManager issues "credits" through a sticky ROUTER ([`credit/sticky_router.py` L176](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/credit/sticky_router.py#L176)) to Workers ([`workers/worker.py` L469](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/workers/worker.py#L469)). Workers read conversations from a memory-mapped dataset, send HTTP requests, and push raw records to RecordProcessors and then the RecordsManager ([L434](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/records/records_manager.py#L434)). The ServerMetricsManager scrapes Prometheus alongside. Services are registered in [`plugin/plugins.yaml`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/plugin/plugins.yaml#L27) L27–128.

### AgentX path, in reading order

| Where | Symbol | What it does |
| --- | --- | --- |
| [`common/scenario/inferencex_agentx_mvp.py` L7](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/scenario/inferencex_agentx_mvp.py#L7) | `INFERENCEX_AGENTX_MVP` | The locked scenario: `AGENTIC_REPLAY` timing, streaming + `ignore_eos` required, no truncation, weka loaders only, ≥ 900 s duration, first-turn cache-bust marker, 10 s system-idle cap, ≥ 95% metric-coverage gate |
| [`common/scenario/validator.py` L61](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/scenario/validator.py#L61) | `apply_scenario` | Enforces the locks. A violation raises `ScenarioLockError`; `--unsafe-override` downgrades it to a warning and marks `submission_valid=False` |
| [`plugins.yaml` L2423](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/plugin/plugins.yaml#L2423) | `semianalysis_cc_traces_weka_062126` | Maps the `--public-dataset` alias to HF `semianalysisai/cc-traces-weka-062126` (393 traces, \~98.8k requests), block size 64. Older dated variants follow at L2141–2461 |
| [`dataset/loader/semianalysis_cc_traces_weka.py` L50](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/dataset/loader/semianalysis_cc_traces_weka.py#L50) | `SemiAnalysisCCTracesWekaLoader` | Downloads from HF and validates each row as `WekaTrace`. Filters by `--max-context-length` first, then caps with `--num-dataset-entries` (L136–212). Delegates conversion to the file loader |
| [`dataset/loader/weka_trace_models.py` L21–142](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/dataset/loader/weka_trace_models.py#L21) | `WekaNormalRequest`, `WekaStreamingRequest`, `WekaSubagentEntry`, `WekaTrace` | The strict (`extra=forbid`) schema written by `proxy_to_weka.py` |
| [`dataset/loader/weka_trace.py` L685](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/dataset/loader/weka_trace.py#L685) | `WekaTraceLoader` | One root conversation per trace, plus child conversations per subagent, linked by SPAWN/JOIN prerequisites. Inter-turn delay = tₖ − (tₖ₋₁ + api\_timeₖ₋₁), clamped at 0 (L131–157) |
| [`weka_trace.py` L1362](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/dataset/loader/weka_trace.py#L1362) | `_decode_block_tokens` | Prompt synthesis: re-seeds an RNG per `hash_id` and slices 64 tokens from a coding corpus. Identical hash IDs give byte-identical blocks, which reproduces the recorded prefix sharing on the server's KV cache |
| [`timing/trajectory_source.py` L223](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/trajectory_source.py#L223) | `TrajectorySource` | One trajectory per concurrency lane. Each lane joins its session at a random point t\* between 25% and 75% of the trace (L607–640), so the run starts in steady state, not at an empty cache |
| [`timing/strategies/agentic_replay.py` L103](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/strategies/agentic_replay.py#L103) | `AgenticReplayStrategy` | Warmup primes each lane with `max_tokens=1` primers up to t\*, then `--warmup-requests-per-lane` cache-pressure requests. Profiling follows. A lane recycles to a new trace only when its whole tree, subagents included, has drained |
| [`timing/replay_dependencies.py` L167](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/replay_dependencies.py#L167) | `ReplayBarrierCoordinator` | `--trace-idle-gap-cap-seconds` (300 in InferenceX): if a tree has had nothing in flight for longer than the cap, its next delay is cut to 0 |
| [`workers/session_manager.py` L24](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/workers/session_manager.py#L24) | `UserSession` | Multi-turn state keyed by `x_correlation_id`; accumulates the message history from per-turn deltas |
| [`server_metrics/manager.py` L49](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/server_metrics/manager.py#L49) | `ServerMetricsManager` | One collector per `--server-metrics` URL, scraping every 0.333 s with the `prometheus_client` parsers, then `server_metrics_export.json` ([`json_exporter.py` L28](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/server_metrics/json_exporter.py#L28)) |
| [`records/records_manager.py` L1730](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/records/records_manager.py#L1730) | coverage gate | TTFT/ITL observations must cover at least 95% of the profiling window |
| [`config/artifacts.py` L50](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/config/artifacts.py#L50) + [`record_models.py` L1648](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/models/record_models.py#L1648) | outputs | `profile_export.jsonl`: one record per request, with metadata (session, turn, source trace index, phase, timestamps, agent depth) and metrics. `profile_export_aiperf.json` holds the aggregates and `submission_valid` |

The InferenceX invocation ([`benchmark_lib.sh` L3236](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/benchmarks/benchmark_lib.sh#L3236)):

```bash
aiperf profile --scenario inferencex-agentx-mvp --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model $MODEL --tokenizer $MODEL --concurrency $CONC --benchmark-duration 3600 --random-seed 42 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 \
  --trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 --use-server-token-count --no-gpu-telemetry \
  --max-context-length $MAX_MODEL_LEN --num-dataset-entries 393 --slice-duration 1.0 \
  --server-metrics <worker /metrics URLs> --output-artifact-dir <dir>/aiperf_artifacts \
  --public-dataset semianalysis_cc_traces_weka_062126
```

Easy to miss:

- `--use-server-token-count` makes the server's usage counts authoritative for token metrics. The tokenizer is still needed to synthesise prompts.
- Possible mismatch, not run-verified: `benchmark_lib.sh` L3314–3321 appends `--use-dynamo-conv-aware-routing` for `dynamo-*` frameworks, but that flag does not appear anywhere in the pinned aiperf source. Many GB300 AgentX recipes disable it and use the `X-Dynamo-Session-ID` header path instead.

## 9. `AGENTS.md`, `.agents/skills/`, `.claude/commands/` — instructions for coding agents

Much of this repo is maintained by agents: Klaud Cold, `@claude` on PRs, and maintainers running Claude Code locally. These files are their operating manual. For a human they are also the most compact statement of the repo's invariants, so read `AGENTS.md` even if you never use an agent. `CLAUDE.md` just points to it.

### [`AGENTS.md`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/AGENTS.md) (129 lines)

| Lines | Rule set |
| --- | --- |
| L1–10 | Start at `docs/index.md` and load one focused guide. Source beats docs |
| L12–23 | PR policy: bilingual `English / 中文` titles, disclosure of the AI model used, one CODEOWNER checklist per PR, Ruff with all rules (line length 100), kebab-case YAML, shared bash lives in `benchmark_lib.sh` |
| L25–52 | Bash: configuration comes from the caller; no `${VAR:-default}`; validate with `check_env_vars`; no `set -u` |
| L54–66 | Delete deprecated configs rather than archive them. Exactly one `runners/launch_<pool>.sh` per pool, routed by runner-name prefix |
| L68–83 | srt-slurm hooks do host checks only, no engine patches. Acceptance length always comes from `golden_al_distribution` |
| L85–118 | Test quality: nine forbidden patterns (source grepping, AST inspection, tautologies, reimplementing the code under test…) |
| L120–127 | Benchmark invariants: exactly one `nodes:N` label, an append-only byte-exact changelog, recipe and master config change together with `model.container == image`, and "generated config is not proof of a working run" |

### Skills ([`.agents/skills/`](https://github.com/SemiAnalysisAI/InferenceX/tree/45aa3a24be4d97aad878076391ebd38690cce1cb/.agents/skills); `.claude/skills` symlinks here)

| Skill | Loop it teaches |
| --- | --- |
| [`debug-runs`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.agents/skills/debug-runs/SKILL.md) | Label `full-sweep-fail-fast` or dispatch `e2e-tests.yml` for one key. Then `gh run watch`, `--log-failed` + grep for signatures, reproduce on the node (`salloc`/`srun`, `enroot exec`), fix, re-run. Never merge; report the perf delta vs main |
| [`debug-agentx-runs`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.agents/skills/debug-agentx-runs/SKILL.md) | For hour-long AgentX runs: find the Slurm WorkDir, tail `benchmark.out` and worker logs, check every `/metrics` URL (KV%, prefix hit, queue), estimate ETA from phase markers, cancel bad runs early (ask first) |

### Commands ([`.claude/commands/`](https://github.com/SemiAnalysisAI/InferenceX/tree/45aa3a24be4d97aad878076391ebd38690cce1cb/.claude/commands))

| Command | What it automates |
| --- | --- |
| [`add-model-hardware`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.claude/commands/add-model-hardware.md) | New single-node recipe from the nearest sibling, plus master entry with `srt-recipe:` and changelog entry. Validates with `infx.matrix.generate test-config`, opens a PR, labels `full-sweep-fail-fast`. The best worked example of adding a model or GPU |
| [`nuke`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.claude/commands/nuke.md) | Bulk vLLM/SGLang image bump: one PR per model + precision + SKU, after checking the tag exists on Docker Hub |
| [`merge-prs`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.claude/commands/merge-prs.md), [`find-mergeable-claude-prs`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.claude/commands/find-mergeable-claude-prs.md), [`list-claude-pr-status`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.claude/commands/list-claude-pr-status.md), [`klaud-pr-status-html`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.claude/commands/klaud-pr-status-html.md) | PR triage and merging, using `infx.workflows.merge_with_reuse` so GPU results are reused |
| [`fix-klaud-cron-prs`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.claude/commands/fix-klaud-cron-prs.md) | Diagnose failing bot PRs from sweep logs and propose a minimal fix; ask before pushing |
| [`recover-failed-ingest`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.claude/commands/recover-failed-ingest.md) | Re-ingest a failed main run through a recovery PR that reuses the artifacts, with no GPU rerun |
| [`clean-amd-mi355-runner-root-files`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.claude/commands/clean-amd-mi355-runner-root-files.md), [`debug-mi300-enroot-pyxis`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.claude/commands/debug-mi300-enroot-pyxis.md) | AMD fleet runbooks: root-owned files after cancelled jobs, and pyxis user-namespace failures from an AppArmor sysctl on Ubuntu 24.04 |

The prompts that drive the CI agents are also worth reading. [`.github/klaud-candidate-prompt.md`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.github/klaud-candidate-prompt.md) is the image-bump agent's brief: edit only the image and compatibility flags, smoke test first, repair budget, `/use` only when validated. [`.github/codeowner-signoff-verify-prompt.md`](https://github.com/SemiAnalysisAI/InferenceX/blob/45aa3a24be4d97aad878076391ebd38690cce1cb/.github/codeowner-signoff-verify-prompt.md) has 15 review checks, including no engine patches without a waiver, golden AL, draft-as-shipped, and Pareto coverage. Together they are the review checklist a PR must pass.
