# vLLM-Ascend 异步调度原理与方案

## 摘要

vLLM-Ascend 是构建在上游 vLLM V1 引擎之上的昇腾（Ascend NPU）硬件插件。其异步调度（Async Scheduling）能力继承自上游 vLLM 的 `AsyncScheduler`，并在此基础上针对昇腾硬件做了大量定制与优化。本文介绍其背景、核心原理、整体方案架构，以及从昇腾自有调度器演进到上游异步调度器的完整过程。

---

## 1. 背景

### 1.1 vLLM V1 的调度-执行模型

vLLM V1 引擎采用经典的「CPU 调度 + 设备执行」分离架构：

1. **CPU 侧调度器**（`Scheduler`）根据请求队列、KV Cache 余量、token 预算，决定本步调度哪些请求、分配哪些 KV block。
2. **设备侧 worker**（`GPUModelRunner` / `NPUModelRunner`）执行模型前向、采样，产出新 token。
3. 采样结果需要传回 CPU 侧，供调度器维护请求状态（已生成 token 数、是否结束等），并为下一步组装 `input_ids`。

### 1.2 同步调度的瓶颈

在传统**同步调度**模式下，每一步的流程是一个串行闭环：

```
调度 → 下发执行 → 等待执行完成 → 采样结果 device→host 拷贝 → 更新 CPU 状态 → 下一步调度
```

这里存在两个关键问题：

1. **调度循环被设备→主机拷贝阻塞**。每一步调度器都必须拿到采样 token 才能继续，而 device→host 的同步拷贝（`tensor.item()`、`cpu()`、`.numpy()` 等）会强制等待设备执行完成。
2. **host-device 往返延迟放大**。在执行新一步之前，`input_ids` 需要在 CPU 上组装再拷贝回设备，形成「设备→CPU→设备」的往返。

在大批量 decode、长序列、投机解码（Speculative Decoding）等场景下，这种同步开销会占整体时延的很大比例，成为吞吐瓶颈。

### 1.3 昇腾硬件的特殊性

昇腾 NPU 上，device→host 同步的开销相比 GPU 更为昂贵。vLLM-Ascend 的开发规范中明确要求：

> `tensor.item()` 在设备张量上会触发同步的 NPU→CPU 数据传输，在热路径中会严重阻塞 `AsyncScheduler`。

因此，vLLM-Ascend 把「消除热路径中的 host-device 同步」作为调度与执行优化的核心目标。

---

## 2. 核心原理

异步调度解决上述问题的根本思路是：

> **调度器不再等待本步的采样输出，采样获得的 token 保持在设备侧，下一步的输入直接在设备上组装。**

### 2.1 关键技术点

#### (1) 采样 token 常驻设备：`prev_sampled_token_ids`

在异步模式下，worker 采样得到的 token id 不立即拷贝回 CPU，而是保留在设备上的 `prev_sampled_token_ids` 缓冲区中。下一步组装 `input_ids` 时，直接从该缓冲区在设备上拷贝，避免了「设备→CPU→设备」的往返。

对应代码（上游 `vllm/v1/worker/gpu_input_batch.py`）：

- `set_async_sampled_token_ids()`：写入上一步采样结果
- `update_async_output_token_ids()`：更新异步输出 token
- `sampled_token_ids_cpu`：仅在真正需要（如结构化输出、返回 token id）时才拉回 CPU

#### (2) 乐观占位：`num_output_placeholders`

由于调度器在采样结果回来之前就要调度下一步，它需要**乐观地假设**每个请求将要生成多少个 token，从而为这些 token 预留 `input_ids` 的位置。这个预留计数就是 `request.num_output_placeholders`。

在上游 `AsyncScheduler._update_after_schedule()` 中：

```python
request.num_output_placeholders += (
    self.num_sampled_tokens_per_step + cur_num_spec_tokens
)
```

即每调度一步，就为「基础采样 token 数 + 投机 token 数」预留占位。当真实的 token 在后续步骤到达时，`_update_request_with_output()` 再递减该计数：

```python
request.num_output_placeholders -= len(new_token_ids)
```

#### (3) 抢占请求的过期 token 处理：`async_tokens_to_discard`

异步模式下，请求可能被强占（force-preempt）而 `reset_prefix_cache`。此时设备上仍有尚未回传、但已过期的一步输出帧。`_update_request_with_output()` 通过 `async_tokens_to_discard` 计数器，每次调用丢弃一个过期的 in-flight 帧，直到排空。

---

## 3. 方案架构

异步调度贯穿「调度器 → 执行器 → worker」三层，vLLM-Ascend 在每一层都有定制。

### 3.1 调度器层

#### 上游基础能力

- **开关**：`SchedulerConfig.async_scheduling`（CLI `--async-scheduling` / `--no-async-scheduling`，默认 None = 自动启用）。
- **选类**：`SchedulerConfig.get_scheduler_cls()` 根据 `async_scheduling` 选择 `AsyncScheduler` 或 `Scheduler`。
- **核心实现**：`vllm/v1/core/sched/async_scheduler.py` 中的 `AsyncScheduler(Scheduler)`，通过 `num_output_placeholders`、`_spec_token_placeholders` 实现乐观占位。

#### vllm-ascend 定制调度器

vLLM-Ascend 通过 `NPUPlatform.check_and_update_config()` 设置 `scheduler_cls`，注入昇腾定制的调度器：

| 调度器 | 类 | 触发条件 | 场景 |
|--------|-----|----------|------|
| 异步重计算调度器 | `AsyncRecomputeScheduler(AsyncScheduler, RecomputeScheduler)` | `recompute_scheduler_enable` | P/D 分离（重计算） |
| 均衡调度器 | `BalanceScheduler` | `enable_balance_scheduling` | PD 混合（prefill/decode 均衡） |
| 动态批量调度器 | `SchedulerDynamicBatch` | `SLO_limits_for_dynamic_batch != -1` | 基于 SLO 的动态 token 预算 |
| 分块流水调度器 | `ProfilingChunkScheduler` | `profiling_chunk_config.enabled` | 动态分块 PP（CPP） |

其中 `AsyncRecomputeScheduler` 是典型的**异步 + 昇腾定制**组合：继承上游 `AsyncScheduler` 的异步占位逻辑，叠加昇腾重计算的 P/D 抢占通知逻辑。

### 3.2 执行器层

- `MultiprocExecutor.use_async_scheduling` 标志。
- 异步模式下，一个守护线程 `WorkerAsyncOutputCopy`（`async_output_busy_loop`）持续从 `async_output_queue` 取出 worker 输出，使「输出卸下 + 拷贝」与「模型执行」并行。
- `handle_output()` 在异步模式下把输出路由到异步队列。
- `supports_async_scheduling()` 判断当前执行器后端是否支持异步调度。

### 3.3 Worker / Model Runner 层

- `GPUModelRunner.use_async_scheduling = scheduler_config.async_scheduling`。
- 采样结果封装为 `AsyncGPUModelRunnerOutput`，采样 token 在副作用流（side stream）上拷贝，拷贝完成事件被锁存，引擎只在真正需要时才读取。
- **PP（流水线并行）+ 异步**：`_pp_broadcast_prev_sampled_token_ids()` / `_pp_receive_prev_sampled_token_ids_to_input_batch()` 在 PP rank 之间直接搬运采样 token id，绕开引擎的 IPC 路径。

### 3.4 vLLM-Ascend NPU 定制

核心 NPU 技术是**避免热路径中的 device→CPU 同步**：

1. **采样 token 常驻设备**：`prev_sampled_token_ids`（见 `_bookkeeping_sync`、`gpu_input_batch.py`）。
2. **异步拷贝流**：`async_output_copy_stream`、`draft_token_ids_copy_stream`、`CpuGpuBuffer` + `copy_to_gpu(non_blocking)`（`pcp_utils.py`、`model_runner_v1.py`）。
3. **`AsyncGPUModelRunnerOutput`**：采样 token/logprobs 在副作用流上拷贝，事件被锁存，引擎按需读取。
4. **延迟状态修正**：`deferred_state_corrections_fn` 在 forward 启动后执行，避免 forward 期间的 host 往返（例如 GDN 注意力相关的修正）。
5. **PCP 管理器**（`PCPManager`）：`generate_pcp_mtp_input()` 在异步调度下，通过从「设备常驻/CPU 镜像」散射上一步采样 token 与 draft token 来填充下一步 PCP 输入，避免调度往返。
6. **算力↔通信 overlap**：`prefill_comm_compute_overlap`、`multistream_dsv4_dsa_overlap`、`multistream_overlap_shared_expert`、layer-sharding 异步权重广播、KV cache 异步 D2H/H2D 卸载等。

---

## 4. 与 DeepSeek V4 的结合

针对 DeepSeek V4，vLLM-Ascend 在异步调度之上叠加了若干多流 overlap 优化：

- **DSA 多流 overlap**：compressor、indexer-select、CV 并行、纯 prefill 算力-通信 overlap（PR #9450、#9441、#9433、#9504）。
- **IndexCache 复用 `topk_indices`**：跨 decode 步复用 DeepSeek V4 索引计算结果，减少重复计算（#9390）。
- **共享专家 overlap**：FusedMoE 中共享专家计算的细粒度 overlap（#5962、#9413）。
- **`enable_async_exponential`**：异步指数采样 overlap（#4501）。

这些优化与异步调度是**正交协同**的关系：异步调度消除了 host-device 同步的「纵向」气泡，多流 overlap 消除了设备内部计算-通信的「横向」串行。

---

## 5. 方案演进

结合 vllm-ascend 的 release notes，异步调度方案经历了如下演进：

### 5.1 阶段一：昇腾自有调度器（v0.x 早期）

V1 引擎初期，vLLM-Ascend 维护了一套昇腾自有的调度器与多步调度器（`ascend_scheduler`、multi-step scheduler），以最大化昇腾硬件亲和度。该阶段异步能力是昇腾定制的、与上游相对独立。

### 5.2 阶段二：迁移到上游 `AsyncScheduler`（v0.11 → v0.12）

- 「Ascend scheduler 已废弃，将在 v0.12.0rc1 移除」（#4623）。
- 随后 Ascend scheduler 被彻底移除，`--additional-config {"ascend_scheduler": ...}` 不再生效。
- 迁移到上游 `AsyncScheduler` 后，通过 `scheduler_cls` 注入昇腾定制逻辑，异步能力「更稳定、可直接启用」（文档明确指出在 v0.12.0 中 EAGLE 与 `async_scheduling=True` 组合更稳定）。

### 5.3 阶段三：DeepSeek 特定异步优化

- DeepSeek-R1 & DeepSeek-V3.2 的异步调度增强（#3631、#3900、#3908、#4191、#4805）。
- **消除异步调度中的 HD 同步**（#4805）。
- 异步调度 async copy 修复、消除 hang（#4113、#4233）。
- 模型执行期间异步指数采样（#4501）。

### 5.4 阶段四：Zero-bubble 与 PP 支持（近期）

- **Zero Bubble Async Scheduling**：异步调度与投机解码的零气泡优化（#7640）。
- 异步调度下发气泡优化（#8766）。
- [实验] Pipeline Parallel 支持异步调度（#7136）。

### 5.5 阶段五：持续 overlap 优化（最新）

- DeepSeek V4 DSA 多流 overlap 系列（compressor、indexer-select、CV 并行、纯 prefill 通算 overlap：#9450、#9441、#9433、#9504）。
- 细粒度共享专家 overlap（#5962）。
- 历史遗留的 DeepSeek「双批 overlap（DBO）」（`VLLM_ASCEND_ENABLE_DBO=1`，#941）已被后续更通用的 overlap 机制取代。

### 演进主线总结

```
昇腾自有调度器（定制、独立）
      │  废弃 / 移除
      ▼
上游 AsyncScheduler + scheduler_cls 注入（标准化）
      │  昇腾 NPU 定制（常驻 device、异步拷贝、延迟修正）
      ▼
DeepSeek 特定优化（消除 HD 同步、async copy 修复）
      │  零气泡 / PP
      ▼
Zero-bubble Async Scheduling + 多流 overlap（当前）
```

---

## 6. 关键配置

### 6.1 上游配置

- `--async-scheduling` / `--no-async-scheduling`：显式开关异步调度（默认自动）。

### 6.2 vLLM-Ascend `additional_config`

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `recompute_scheduler_enable` | False | 启用 `RecomputeScheduler`/`AsyncRecomputeScheduler`（仅 P/D 分离） |
| `enable_balance_scheduling` | False | 启用 `BalanceScheduler`（仅 PD 混合） |
| `SLO_limits_for_dynamic_batch` | -1 | 启用 `SchedulerDynamicBatch` |
| `profiling_chunk_config` | `{enabled:false}` | 启用 `ProfilingChunkScheduler`（CPP） |
| `enable_async_exponential` | False | 异步指数采样 overlap |
| `multistream_overlap_shared_expert` | False | 共享专家多流 overlap |
| `multistream_overlap_gate` | False | gate 多流 overlap |
| `prefill_comm_compute_overlap` | False | prefill 算力↔通信 overlap |
| `multistream_dsa_preprocess` | False | DSA 预处理多流 overlap |
| `multistream_dsv4_dsa_overlap` | False | DeepSeek V4 DSA 多流 overlap |

---

## 7. 关键源码索引

### 上游 vLLM（异步基础）

| 文件 | 角色 |
|------|------|
| `vllm/v1/core/sched/async_scheduler.py` | `AsyncScheduler`，乐观占位核心 |
| `vllm/v1/core/sched/scheduler.py` | 基类 `Scheduler`，异步分支 |
| `vllm/v1/worker/gpu_input_batch.py` | `prev_sampled_token_ids` 等常驻设备机制 |
| `vllm/v1/worker/gpu_model_runner.py` | `AsyncGPUModelRunnerOutput`、PP 采样广播 |
| `vllm/v1/executor/multiproc_executor.py` | `async_output_queue`、异步输出守护线程 |
| `vllm/config/scheduler.py` | `async_scheduling` 开关与 `get_scheduler_cls()` |

### vLLM-Ascend（定制）

| 文件 | 角色 |
|------|------|
| `vllm_ascend/core/recompute_scheduler.py` | `AsyncRecomputeScheduler` |
| `vllm_ascend/core/scheduler_dynamic_batch.py` | `SchedulerDynamicBatch` |
| `vllm_ascend/core/scheduler_profiling_chunk.py` | `ProfilingChunkScheduler` |
| `vllm_ascend/core/profiling_chunk_predictor.py` | `ChunkSizePredictor` |
| `vllm_ascend/patch/platform/patch_balance_schedule.py` | `BalanceScheduler` |
| `vllm_ascend/platform.py` | 调度器安装（`scheduler_cls` 注入） |
| `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner` 异步执行 |
| `vllm_ascend/worker/pcp_utils.py` | `PCPManager` 异步输入填充 |
| `vllm_ascend/patch/platform/patch_multiproc_executor.py` | `AscendMultiprocExecutor` |

---

## 8. 总结

vLLM-Ascend 的异步调度，本质上是**「调度器不阻塞等待采样输出」**这一思想在昇腾硬件上的落地：

1. **上游提供标准化的异步调度框架**（`AsyncScheduler`、`num_output_placeholders`、设备常驻采样 token）。
2. **昇腾通过 `scheduler_cls` 注入定制调度器**，覆盖 P/D 分离、动态批量、分块流水等场景。
3. **worker 侧做 NPU 定制**，彻底消除热路径中的 host-device 同步，并用多流 overlap 消除设备内部气泡。

其演进主线是从「昇腾自有、与上游割裂」的路线，逐渐收敛到「复用上游标准框架 + 昇腾定制优化」的分层架构，最终形成当前「Zero-bubble 异步调度 + 多流 overlap」的方案。