## DeepSeek V4 KV Cache 架构解析
DeepSeek V4 采用了 MLA (Multi-head Latent Attention) + 滑动窗口 + KV 压缩 的混合 KV cache 策略，这与传统 Transformer 的 KV cache 完全不同。

### 核心设计
从 model.py 中可以看到关键代码：

1. MLA 低秩压缩 ：KV 不是按 n_heads × head_dim 存储，而是投影到一个低秩隐向量，维度为 head_dim = 512 （远小于传统 n_heads × head_dim = 64 × 512 = 32768 ）。

2. 每层的 KV cache 由两部分组成 （见 model.py:473-474 ）：

```
kv_cache_size = window_size + (max_seq_len // compress_ratio if compress_ratio 
else 0)
self.register_buffer("kv_cache", torch.zeros(max_batch_size, kv_cache_size, 
head_dim))
```
- 滑动窗口部分 ：存储最近 window_size = 128 个 token 的完整 KV
- 压缩部分 ：当 compress_ratio > 0 时，存储 max_seq_len // compress_ratio 个压缩后的 KV token
3. 每层压缩比不同 （见 config.json ）：

```
compress_ratios = [0, 0, 4, 128, 4, 128, 4, 128, ..., 4, 128, 4, 0]
```
- compress_ratio = 0 ：纯滑动窗口，不压缩
- compress_ratio = 4 ：4 个 token 压缩为 1 个（重叠窗口压缩）
- compress_ratio = 128 ：128 个 token 压缩为 1 个
4. Indexer 的额外 KV cache （仅 compress_ratio = 4 的层，见 model.py:399 ）：

```
self.register_buffer("kv_cache", torch.zeros(max_batch_size, max_seq_len // 
compress_ratio, index_head_dim))
```
## KV Cache 大小计算公式
### 基本参数定义
参数 符号 默认值 最大序列长度 S 1,048,576 最大批大小 B 4 层数 L 43 滑动窗口大小 W 128 MLA head_dim D 512 Indexer head_dim D_idx 128 第 i 层压缩比 R_i 见 compress_ratios KV 数据类型字节数 bytes 2 (bf16)

### 单层 KV Cache 大小
对于第 i 层：
 (a) 主 KV Cache（Attention 用） [ o bj ec tO bj ec t ] kv_tokens i ​ = W + { 0 ⌊ S / R i ​ ⌋ ​ if R i ​ = 0 if R i ​ > 0 ​ [ o bj ec tO bj ec t ] kv_size i ​ = B × kv_tokens i ​ × D × bytes (b) Indexer KV Cache（仅 R_i = 4 的层） [ o bj ec tO bj ec t ] idx_size i ​ = B × ⌊ S / R i ​ ⌋ × D idx ​ × bytes (c) Compressor 状态缓冲区（仅 R_i > 0 的层）
对于 R_i = 4 （有 overlap， coff = 2 ）：
 [ o bj ec tO bj ec t ] comp_state i ​ = B × ( 2 × R i ​ ) × ( 2 × D ) × 4 (fp32)
对于 R_i = 128 （无 overlap， coff = 1 ）：
 [ o bj ec tO bj ec t ] comp_state i ​ = B × R i ​ × D × 4 (fp32)
### 总 KV Cache 大小 [ o bj ec tO bj ec t ] Total = i = 0 ∑ L − 1 ​ ( kv_size i ​ + idx_size i ​ + comp_state i ​ ) ​
## 具体计算示例（使用默认配置）
以 S = 1,048,576 、 B = 4 、 bytes = 2 (bf16) 为例：

compress_ratios 分布 ：43 层中

- 2 层 R = 0 （第 0、1 层）
- 1 层 R = 0 （第 42 层，最后一层）
- 20 层 R = 4
- 20 层 R = 128
### R = 0 的层（3 层）
```
kv_tokens = 128 + 0 = 128
kv_size = 4 × 128 × 512 × 2 = 524,288 bytes ≈ 0.5 MB/层
```
### R = 4 的层（20 层）
```
kv_tokens = 128 + 1,048,576/4 = 128 + 262,144 = 262,272
kv_size = 4 × 262,272 × 512 × 2 = 1,073,741,824 bytes ≈ 1,024 MB/层

idx_size = 4 × 262,144 × 128 × 2 = 268,435,456 bytes ≈ 256 MB/层

comp_state = 4 × 8 × 1024 × 4 = 131,072 bytes ≈ 0.125 MB/层（可忽略）
```
### R = 128 的层（20 层）
```
kv_tokens = 128 + 1,048,576/128 = 128 + 8,192 = 8,320
kv_size = 4 × 8,320 × 512 × 2 = 34,078,720 bytes ≈ 32.5 MB/层

comp_state = 4 × 128 × 512 × 4 = 1,048,576 bytes ≈ 1 MB/层（可忽略）
```
### 总计
类型 层数 单层大小 小计 R=0 主KV 3 ~0.5 MB ~1.5 MB R=4 主KV 20 ~1,024 MB ~20,480 MB R=4 Indexer 20 ~256 MB ~5,120 MB R=128 主KV 20 ~32.5 MB ~650 MB 总计 ~26,252 MB ≈ 25.6 GB

## 简化公式（快速估算）
如果你只需要快速估算，可以忽略 Compressor 状态和 Indexer，用以下简化公式：
 [ o bj ec tO bj ec t ] Total ≈ B × D × bytes × i = 0 ∑ L − 1 ​ ( W + R i ​ S ​ )
其中 R_i = 0 时 S/R_i 项为 0。

如果加上 Indexer 的贡献：
 [ o bj ec tO bj ec t ] Total ≈ B × bytes × [ D × i ∑ ​ ( W + R i ​ S ​ ) + D idx ​ × N R = 4 ​ × 4 S ​ ]
其中 N_{R=4} 是 compress_ratio = 4 的层数。

## 关键洞察
1. MLA 大幅降低了 KV cache ：传统模型每 token 存 2 × n_heads × head_dim = 2 × 64 × 512 = 65,536 字节（bf16），DeepSeek V4 只存 512 字节（bf16），压缩了 128 倍 。
2. R=4 的层是内存瓶颈 ：它们占了总 KV cache 的 ~97%。如果需要进一步优化，这是重点目标。
3. R=128 的层极其高效 ：128 个 token 压缩为 1 个，长序列下每层只需 ~32 MB。
4. 实际部署中 ， S 应该用实际最大序列长度而非 max_position_embeddings ， B 用实际并发数，这样计算出的才是真实需求。
