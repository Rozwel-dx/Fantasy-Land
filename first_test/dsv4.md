# DeepSeek V4 Indexer 模块深度解析
## 一、核心思想
### 1.1 设计背景
DeepSeek V4 作为新一代 MoE（Mixture of Experts）大模型，在处理超长序列时面临着巨大的计算挑战。传统的 Full Attention 机制复杂度为 [ o bj ec tO bj ec t ] O ( n 2 ) ，当序列长度达到 64K 甚至 128K 时，计算量和内存占用都会呈平方级增长。

Indexer 模块的核心目标 ：通过 稀疏注意力机制 ，从海量 Key 序列中快速筛选出与 Query 最相关的 Top-K 个位置，将注意力计算复杂度从 [ o bj ec tO bj ec t ] O ( n 2 ) 降低到 [ o bj ec tO bj ec t ] O ( n ⋅ K ) ，其中 [ o bj ec tO bj ec t ] K 为选中的稀疏度（DeepSeek V4 中固定为 2048）。

### 1.2 关键设计理念
设计原则 实现方式 技术价值 块级稀疏 基于 Block Table 管理有效 Key 块 跳过 padding 区域，减少无效计算 两阶段计算 Prolog（预处理）+ Indexer（选择） 算子融合，提升计算效率 分级 TopK 2K/8K/128K 自适应算法路径 针对不同序列长度优化 量化加速 INT8 量化 Query 投影 减少内存带宽占用

## 二、完整流程
### 2.1 整体架构
```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DeepSeek V4 Indexer 完整流程                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐     ┌─────────────────────┐                    │
│  │  Stage 1: Prolog     │────▶│  Stage 2: Indexer    │                    │
│  │  (Query 预处理)      │     │  (TopK 索引选择)     │                    │
│  └─────────────────────┘     └─────────────────────┘                    │
│           │                            │                                │
│           ▼                            ▼                                │
│  ┌─────────────────────┐     ┌─────────────────────┐                    │
│  │ 1. LoRA 投影        │     │ 1. 块级 Matmul      │                    │
│  │ 2. RoPE 编码        │     │ 2. ReLU + Weight    │                    │
│  │ 3. Hadamard 变换    │     │ 3. 行求和聚合        │                    │
│  │ 4. 量化输出         │     │ 4. 分级 TopK 选择   │                    │
│  └─────────────────────┘     └─────────────────────┘                    │
│           │                            │                                │
│           ▼                            ▼                                │
│  ┌─────────────────────┐     ┌─────────────────────┐                    │
│  │ 输出: q, weights,   │     │ 输出: selectedIndices│                    │
│  │       q_scale       │     │  (B, S1, N2, 2048)  │                    │
│  └─────────────────────┘     └─────────────────────┘                    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Stage 3: IndexerAttention (稀疏注意力计算，使用 selectedIndices)   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
### 2.2 Stage 1: Prolog 预处理
Prolog 阶段完成 Query 向量的预处理，为后续 Indexer 计算提供输入。

计算公式 ：

```
q_tmp = qr @ idx_wq_b · qr_scale · idx_wq_b_scale

q_hadamard = Cat({q_tmp[:, :nope_dim], Rope(q_tmp[:, nope_dim:])}, -1) @ hadamard

q, q_scale = Quant(q_hadamard)

weights = x @ weights_proj · (1/√(idx_nq · head_dim))
```
输入输出说明 （来自 Indexer_Prolog.md ）：

输入 类型 Shape 说明 qr INT8 [t, q_lora_rank] LoRA 投影输入 idx_wq_b INT8 [q_lora_rank, idx_nq*head_dim] 量化权重 x BF16 [t, h] Transformer 层输出 cos/sin BF16 [t, rope_dim] RoPE 编码系数 hadamard BF16 [head_dim, head_dim] Hadamard 变换矩阵

输出 类型 Shape 说明 q INT8 [t, idx_nq*head_dim] 量化后的 Query weights FP16 [t, idx_nq] 注意力权重 q_scale FP16 [t, idx_nq] 反量化 scale

### 2.3 Stage 2: Indexer 核心计算
Indexer 阶段是稀疏注意力的核心，实现从海量 Key 中选择 Top-K 相关位置。

输入输出 ：

输入 类型 Shape 说明 query BF16 [B, S1, indexN1, indexD] 查询向量 key BF16 [blockNum, blockSize, n2, indexD] 键向量（分块存储） weights BF16 [B, S1, indexN1] 权重向量 actSeqKey INT32 [B] 实际序列长度 blockTable INT32 [B, maxBlockNum] 有效块映射表

输出 类型 Shape 说明 selectedIndices INT32 [B, S1, N2, 2048] 选中的 Top-K 索引

核心计算流程 ：

```
┌────────────────────────────────────────────────────────────────┐
│                    Indexer 核心计算流程                         │
├────────────────────────────────────────────────────────────────┤
│  1. Reshape 4D → 2D                                            │
│     query [B,S1,N1,D] → query2D [B*S1*N1, D]                   │
│     key [blockNum,blockSize,n2,D] → key2D [blockNum*blockSize, n2*D] │
├────────────────────────────────────────────────────────────────┤
│  2. 三重循环遍历 (Batch × S1 × N2)                              │
│     ├── 计算有效序列长度: effSeq = curSeq - casualOffset        │
│     ├── 计算有效块数: actBlock = ceil(effSeq / blockSize)       │
│     └── 块级注意力计算:                                          │
│         curQ [group, D] × curK [D, blockSize] → score [group, blockSize] │
│         score = ReLU(score) × weights                           │
│         localSum += RowSum(score)                               │
├────────────────────────────────────────────────────────────────┤
│  3. 分级 TopK 选择                                              │
│     ├── effSeq ≤ 2K: 直接 TopKSort                             │
│     ├── 2K < effSeq ≤ 8K: 直接 TopKSort                        │
│     └── effSeq > 8K: 两级归并 (128K→32K→8K→2K)                  │
└────────────────────────────────────────────────────────────────┘
```
### 2.4 分级 TopK 策略
针对不同序列长度，Indexer 采用自适应的 TopK 算法：

序列长度范围 算法策略 设计考量 ≤ 2048 直接排序 数据量小，直接排序效率最高 (2048, 8192] 直接排序 8K 在单个 Tile 内可处理 > 8192 两级归并 避免 O(n log n) 复杂度爆炸

>8K 时的两级归并流程 ：

```
128K 序列 → 每 8K 块独立排序 → 每个块选 top-2048 → 得到 16 × 2048 = 32K 候选
    ↓
32K 候选 → 每 8K 块独立排序 → 每个块选 top-2048 → 得到 4 × 2048 = 8K 候选
    ↓
8K 候选 → 排序 → 选 top-2048 → 最终结果
```
## 三、关键代码段分析
### 3.1 算子注册与参数配置
文件位置 ： ops/pypto/src/lightning_indexer_pto/op_kernel/lightning_indexer_impl.cpp

```
struct LightningIndexerPtoParams {
    int b = -1;                    // Batch Size
    int s1 = -1;                   // Query 序列长度
    int blockNum = -1;             // Key 块数量
    int maxBlockNum = -1;          // 最大块数
    int blockSize = 128;           // 块大小（固定）
    int indexNHeads = 64;          // 索引头数
    int indexHeadDim = 128;        // 索引头维度
    int n1 = 64;                   // 输入特征维度
    int n2 = 1;                    // 输出特征维度
    DataType dType = DT_BF16;      // 数据类型
    int selectedCount = 2048;      // 选中数量（固定）
};

void DynamicLightningIndexerPto(uint64_t configKey) {
    // 配置优化选项
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, true);
    config::SetRuntimeOption(MACHINE_SCHED_MODE, 
        static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH) |
        static_cast<uint8_t>(MachineScheduleConfig::MULTI_CORE_FAIR_SCH));
    
    // 算子注册
    FUNCTION("LightningIndexer", {query, key, weights, actualSeqLengthsKey, 
    blockTable}, {selectedIndices}) {
        LightningIndexerTopk(query, key, weights, actualSeqLengthsKey, 
                            blockTable, selectedIndices, params.selectedCount, 
                            indexerConfig);
    }
}

REGISTER_OP(LightningIndexerPto)
    .ImplFunc({{Lightning_Indexer_PTO_ConfigKey, DynamicLightningIndexerPto}});
```
关键配置解析 ：

- L2CACHE_AFFINITY_SCH ：启用 L2 缓存亲和性调度，减少缓存抖动
- MULTI_CORE_FAIR_SCH ：多核公平调度，平衡各核心负载
- CODEGEN_EXPRESSION_FUSION ：表达式融合，减少中间结果写入
### 3.2 块级注意力计算核心
文件位置 ： ops/pypto/src/lightning_indexer_pto/op_kernel/lightning_indexer_topk.cpp

```
void LightningIndexerTopkImpl(const Tensor &query, const Tensor &key, const 
Tensor &weights, 
    const Tensor &actSeqKey, const Tensor &blockTable, Tensor &topkRes, 
    const int selectedCount, IndexerTile tileConfig, std::set<int> unrollList, 
    Tensor *tmpOut, Tensor *topkValue) {
    
    // 符号化维度
    SymbolicScalar b = GetInputShape(query, 0);
    SymbolicScalar s1 = GetInputShape(query, 1);
    SymbolicScalar blockNum = GetInputShape(key, 0);
    
    auto indexN1 = query.GetShape()[SHAPE_DIM2];
    auto indexD = query.GetShape()[SHAPE_DIM3];
    auto blockSize = key.GetShape()[1];
    auto n2 = key.GetShape()[SHAPE_DIM2];
    auto group = indexN1 / n2;  // 分组数
    
    // 核心三重循环
    LOOP("INDEX_LOOP_BATCH", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(b)) {
        auto curSeq = GetTensorData(actSeqKey, {bIdx});
        
        LOOP("INDEX_LOOP_S1", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(s1)) {
            auto casualOffset = s1 - s1Idx - 1;  // 因果偏移（自回归推理）
            auto effSeq = curSeq - casualOffset;
            auto actBlock = (effSeq + blockSize - 1) / blockSize;  // 有效块数
            
            LOOP("INDEX_LOOP_N2", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange
            (n2)) {
                auto bs1n2Offset = bIdx * s1 * n2 + s1Idx * n2 + n2Idx;
                auto qOffset = bIdx * s1 * indexN1 + s1Idx * indexN1 + n2Idx * 
                group;
                
                // 展开处理模板
                auto unrollingProcess = [&](int unrollLength, auto &&
                firstBlockIdx) {
                    auto curQ = View(query2D, {group, indexD}, {qOffset, 0});
                    std::vector<Tensor> concatSrcs;
                    
                    for (int subblockIdx = 0; subblockIdx < unrollLength; 
                    subblockIdx++) {
                        auto blockIdx = firstBlockIdx + subblockIdx;
                        SymbolicScalar curBlockIdx = GetTensorData(blockTable, 
                        {bIdx, blockIdx});
                        
                        // 获取当前 Key 块（处理边界情况）
                        auto curK = View(key2D, {blockSize, indexD},
                            {std::min(blockSize, effSeq - (blockIdx * blockSize)), 
                            indexD},
                            {curBlockIdx * blockSize, n2Idx * indexD});
                        
                        // Matmul + ReLU + Weight + RowSum 融合
                        TileShape::Current().SetCubeTile(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile
                            [4], c1Tile[5]}, false);
                        auto mmRes = Matrix::Matmul<false, true>
                        (DataType::DT_FP32, curQ, curK);
                        
                        TileShape::Current().SetVecTile(tileConfig.weightTile);
                        auto curW = View(weight2D, {group, 1}, {qOffset, 0});
                        auto wB32 = Cast(curW, DT_FP32);
                        
                        auto reluRes = MaxS(mmRes, Element(DT_FP32, 0.0f));
                        auto mulRes = Mul(reluRes, wB32);
                        auto sumRes = RowSumSingle(mulRes, 0);
                        
                        Assemble(sumRes, {bs1n2Offset, firstBlockIdx * blockSize}, 
                        localSum);
                    }
                };
                // ... 展开循环调用
            }
        }
    }
}
```
关键技术点 ：

1. 因果偏移处理 ： casualOffset = s1 - s1Idx - 1 确保自回归推理时 Query 只能关注历史位置
2. 块表索引 ： curBlockIdx = GetTensorData(blockTable, {bIdx, blockIdx}) 从块表获取实际物理块位置
3. 边界处理 ： std::min(blockSize, effSeq - (blockIdx * blockSize)) 处理最后一个不完整块
4. 算子融合 ：Matmul、ReLU、Weight 乘法、行求和在单个 Tile 内完成，减少内存访问
### 3.3 分级 TopK 实现
文件位置 ： ops/pypto/src/lightning_indexer_pto/op_kernel/lightning_indexer_topk.cpp

```
ASSERT(selectedCount == 2048);  // DeepSeek V4 固定为 2048

const int length2K = selectedCount;
const int length8K = 1024 * 8;
const int length32K = 1024 * 32;

LOOP("INDEX_LOOP_TOPK_bs1n2Offset", FunctionType::DYNAMIC_LOOP, bs1n2Offset, 
LoopRange(b * s1 * n2)) {
    // ... 计算 effSeq
    
    // 分支 1: effSeq ≤ 2K
    auto lengthIsLE2K = effSeq <= length2K;
    LOOP("2K_TOPK", FunctionType::DYNAMIC_LOOP, unused, LoopRange(lengthIsLE2K)) {
        // 直接排序
        auto [res, tmp] = TopKSort(View(localSum, {1, length2K}, {1, effSeq}, 
        {bs1n2Offset, 0}), 0);
        auto resIdx = TopKExtract(res, selectedCount, true);
        // ... 结果组装
    }
    
    // 分支 2: 2K < effSeq ≤ 8K
    auto lengthIsLE8K = effSeq <= length8K;
    auto lengthIsGT2K = effSeq > length2K;
    LOOP("8K_TOPK", FunctionType::DYNAMIC_LOOP, unused, LoopRange(lengthIsGT2K * 
    lengthIsLE8K)) {
        // 直接排序（8K 可在单个 Tile 内处理）
        auto [res, tmp] = TopKSort(View(padX8K, {1, length8K}, {bs1n2Offset, 0}), 
        0);
        auto resIdx = TopKExtract(res, selectedCount, true);
        // ... 结果组装
    }
    
    // 分支 3: effSeq > 8K（两级归并）
    auto lengthIsGT8K = effSeq > length8K;
    auto numOf8K = (effSeq - 1) / length8K + 1;
    
    // 第一级: 128K → 32K（每 8K 选 top-2048）
    LOOP("128K_TO_32K_FULL_SORT", FunctionType::DYNAMIC_LOOP, idx1, LoopRange
    (numOf8KFullBlock * lengthIsGT8K)) {
        auto ax = View(effSumRes, {1, length8K}, {0, idx1 * length8K});
        auto [res, tmp] = TopKSort(ax, idx1);
        Assemble(Assign(View(res, {1, selectedCount * 2}, {0, 0})), 
            {bs1n2Offset, idx1 * selectedCount * 2}, localY1);
    }
    
    // 第二级: 32K → 8K
    LOOP("32K_TO_8K_MERGE", FunctionType::DYNAMIC_LOOP, idx2, LoopRange(numOf32K * 
    lengthIsGT8K)) {
        auto res = TopKMerge(View(localY1, {1, length8K * 2}, {bs1n2Offset, idx2 * 
        length8K * 2}), selectedCount);
        Assemble(Assign(View(res, {1, selectedCount * 2}, {0, 0})), 
            {bs1n2Offset, idx2 * selectedCount * 2}, localY2);
    }
    
    // 第三级: 8K → 2K（最终结果）
    LOOP("8K_TO_2K_MERGE", FunctionType::DYNAMIC_LOOP, unused, LoopRange(1 * 
    lengthIsGT8K)) {
        auto res = TopKMerge(View(localY2, {1, length8K * 2}, {bs1n2Offset, 0}), 
        selectedCount);
        auto resIdx = TopKExtract(res, selectedCount, true);
        // ... 结果组装
    }
}
```
技术亮点 ：

1. 条件循环 ：使用 LoopRange(condition) 实现条件分支，避免显式 if-else 带来的控制流开销
2. TopKMerge ：专用于归并已排序子序列的高效算子
3. 内存预分配 ： localY1 、 localY2 等中间结果使用最大尺寸预分配，避免动态内存分配
## 四、性能分析与优化
### 4.1 耗时影响因素
因素 影响程度 量化分析 有效序列长度 高 128K 序列耗时约为 2K 序列的 8-10 倍 Batch Size 高 线性增长（假设内存带宽充足） S1（Query 长度） 中 每个 Query 位置独立计算 Tile 配置 中 影响算子内部并行度和缓存命中率 Unroll 长度 低 减少循环开销，提升指令级并行

### 4.2 关键优化技术
优化技术 代码位置 实现效果 循环展开 unrollingProcess 模板 减少分支预测失败 L2 缓存亲和性 L2CACHE_AFFINITY_SCH 提升数据局部性 表达式融合 CODEGEN_EXPRESSION_FUSION 减少中间结果写入内存 动态形状支持 SymbolicScalar 支持变长输入，避免重复编译

## 五、总结
DeepSeek V4 的 Indexer 模块通过以下核心技术实现了长序列稀疏注意力的高效计算：

1. 块级稀疏管理 ：基于 Block Table 跳过无效区域，提升计算效率
2. 两阶段算子设计 ：Prolog + Indexer 融合，减少内存带宽占用
3. 分级 TopK 算法 ：针对不同序列长度选择最优算法路径
4. 量化加速 ：INT8 量化 Query 投影，降低内存占用
实现位置确认 ：所有核心代码均位于本代码仓的 ops/pypto/src/ 目录下，包括算子定义、核心算法实现、Tiling 策略等，是一个完整的自包含实现。
