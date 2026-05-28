# LightningIndexerTopk.cpp 深度解析
## 一、文件整体功能定位
lightning_indexer_topk.cpp 是 DeepSeek V4 稀疏注意力机制 的核心算子实现，主要完成以下任务：

阶段 功能 代码位置 阶段1：注意力分数计算 计算每个位置与有效Key的注意力分数 30-131行 阶段2：TopK选择 从分数中选择Top-2048个索引 133-314行

核心设计思想 ：通过 块级稀疏处理 和 自适应算法选择 ，在长序列场景下实现高效的稀疏注意力计算。

## 二、前131行详细解析
### 2.1 头文件和命名空间（1-22行）
```
#include <cfloat>
#include "tilefwk/tensor.h"
#include "tilefwk/tilefwk.h"
#include "lightning_indexer_topk.h"
#include "parallel_sort.h"

using namespace npu::tile_fwk;
```
关键点 ：

- tilefwk ：NPU Tile 框架，提供张量操作和 Tile 配置功能
- parallel_sort.h ：并行排序算法头文件
- npu::tile_fwk ：NPU Tile 框架命名空间
### 2.2 函数入口（23-28行）
```
void LightningIndexerTopk(const Tensor &query, const Tensor &key, const Tensor &
weights, 
    const Tensor &actSeqKey, const Tensor &blockTable, Tensor &topkRes, 
    const int selectedCount, IndexerTile tileConfig, std::set<int> unrollList) {
    LightningIndexerTopkImpl(query, key, weights, actSeqKey, blockTable, 
                            topkRes, selectedCount, tileConfig, unrollList);
}
```
设计模式 ：采用 包装函数 模式，将实际实现委托给 LightningIndexerTopkImpl 。

### 2.3 主函数签名和参数说明（30-41行）
```
void LightningIndexerTopkImpl(
    const Tensor &query,           // [B, S1, indexN1, indexD] - 查询向量
    const Tensor &key,             // [blockNum, blockSize, n2, indexD] - 键向量（分
    块存储）
    const Tensor &weights,         // [B, S1, indexN1] - 权重向量
    const Tensor &actSeqKey,       // [B] - 实际序列长度
    const Tensor &blockTable,      // [B, maxBlockNum] - 块映射表（支持稀疏访问）
    Tensor &topkRes,               // [B, s1, N2, selectedCount] - 输出TopK索引
    const int selectedCount,       // 固定为2048
    IndexerTile tileConfig,        // Tile配置（影响并行效率）
    std::set<int> unrollList,      // 循环展开配置
    Tensor *tmpOut,                // 可选：临时输出
    Tensor *topkValue              // 可选：TopK值（非索引）
)
```
核心参数解读 ：

参数 维度 说明 query [B, S1, 64, 128] 通常 indexN1=64（索引头数），indexD=128（头维度） key [blockNum, 128, 1, 128] blockSize固定为128 blockTable [B, maxBlockNum] 关键 ：支持变长序列的稀疏块访问

### 2.4 符号化处理（43-58行）
```
// Symbolization
SymbolicScalar b = GetInputShape(query, 0);      // Batch大小（动态）
SymbolicScalar s1 = GetInputShape(query, 1);     // 序列长度（动态）
SymbolicScalar blockNum = GetInputShape(key, 0); // 块数量（动态）

auto indexN1 = query.GetShape()[SHAPE_DIM2];  // 索引头数（固定）
auto indexD = query.GetShape()[SHAPE_DIM3];   // 头维度（固定）
auto blockSize = key.GetShape()[1];           // 块大小（固定为128）
auto n2 = key.GetShape()[SHAPE_DIM2];         // 输出头数（固定为1）
auto group = indexN1 / n2;                    // 分组数 = 64 / 1 = 64

// 常量定义
constexpr int64_t maxBatch = 128;
constexpr int64_t maxS1 = 4;
constexpr int64_t maxN2 = 1;
constexpr int64_t maxS2 = 128 * 1024;  // 最大支持128K序列
```
设计要点 ：

- 动态维度 ： b 、 s1 、 blockNum 用 SymbolicScalar 表示，支持变长输入
- 固定维度 ： indexN1 、 indexD 、 blockSize 是编译期固定值
- maxS2=128K ：预分配最大128K空间，避免动态内存分配
### 2.5 张量初始化（60-70行）
```
Tensor query2D(dtype, {b * s1 * indexN1, indexD}, "query2D");  // [B×S1×64, 128]
Tensor key2D(dtype, {blockNum * blockSize, n2 * indexD}, "key2D");  // 
[blockNum×128, 128]
Tensor weight2D(dtype, {b * s1 * indexN1, 1}, "weight2D");  // [B×S1×64, 1]
Tensor localSum(DT_FP32, {maxBatch * maxS1 * maxN2, maxS2}, "localSum");  // [512, 
131072]

// 4D → 2D 重排
LOOP("INPUT_4D_2_2D", FunctionType::DYNAMIC_LOOP, unUsedIdx, LoopRange(1)) {
    ReshapeInplace(query, query2D);
    ReshapeInplace(key, key2D);
    ReshapeInplace(weights, weight2D);
}
```
核心设计 ：

- 4D→2D转换 ：将高维张量展平，便于后续块级处理
- localSum预分配 ：这是 最关键的设计
  - 大小：[128×4×1, 131072] = [512, 131072]
  - 用途：存储所有位置的注意力分数
  - 优势：一次性分配，避免运行时动态分配
### 2.6 三重循环结构（72-131行）
这是前131行的 核心部分 ，实现了块级注意力分数计算：

```
LOOP("INDEX_LOOP_BATCH", ..., bIdx, LoopRange(b)) {        // 批处理循环
    LOOP("INDEX_LOOP_S1", ..., s1Idx, LoopRange(s1)) {    // 序列位置循环
        LOOP("INDEX_LOOP_N2", ..., n2Idx, LoopRange(n2)) {  // 输出头循环
            // 核心计算逻辑
        }
    }
}
``` （1）因果约束和有效序列长度（75-79行）
```
auto casualOffset = s1 - s1Idx - 1;  // 因果偏移
auto effSeq = curSeq - casualOffset; // 有效序列长度
auto actBlock = (effSeq + blockSize - 1) / blockSize;  // 有效块数
```
数学推导 ：

```
对于Prefill阶段（curSeq == s1）：
effSeq = s1 - (s1 - s1Idx - 1) = s1Idx + 1

这意味着：
- 位置0：effSeq=1（只能看到自己）
- 位置1：effSeq=2（能看到位置0和1）
- ...
- 位置s1-1：effSeq=s1（能看到全部）
```
因果约束的可视化 ：

```
位置:     0     1     2     3     4     5     6     7
effSeq:   1     2     3     4     5     6     7     8
          │     │     │     │     │     │     │     │
          ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
能看到:  [0]  [0,1] [0,2] [0,3] [0,4] [0,5] [0,6] [0,7]
``` （2）Unrolling处理模板（84-120行）
```
auto unrollingProcess = [&](int unrollLength, auto &&firstBlockIdx) {
    // 1. 获取当前Query切片
    auto curQ = View(query2D, {group, indexD}, {qOffset, 0});  // (64, 128)

    std::vector<Tensor> concatSrcs;
    
    // 2. 静态展开处理多个块
    for (int subblockIdx = 0; subblockIdx < unrollLength; subblockIdx++) {
        auto blockIdx = firstBlockIdx + subblockIdx;
        
        // 关键：从blockTable获取实际块位置（支持稀疏）
        SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, blockIdx});
        
        // 获取当前Key块（处理边界）
        auto curK = View(key2D, {blockSize, indexD},
            {std::min(blockSize, effSeq - (blockIdx * blockSize)), indexD},
            {curBlockIdx * blockSize, n2Idx * indexD});  // (128, 128)

        // 3. Matmul计算（核心）
        TileShape::Current().SetCubeTile({c1Tile[0], c1Tile[1]}, 
                                          {c1Tile[2], c1Tile[3]}, 
                                          {c1Tile[4], c1Tile[5]}, false);
        auto mmRes = Matrix::Matmul<false, true>(DataType::DT_FP32, curQ, curK);
        concatSrcs.emplace_back(mmRes);
    }

    // 4. ReLU + Weight融合
    auto curW = View(weight2D, {group, 1}, {qOffset, 0});  // (64, 1)
    auto wB32 = Cast(curW, DT_FP32);
    auto mmRes = Concat(concatSrcs, -1);
    auto reluRes = MaxS(mmRes, Element(DT_FP32, 0.0f));  // ReLU
    auto mulRes = Mul(reluRes, wB32);                     // ×权重

    // 5. 行求和并聚合到localSum
    auto sumRes = RowSumSingle(mulRes, 0);  // (1, superBlockSize)
    Assemble(sumRes, {bs1n2Offset, firstBlockIdx * blockSize}, localSum);
};
```
核心计算流程 ：

```
Query [64,128] × Key^T [128,128] → Score [64,128]
        ↓
    ReLU(Score) → [64,128]
        ↓
    Score × Weight [64,1] → [64,128]
        ↓
    RowSum → [1,128]（聚合到单向量）
        ↓
    写入localSum
```
关键技术点 ：

技术 实现 收益 块稀疏访问 blockTable 动态获取块位置 支持变长序列，跳过padding 算子融合 Matmul + ReLU + Weight + RowSum 在单个Tile内完成 减少内存访问，提升效率 边界处理 std::min(blockSize, effSeq - ...) 正确处理不完整块 循环展开 unrollLength 参数化展开 减少循环开销，提升并行度
 （3）Matmul循环（122-128行）
```
LOOP("INDEX_LOOP_MATMUL", ..., blockIdx, LoopRange(actBlock), unrollList) {
    for (int unrollLength : unrollList) {
        UNROLL(unrollLength) {
            unrollingProcess(unrollLength, blockIdx);
        }
    }
}
```
设计要点 ：

- actBlock ：有效块数，随位置递增
- unrollList ：支持多种展开长度（如{1, 2, 4}）
- UNROLL ：NPU特有的循环展开指令，提升指令级并行
## 三、前131行关键点总结
### 3.1 核心数据流
```
输入张量 → 2D重排 → 块级Matmul → ReLU → Weight融合 → RowSum → localSum
     │                                                           │
     └───────────────────────────────────────────────────────────┘
                          三重循环驱动
```
### 3.2 关键设计决策
决策 实现方式 技术价值 动态形状支持 SymbolicScalar 支持变长输入，避免重复编译 预分配内存 localSum 固定128K大小 避免动态内存分配，提升性能稳定性 因果约束 casualOffset 计算 确保自回归生成的正确性 块级处理 blockSize=128 提升数据局部性，利用缓存 循环展开 unrollingProcess 模板 减少循环开销，提升并行度 算子融合 四操作在单个Tile内完成 减少内存带宽压力

### 3.3 计算量分析
以 group=64 、 blockSize=128 、 indexD=128 为例：

操作 输入 输出 计算量 Matmul (64,128) × (128,128) (64,128) 64×128×128×2 ≈ 2.1M FLOPs ReLU (64,128) (64,128) 8K操作 Mul (64,128) × (64,1) (64,128) 8K操作 RowSum (64,128) (1,128) 8K操作

单块总计算量 ：约 2.1M FLOPs

## 四、与后段代码的衔接
前131行计算得到的 localSum 张量，在后段代码中被用于TopK选择：

```
// 133行之后的TopK选择部分
ASSERT(selectedCount == 2048);  // 固定选择2048个

// 根据effSeq选择算法路径：
// - effSeq ≤ 2K：直接TopKSort
// - 2K < effSeq ≤ 8K：直接TopKSort  
// - effSeq > 8K：两级归并
```
数据流完整路径 ：

```
前131行：计算注意力分数 → localSum[B×S1×N2, 128K]
    ↓
后183行：TopK选择 → topkRes[B, S1, N2, 2048]
```
## 五、总结
### 5.1 前131行核心功能
1. 动态形状处理 ：支持变长Batch和序列长度
2. 因果约束实现 ：确保自回归生成的正确性
3. 块级稀疏计算 ：基于blockTable实现稀疏块访问
4. 算子融合优化 ：Matmul+ReLU+Weight+RowSum融合
5. 循环展开 ：提升指令级并行度
### 5.2 设计亮点
亮点 说明 自适应计算 有效序列长度随位置递增，计算量动态调整 内存效率 预分配固定大小缓冲区，避免动态分配 硬件友好 Tile配置和循环展开针对NPU架构优化 稀疏支持 blockTable机制支持高效的稀疏注意力

### 5.3 后续优化方向
1. 动态Tile配置 ：根据实际输入形状调整Tile大小
2. 混合精度计算 ：在保证精度的前提下使用更低精度
3. 流水线优化 ：将计算和内存访问重叠
