# 04.分布式训练

## Fast Look

**选型链路**：先估算 `参数 + 梯度 + 优化器状态 + 激活`，再按瓶颈选择 `DP/FSDP → TP → PP → SP → EP`，最后用混合并行和 ZeRO/Offload 落地；所有方案都受通信拓扑约束。

#### DP 与 DDP

- **DP 描述**：单进程多卡复制模型、集中 gather 输出；**DDP 描述**：一张卡一个进程，反向传播时 all-reduce 梯度。
- **与单卡训练的区别**：模型复制到各卡，每卡处理不同 batch 分片。
- **优点**：DDP 简单、扩展成熟、计算通信可部分重叠。
- **缺点/注意点**：参数、梯度和优化器状态仍完整复制；全局 batch 增大需要调整学习率与数据量。

#### FSDP 与 ZeRO Stage 1/2/3

- **描述**：Stage 1 分片优化器状态，Stage 2 再分片梯度，Stage 3 再分片参数；FSDP 是 PyTorch 的全分片实现，按需 all-gather 参数、reduce-scatter 梯度。
- **与 DDP 的区别**：DDP 复制模型状态；ZeRO/FSDP 在数据并行组内切分状态。
- **优点**：显著降低每卡模型状态显存，适合超大模型微调/训练。
- **缺点/注意点**：通信和参数 gather 频繁；层粒度、prefetch、CPU/NVMe offload 配置不当会降低吞吐。

#### Pipeline Parallelism（PP）、Micro-batch 与 1F1B

- **描述**：按层切分模型到多个 stage；micro-batch 将 batch 切块流水执行；1F1B 在稳态交替一次 forward 和一次 backward。GPipe 使用填充—排空，PipeDream/交错 1F1B 改善气泡。
- **与 DP/TP 的区别**：PP 按深度切模型，不复制完整层也不切单层矩阵。
- **优点**：能训练无法放入单卡的一串深层网络，适合跨节点切分。
- **缺点/注意点**：存在 pipeline bubble，micro-batch 增多会提高调度与激活存储开销；异步权重版本会引入一致性问题。

#### Tensor Parallelism（TP）

- **描述**：在单层内按行/列切分线性层、Attention 投影和 FFN，中间以 all-reduce/all-gather 合并；Megatron-LM 是典型实现。
- **与 PP 的区别**：PP 切不同层，TP 同时在多卡上计算同一层的不同矩阵块。
- **优点**：每层参数和计算分摊，适合单节点高速 NVLink/NVSwitch 环境。
- **缺点/注意点**：通信发生在几乎每层，跨慢网络扩展差；分片方式需匹配行/列并行的通信模式。

#### Sequence Parallelism（SP）

- **描述**：沿序列维度切分激活和部分非矩阵计算，常与 TP 配合以降低长序列激活显存。
- **与 TP 的区别**：TP 主要切 hidden/权重维度，SP 切 token/sequence 维度。
- **优点**：长上下文训练时有效降低激活内存，可补足 TP 的重复激活问题。
- **缺点/注意点**：需要额外 all-gather/reduce-scatter；并非所有 Attention 实现和操作都可无缝切分。

#### Expert Parallelism（EP）/MoE 并行

- **描述**：将不同 MoE 专家分布在不同卡，router 后通过 all-to-all 把 token 发送到对应专家。
- **与稠密 TP/PP 的区别**：切分对象是条件激活的专家网络，而非固定层或矩阵。
- **优点**：在稀疏激活下扩展总参数容量，专家可跨设备放置。
- **缺点/注意点**：all-to-all 对网络敏感；负载不均、容量溢出和小 batch 会造成低利用率。

#### Hybrid Parallelism 与自动并行

- **描述**：用 DP/FSDP、TP、PP、SP、EP 的多维设备网格组合；Mesh TensorFlow、GSPMD、FlexFlow、Alpa 等从计算图和成本模型自动搜索切分/调度。
- **与单一并行的区别**：不同维度分别解决状态、层、矩阵、序列和专家瓶颈。
- **优点**：能适配万卡级模型与不同内存/网络层级，自动并行减少人工搜索。
- **缺点/注意点**：配置空间巨大；自动方案依赖成本模型和编译器，实际性能仍需 profile 验证。

#### 混合精度、TF32、BF16 与 FP16

- **FP16 描述**：显存/带宽低但动态范围小，常需 loss scaling；**BF16** 动态范围接近 FP32，精度位数较少；**TF32** 是 NVIDIA Tensor Core 的 FP32 输入加速格式。
- **与 FP32 的区别**：以较低存储和更高 Tensor Core 吞吐交换部分数值精度。
- **优点**：显著提升训练速度并降低显存，BF16 通常更稳。
- **缺点/注意点**：精度格式应与硬件和算子支持匹配；数值不稳、溢出或收敛变化必须监控。

### 4.1 基础知识

[1.概述](/04.分布式训练/1.概述/1.概述.md "1.概述")

[2.数据并行](/04.分布式训练/2.数据并行/2.数据并行.md "2.数据并行")

[3.流水线并行](/04.分布式训练/3.流水线并行/3.流水线并行.md "3.流水线并行")

[4.张量并行](/04.分布式训练/4.张量并行/4.张量并行.md "4.张量并行")

[5.序列并行](/04.分布式训练/5.序列并行/5.序列并行.md "5.序列并行")

[6.多维度混合并行](/04.分布式训练/6.多维度混合并行/6.多维度混合并行.md "6.多维度混合并行")

[7.自动并行](/04.分布式训练/7.自动并行/7.自动并行.md "7.自动并行")

[8.moe并行](/04.分布式训练/8.moe并行/8.moe并行.md "8.moe并行")

[9.总结](/04.分布式训练/9.总结/9.总结.md "9.总结")

### 4.2 DeepSpeed

[deepspeed介绍](/04.分布式训练/deepspeed介绍/deepspeed介绍.md "deepspeed介绍")

### 4.3 Megatron

### 4.4 训练加速

### 4.5 一些有用的文章


### 4.6 一些题目

[1.分布式训练题目](/04.分布式训练/分布式训练题目/分布式训练题目.md "分布式训练题目")
[2.显存问题](/04.分布式训练/1.显存问题/1.显存问题.md "1.显存问题")

### 4.7 参考资料：

-   [大模型分布式训练并行技术（九）-总结 - 掘金 (juejin.cn)](https://juejin.cn/post/7290740395913969705 "大模型分布式训练并行技术（九）-总结 - 掘金 (juejin.cn)")
-   [https://www.zhangzhenhu.com/deepspeed/index.html](https://www.zhangzhenhu.com/deepspeed/index.html "https://www.zhangzhenhu.com/deepspeed/index.html")
-   [https://blog.csdn.net/zwqjoy/article/details/130732601](https://blog.csdn.net/zwqjoy/article/details/130732601 "https://blog.csdn.net/zwqjoy/article/details/130732601")
-   [https://techdiylife.github.io/](https://techdiylife.github.io/ "https://techdiylife.github.io/")
