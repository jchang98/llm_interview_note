# 02.大语言模型架构

## Fast Look

**阅读链路**：`BPE/WordPiece/Unigram → token embedding + RoPE/ALiBi → MHA/GQA + RMSNorm + SwiGLU → KV Cache/FlashAttention → Temperature/Top-k/Top-p → Llama、ChatGLM、MoE`。以下以具体技术实体为最小查询单位。

### 输入与位置信息

#### BPE（Byte-Pair Encoding）

- **描述**：从字符或字节开始，反复合并语料中最频繁的相邻符号对，得到子词词表。
- **与之前方法的区别**：相对词级分词不会因新词直接 OOV；相对字级分词可用更短序列表示高频片段。
- **优点**：算法直观、训练快速、词表大小可控，是 GPT 系模型的常见基础。
- **缺点**：按频率贪心合并，不直接优化语言模型概率；中文、代码和罕见词可能产生较碎的 token。

#### WordPiece

- **描述**：也是子词词表，但以提升训练语料的语言模型似然为准则选择合并，而非只看 pair 频率。
- **与 BPE 的区别**：BPE 选最高频 pair；WordPiece 选带来更大概率增益的候选，典型用于 BERT。
- **优点**：子词切分更贴近语言模型目标，开放词表能力强。
- **缺点**：训练实现更复杂；不同实现的 greedy 切分和特殊 token 约定不能混用。

#### Unigram 与 SentencePiece

- **Unigram 描述**：先建立大词表，再逐步删除对整体似然影响最小的子词，保留概率模型最优的词表。
- **SentencePiece 描述**：一种语言无关的训练/编码框架，可实现 BPE 或 Unigram，并把空格作为普通符号处理。
- **与前两者的区别**：Unigram 是“删减式”词表优化；SentencePiece 不依赖按空格预分词，适合中文、多语言和无空格语言。
- **优点**：支持子词采样和多语言；原始文本可逆性较好。
- **缺点**：token 长度和分词边界仍会影响上下文长度、训练成本和下游效果。

#### Absolute Position Embedding（绝对位置编码）

- **描述**：为序列中的第 `i` 个位置提供一个位置向量，与 token embedding 相加；可为固定正弦/余弦或可学习表。
- **与之前方法的区别**：解决 Attention 的置换不变性，使模型知道 token 的先后次序。
- **优点**：实现简单，固定正弦编码可无需额外训练参数。
- **缺点**：主要表达“位置编号”而非相对距离；可学习位置表通常难直接外推到训练长度之外。

#### Relative Position Encoding（相对位置编码）

- **描述**：将 `i-j` 的相对距离作为 attention score 或 value 的附加信息。
- **与绝对位置编码的区别**：关注 token 之间相隔多远，而非各自处于第几个位置。
- **优点**：更符合语言局部关系和长度平移不变性，对长文本更友好。
- **缺点**：实现与显存开销更复杂；距离桶、截断范围等设计会影响效果。

#### RoPE（Rotary Position Embedding）

- **描述**：将 Q、K 的二维通道按位置角度旋转，使它们的点积自然包含相对位置信息。
- **与前法的区别**：不同于把位置向量直接加到 embedding，RoPE 在注意力的 Q/K 空间注入位置信息；不同于通用相对 bias，它具有旋转形式的相对距离性质。
- **优点**：无额外位置表、计算高效，是 Llama 系列常用方案；较适合长度扩展技巧。
- **缺点**：超出训练长度仍会失真；NTK/插值/YaRN 等外推方案需重新验证，不能只改 `max_position`。

#### ALiBi（Attention with Linear Biases）

- **描述**：为每个注意力头的 score 加上随距离线性衰减的负 bias。
- **与 RoPE 的区别**：RoPE 改变 Q/K 表示；ALiBi 直接改 attention logits，且不需要位置 embedding 表。
- **优点**：实现极简，长度外推通常较自然，训练显存开销小。
- **缺点**：位置表达能力与 head slope 是固定结构假设；在某些任务和已有 RoPE checkpoint 上不能无代价替换。

### Transformer 层内实体

#### Scaled Dot-Product Attention

- **描述**：`softmax(QKᵀ / √dₖ)V`；`1/√dₖ` 控制高维点积尺度，mask 负责屏蔽 padding 或未来 token。
- **与普通点积 Attention 的区别**：增加缩放以避免 softmax 饱和和梯度过小。
- **优点**：数值稳定、易矩阵并行，是所有 MHA 类机制的基础。
- **缺点**：完整 score 矩阵的计算与显存为 `O(n²)`；mask 错误会造成信息泄漏或无效 token 干扰。

#### MHA（Multi-Head Attention）

- **描述**：使用 `h` 组独立 Q/K/V 投影并行计算注意力，再拼接并投影输出。
- **与单头 Attention 的区别**：每个头可在不同表示子空间关注不同依赖、距离或模式。
- **优点**：表达能力最强，是原始 Transformer 的标准注意力形式。
- **缺点**：Decode 阶段每个头均保存 K/V，KV Cache 占用和带宽需求最高。

#### MQA（Multi-Query Attention）

- **描述**：保留多个 Query 头，但所有 Query 头共享同一组 K/V。
- **与 MHA 的区别**：将 K/V 头数从 `h` 降到 `1`，只保留多 Query。
- **优点**：KV Cache 约缩小 `h` 倍，显著改善自回归推理内存带宽瓶颈。
- **缺点**：K/V 表示被过度共享，复杂任务的质量可能低于 MHA。

#### GQA（Grouped-Query Attention）

- **描述**：将多个 Query 头分组，每组共享一组 K/V；K/V 头数介于 MHA 和 MQA 之间。
- **与 MHA/MQA 的区别**：MHA 不共享 K/V，MQA 全部共享，GQA 进行组内共享，是两者折中。
- **优点**：保留多组 K/V 的表达能力，同时获得大部分 KV Cache 节省；Llama 2/3 等使用该方案。
- **缺点**：组数/组大小是质量—成本超参数；MHA 权重转 GQA 往往需要 uptraining，而不是直接复制。

#### KV Cache

- **描述**：在 Prefill 时缓存每层历史 token 的 K、V；Decode 时新 token 只计算自身 Q/K/V，并读取历史 K/V。
- **与不缓存的区别**：避免每生成一个 token 都重复计算整个历史前缀。
- **优点**：把自回归生成的重复计算显著降低，是在线 LLM 推理的必要优化。
- **缺点**：上下文、层数、隐藏维度和并发都线性占用显存；需要与 GQA、分页缓存和调度共同设计。

#### FlashAttention

- **描述**：用 tiling 和 online softmax 在 GPU SRAM 中分块计算精确 Attention，避免物化完整 `n×n` 注意力矩阵。
- **与普通 Attention 实现的区别**：不改变数学结果，只优化 IO 和中间激活存储；与 MQA/GQA 这种改变 K/V 头数的架构优化不同。
- **优点**：显著降低 HBM 读写和显存，提高长序列训练及 Prefill 吞吐。
- **缺点**：总 FLOPs 仍是 `O(n²)`；收益依赖 GPU、内核版本、序列长度和算子支持。

#### LayerNorm、RMSNorm 与 Pre-LN/Post-LN

- **LayerNorm 描述**：对单个 token 的 hidden dimension 做均值—方差归一化并施加可学习缩放/偏移。
- **RMSNorm 描述**：只按均方根缩放，不减均值、通常也不使用 bias；Llama 常用。
- **Pre-LN/Post-LN 区别**：Pre-LN 在 Attention/FFN 前归一化，Post-LN 在残差相加后归一化。
- **与 BatchNorm 的区别**：BN 依赖 batch 统计量，LN/RMSNorm 不依赖 batch，更适合变长序列和小 batch 推理。
- **优点**：Pre-LN 更利于深层梯度传播；RMSNorm 更省计算，常能保持相近效果。
- **缺点**：归一化位置改变训练动力学；RMSNorm 省去的是中心化而不是“必然更准确”，需随架构整体选择。

#### GeLU、Swish/SiLU、GLU、SwiGLU

- **GeLU 描述**：按输入大小平滑地保留或抑制信号，BERT 常用。
- **Swish/SiLU 描述**：`x·sigmoid(x)`，平滑且允许小负值通过。
- **GLU/SwiGLU 描述**：用一条门分支调制另一条值分支；SwiGLU 以 Swish/SiLU 作为门函数。
- **与 ReLU/普通 FFN 的区别**：ReLU 硬截断负值，普通 FFN 无显式门控；SwiGLU 增加输入依赖的选择性。
- **优点**：SwiGLU 在现代 Decoder-only LLM 中常以相近 FLOPs 得到更好的表达能力。
- **缺点**：需要额外投影，参数/FLOPs 比较须保持中间维度或计算量一致。

### 生成与模型实体

#### Temperature、Top-k、Top-p（Nucleus Sampling）

- **Temperature 描述**：缩放 logits；低温更确定，高温更随机。
- **Top-k 描述**：只在概率最高的 `k` 个 token 中重新归一化采样。
- **Top-p 描述**：取累计概率达到 `p` 的最小候选集合，候选个数会随分布熵变化。
- **与 Greedy 的区别**：Greedy 始终选最大概率；三者均引入或控制随机性。Top-p 相对 Top-k 是自适应候选集。
- **优点**：可按任务调节确定性、多样性和长尾噪声；三者可联合使用。
- **缺点**：无通用最优参数；温度过高易幻觉，阈值过严会重复和模板化。

#### BERT、MLM、NSP 与三类 Embedding

- **BERT 描述**：基于 Encoder 的双向表示模型；MLM 遮盖部分 token 预测原词，NSP 判断句对关系；输入为 token、segment、position embedding 之和。
- **与 GPT/Decoder-only 的区别**：BERT 可看左右上下文但不能自然自回归生成；GPT 使用因果 mask，适合续写和对话生成。
- **优点**：分类、匹配、检索、抽取等文本理解任务强，`[CLS]` 表示可统一用于句级任务。
- **缺点**：`[MASK]` 造成预训练—微调不一致，NSP 效益有限，长上下文和生成能力受限。

#### RoBERTa、ALBERT、SpanBERT、XLNet

- **RoBERTa**：去 NSP、动态 masking、扩大数据和 batch；相对 BERT 改善训练配方，通常效果更好；代价是训练资源更高。
- **ALBERT**：Embedding 因式分解、跨层参数共享、以 SOP 替代 NSP；相对 BERT 显著降参数/显存；共享可能限制层间差异。
- **SpanBERT**：连续 span mask + span boundary objective；相对 token mask 更适合短语/抽取任务；仍属于 Encoder MLM 范式。
- **XLNet**：排列语言模型 + two-stream attention；相对 MLM 避免 `[MASK]` 不一致并获得双向上下文；目标与实现更复杂、训练成本高。

#### Llama、RMSNorm、RoPE、SwiGLU 与 GQA

- **描述**：Llama 是 Decoder-only 基座模型；其典型组件为 RMSNorm、RoPE、SwiGLU，Llama 2 起在部分规模中采用 GQA；Llama 2 代码文档还覆盖 tokenizer、embedding、KV Cache、FFN 和自回归输出的完整数据流。
- **与 GPT-2/BERT 的区别**：相对 BERT 选择纯因果生成；相对早期 GPT 风格组合，采用更现代的 Norm、位置、门控和 KV 共享设计。
- **优点**：架构简洁、开源生态成熟，在训练稳定性、长上下文和推理成本上较均衡。
- **缺点**：仍受自回归逐 token 时延限制；“Llama 架构”不等于所有版本配置完全相同。

#### Alpaca、Code Llama 与 Llama 3

- **Alpaca**：用 Self-Instruct 合成数据对 Llama 做 SFT；相对基座模型获得更强指令遵循，代价是继承教师数据偏差。
- **Code Llama**：在代码上继续训练并支持 infilling；相对通用 Llama 更适合补全/填空，代价是通用能力不必然同步提升。
- **Llama 3**：在 Llama 主干上强化 tokenizer、数据工程、训练规模和指令微调；相对 Llama 2 的提升主要是系统训练配方，不应只归因于单个架构组件。

#### ChatGLM、ChatGLM 2 与 ChatGLM 3

- **描述**：ChatGLM 基于 GLM 的自回归空白填充和多目标预训练；后续版本增强上下文、效率、对话和工具调用能力。
- **与 Llama/GPT 的区别**：GLM 目标通过 blank infilling 融合双向理解和自回归生成；Llama/GPT 则主要是纯因果语言建模。
- **优点**：中文/双语对话与开源部署生态较完善，适合比较不同预训练目标。
- **缺点**：不同版本的 chat template、上下文长度和工具接口差异较大，不能把版本号当作单一能力指标。

#### Qwen 系列开源权重：Qwen → 1.5 → 2 → 2.5 → 3 → 3.5/3.6

- **描述**：Qwen 从中英双语 Decoder-only Base/Chat 起步；1.5 强化 32K 与生态兼容并引入早期 MoE，2 采用 GQA/RoPE/RMSNorm/SwiGLU 并进入 128K/MoE，2.5 扩展 18T 数据及 Coder/Math/VL/1M 分支，3 用 36T、dense+MoE 与四阶段 hybrid-thinking 后训练统一升级。Qwen3.5 已开源 4B、9B、`35B-A3B` 等权重，Qwen3.6 公开 `35B-A3B`。
- **与之前方法的区别**：相对只按参数放大的语言模型路线，Qwen 的演进将 KV 共享、条件计算、长上下文、专项模型、测试时 thinking budget 与 Agent 协议逐步组合；3.5 进一步转向视觉—语言和混合线性/全注意力架构。
- **优点**：覆盖本地小模型、大 MoE、代码/数学/视觉与百万上下文，提供较完整的开源—API—Agent 工具生态。完整时间线见 [Qwen系列模型](./Qwen系列模型/Qwen系列模型.md)。
- **缺点/注意点**：不同分支（Text/Coder/Math/VL/Omni）不可直接互比；3.5/3.6 已有模型卡但缺完整独立技术报告，不能从版本名补造参数、注意力或训练算法。

### MoE 实体

#### MoE（Mixture of Experts）、Router 与 Top-k Gating

- **描述**：以多个 Expert FFN 替代一个稠密 FFN；Router/Gating Network 为每个 token 选择 Top-k 专家，Softmax 或 Noise Top-k Gating 产生路由分数。
- **与稠密 FFN 的区别**：稠密 FFN 对每个 token 激活全部参数；MoE 只激活少数专家，是条件计算。
- **优点**：在近似每 token FLOPs 下大幅扩大总参数容量，专家可专门化。
- **缺点**：会出现专家热点、token overflow、all-to-all 通信和小 batch shrinking 问题；总参数量不能代表单 token 计算量。

#### Load Balancing Loss、Router z-loss 与 Capacity Factor

- **Load Balancing Loss 描述**：约束路由概率和实际 token 分配更均匀，防止少数专家垄断。
- **Router z-loss 描述**：惩罚 router logits 的过大尺度，改善路由训练稳定性。
- **Capacity Factor 描述**：规定每个专家最多接收多少 token，超过容量的 token 可能丢弃或改路由。
- **与普通任务损失的区别**：它们是为了让条件计算可训练、可并行的辅助约束，不直接表示语言建模质量。
- **优点**：提高专家利用率和训练稳定性，减少通信/负载失衡。
- **缺点**：辅助损失权重和容量设置影响质量、溢出率与吞吐，需要联合监控。

#### Switch Transformer

- **描述**：MoE 的 Top-1 路由实现：每个 token 只选一个专家，并配合负载均衡损失、选择性精度、小初始化和正则化。
- **与 Top-2 MoE 的区别**：将每 token 专家数从多个降为一个，降低路由、计算和通信复杂度。
- **优点**：稀疏训练/推理更简单，适合极大参数规模的高吞吐条件计算。
- **缺点**：更依赖路由均衡与容量管理；单专家选择可能牺牲冗余和质量。

#### GShard、GLaM、ST-MoE 与 MoEBERT

- **GShard**：将条件计算与自动分片结合；**优点**是使巨型 MoE 更容易分布式扩展，**代价**是并行系统复杂。
- **GLaM**：以 MoE 做高效语言建模；**区别**是强调用较少激活计算获得大容量，**代价**是通信与路由开销。
- **ST-MoE**：使用 router z-loss 等稳定技术；**优点**是改善训练与微调稳定性，**代价**是仍需精细调参。
- **MoEBERT**：把专家机制用于 BERT 的重要性引导适配；**区别**是从“扩容”转向“轻量化/适配”，**代价**是模型和任务迁移假设更强。

### 快速自测

- [大模型架构高频面试题](./大模型架构_高频面试题.md)：回答任一实体时，按“**定义 → 相对前法改变了什么 → 节省/增强了什么 → 带来什么代价 → 训练还是推理阶段生效**”组织答案。

### 2.1 Transformer模型

[1.attention](/02.大语言模型架构/1.attention/1.attention.md "1.attention")

[2.layer\_normalization](/02.大语言模型架构/2.layer_normalization/2.layer_normalization.md "2.layer_normalization")

[3.位置编码](/02.大语言模型架构/3.位置编码/3.位置编码.md "3.位置编码")

[4.tokenize分词](/02.大语言模型架构/4.tokenize分词/4.tokenize分词.md "4.tokenize分词")

[5.token及模型参数](/02.大语言模型架构/5.token及模型参数/5.token及模型参数.md "5.token及模型参数")

[6.激活函数](/02.大语言模型架构/6.激活函数/6.激活函数.md "6.激活函数")

### 2.2 注意力

[MHA\_MQA\_GQA](/02.大语言模型架构/MHA_MQA_GQA/MHA_MQA_GQA.md "MHA_MQA_GQA")

### 2.3 解码部分

[解码策略（Top-k & Top-p & Temperature）](</02.大语言模型架构/解码策略（Top-k & Top-p & Temperatu/解码策略（Top-k & Top-p & Temperature）.md> "解码策略（Top-k & Top-p & Temperature）")

### 2.4 BERT

[bert细节](/02.大语言模型架构/bert细节/bert细节.md "bert细节")

[Transformer架构细节](/02.大语言模型架构/Transformer架构细节/Transformer架构细节.md "Transformer架构细节")

[bert变种](/02.大语言模型架构/bert变种/bert变种.md "bert变种")

### 2.5 常见大模型

[主流 LLM 模型结构速通（Llama、Qwen、GLM、DeepSeek）](</02.大语言模型架构/主流LLM模型结构速通/主流LLM模型结构速通.md> "主流 LLM 模型结构速通")

[llama系列模型](/02.大语言模型架构/llama系列模型/llama系列模型.md "llama系列模型")

[chatglm系列模型](/02.大语言模型架构/chatglm系列模型/chatglm系列模型.md "chatglm系列模型")

[llama 2代码详解](</02.大语言模型架构/llama 2代码详解/llama 2代码详解.md> "llama 2代码详解")

[llama 3](</02.大语言模型架构/llama 3/llama 3.md> "llama 3")

### 2.6 MoE

[1.MoE论文](/02.大语言模型架构/1.MoE论文/1.MoE论文.md "1.MoE论文")

[2.MoE经典论文简牍](/02.大语言模型架构/2.MoE经典论文简牍/2.MoE经典论文简牍.md "2.MoE经典论文简牍")

[3.LLM MoE ：Switch Transformers](</02.大语言模型架构/3.LLM MoE ：Switch Transformers/3.LLM MoE ：Switch Transformers.md> "3.LLM MoE ：Switch Transformers")
