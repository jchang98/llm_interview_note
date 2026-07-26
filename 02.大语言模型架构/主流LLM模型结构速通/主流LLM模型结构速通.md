# 主流 LLM 模型结构速通：从 Llama 基线到 GLM / Qwen / DeepSeek

> 来源：知乎专栏 [《1.5万字速通LLM主流模型结构（Llama、Qwen、GLM、Deepseek...）》](https://zhuanlan.zhihu.com/p/2060741715095560795)，魔法学院的 Chilia，2026-07-25。本文基于作者文章整理，并与本仓库已有页面交叉补充；重点是建立可用于面试的结构主线，而不是逐字转录。

## Fast Look

**结构链路**：`文本 → Tokenizer（BPE / BBPE）→ Token Embedding → 重复 N 层 [Pre-RMSNorm → Attention（MHA/GQA/MLA/DSA）→ Residual → Pre-RMSNorm → FFN/SwiGLU 或 MoE → Residual] → Final RMSNorm → LM Head → logits / sampling`。现代模型没有推翻 Llama 的 Decoder-only 主干，主要是在 **KV Cache、长序列 Attention、FFN 容量与训练稳定性** 上做效率演进。

#### Llama-style Decoder-only Block

- **描述**：自回归 Transformer 的标准骨架：token embedding 后堆叠 Attention 和 FFN 两个子层，以残差连接串联，最后经 LM Head 预测下一个 token。
- **与之前方法的区别**：相对 Encoder-only BERT，使用 causal mask、只能看左侧上下文并直接生成；相对早期 Transformer，现代实现通常使用 Pre-Norm、RoPE 与 SwiGLU。
- **优点**：训练目标、推理过程和对话生成一致；模块化主干成熟，便于按组件替换和扩展。
- **缺点/注意点**：Decode 必须逐 token 执行；完整 Attention 的 KV Cache 与长序列计算仍会成为推理瓶颈。

#### BPE / Byte-level BPE（BBPE）与 Tokenizer 指标

- **描述**：BPE 反复合并高频相邻符号对；BBPE 固定从 256 个 UTF-8 单字节开始，任意文本都可表示。Tokenizer 决定文本如何映射成 token ID，再由 embedding table 变成向量。
- **与之前方法的区别**：相对词级分词，子词/字节级方案不依赖固定词典；BBPE 相对普通字符 BPE 保证无 OOV。SentencePiece 是可训练 BPE/Unigram 的通用框架；tiktoken 是偏重高速 BBPE 编码的实现。
- **优点**：覆盖多语言、罕见词和代码；词表大小与序列长度可折中。
- **缺点/注意点**：分词并非中性的预处理。应看 **fertility**（一个词平均被切成多少 token）和 **parity**（不同语言相同语义的 token 数是否接近）；差 tokenizer 会让中文等语言消耗更多上下文和 API 成本。

#### LayerNorm / RMSNorm 与 Pre-Norm

- **描述**：LayerNorm 对每个 token 的 hidden 维计算均值、方差后归一化；RMSNorm 只按均方根缩放。Pre-Norm 把 Norm 放在 Attention/FFN 子层之前，残差分支保留恒等梯度路径。
- **与之前方法的区别**：RMSNorm 相对 LayerNorm 不再中心化，少一次均值相关计算；Pre-Norm 相对原始 Transformer 的 Post-Norm 改变了归一化位置。
- **优点**：RMSNorm 更轻量；Pre-Norm 的残差恒等路径通常使深层 LLM 更容易稳定训练。
- **缺点/注意点**：Pre-Norm 的残差幅值可能随层累积，需要 final norm、初始化和训练配方共同控制；不能因为少了均值就假设 RMSNorm 在所有架构中必然更好。

#### MHA → GQA → MLA：KV Cache 的压缩路径

- **描述**：MHA 每个 Query head 都有独立 K/V；GQA 让一组 Query head 共享一组 K/V；MLA（Multi-head Latent Attention）把 K/V 表示压到低维 latent cache，再在计算时恢复所需头部表示。
- **与之前方法的区别**：三者的共同目的都是减少 Decode 阶段的 K/V 存储与内存带宽：MHA 不共享、GQA 做组共享、MLA 做低秩 latent 压缩。
- **优点**：GQA 是质量—显存的简单折中；MLA 能在更强压缩下保留较丰富的 K/V 表示，是 DeepSeek 系的关键推理优化。
- **缺点/注意点**：GQA 的分组数与 MLA 的 latent 维度都是质量—成本超参数；MLA 不等于稀疏 Attention，它首先解决的是 **cache 表示**，不是“看哪些 token”。

#### Sparse Attention / DSA（DeepSeek Sparse Attention）

- **描述**：Sparse Attention 只让当前 token 与部分历史 token 计算 Attention；DSA 使用轻量索引器预测相关性，选择少数历史 token 参与细粒度计算，并保留局部窗口。
- **与之前方法的区别**：相对全量 Attention 的 `O(n²)` score 计算，DSA 改变了“读取哪些历史 token”；相对 MLA，二者可组合：MLA 压缩 KV，DSA 减少被访问的历史位置。
- **优点**：长上下文时显著减少 Attention 计算量；局部窗口保证近期 token 不被遗漏。
- **缺点/注意点**：若索引器漏召回关键远程 token，质量会下降；要区分论文/模型中的 DSA、CSA、HCA 等具体实现，不能把所有稀疏注意力视为同一种算法。详见 [DeepSeek V4 后训练](../../11.%20Agentic%20RL/DeepSeek%20V4%20后训练/DeepSeek%20V4%20后训练.md)。

#### SwiGLU FFN

- **描述**：在每层 Attention 后做逐 token 非线性变换；SwiGLU 用 SiLU/Swish 门控一条值分支，替代普通两层 FFN。
- **与之前方法的区别**：相对 ReLU/GELU FFN，增加输入相关的门控；它仍是稠密 FFN，即每个 token 经过同一组参数。
- **优点**：现代 Decoder-only LLM 的常见高效默认项，表达选择性更强。
- **缺点/注意点**：需额外投影；比较参数量或 FLOPs 时必须对齐中间维度，不能只比较激活函数名称。

#### MoE：Routed Expert、Shared Expert、Fine-grained Expert

- **描述**：用多组 Expert FFN 取代单个稠密 FFN。Router 为每个 token 选 Top-k routed experts；shared experts 则对每个 token 固定激活，承接通用知识。Fine-grained MoE 把一个粗专家拆成更多小专家，并相应增加 Top-k，以提高可组合性。
- **与之前方法的区别**：相对 dense FFN，MoE 增大总参数但每个 token 只激活少数专家；shared expert 相对纯 routed MoE 将通用知识和特化知识显式分工。
- **优点**：在近似 active FLOPs 下获得更大总容量；细粒度专家可形成更多组合，shared expert 减少各 routed expert 重复学习通用模式。
- **缺点/注意点**：总参数不等于每 token 成本；Router 可能塌缩，跨设备 expert parallel 会产生 All-to-All 通信瓶颈。详见 [MoE 论文](../1.MoE论文/1.MoE论文.md)。

#### Load Balancing、Auxiliary-Loss-Free 与 Token Drop

- **描述**：负载均衡约束 Router 不要把 token 集中到少数专家/设备；传统 auxiliary loss 用额外损失引导均匀路由；auxiliary-loss-free 通过动态更新专家 bias 调整“是否入选”，而不直接改入选后的原始门控权重。超过 capacity 的 token 可被 drop，只走残差。
- **与之前方法的区别**：前者以损失函数软约束路由，后者将负载控制从主任务损失中分离；Token Drop 是超容量时的硬保护，而非改善路由本身。
- **优点**：提高专家利用率、设备计算均衡和通信吞吐；auxiliary-loss-free 减少辅助损失对主任务最优路由的直接干扰。
- **缺点/注意点**：负载均衡与专家特化天然有张力；Token Drop 会损失当前层的专家变换，且可能造成训推不一致。工程上还需分别关注 expert、device 与通信接收侧的均衡。

#### LM Head 与 Multi-Token Prediction（MTP）

- **描述**：LM Head 把最终 hidden state 投影到词表 logits，用于 next-token prediction；MTP 在训练时额外预测后续多个 token，可作为多步监督信号，并可用于 speculative decoding 的草稿 token 提议。
- **与之前方法的区别**：普通自回归训练只监督下一个 token；MTP 增加未来多位置预测分支，但主模型的自回归输出接口仍可保持不变。
- **优点**：增加训练监督密度；在合适推理系统中可帮助草稿—验证式加速。
- **缺点/注意点**：训练损失下降不必然转化为生成质量或端到端加速；实际加速依赖验证接受率、batch 与 serving runtime。

## 一张图理解模型演进

| 部位 | Llama 基线 | 现代演进代表 | 要解决的问题 |
| --- | --- | --- | --- |
| K/V 表示 | MHA / GQA | MLA | KV Cache 与 Decode 带宽 |
| 历史 token 访问 | Dense Attention | DSA 等 Sparse Attention | 长序列 Attention 计算 |
| FFN 容量 | Dense SwiGLU | MoE（routed + shared experts） | 容量与每 token FLOPs 脱钩 |
| 路由系统 | 无 | balance / bias / capacity / EP | 热点专家、通信和 token overflow |
| 训练观测 | token 数 / 参数量 | time-to-quality、active FLOPs、端到端吞吐 | 用真实成本比较架构 |

## Llama、Qwen、GLM、DeepSeek 的面试定位

- **Llama**：现代开源 Decoder-only 的清晰基线：RMSNorm、RoPE、SwiGLU、后续版本 GQA。先能画出这一层，再谈演进。
- **Qwen**：在同一 Decoder-only 主干上，系统性组合 GQA、RoPE、RMSNorm、SwiGLU、长上下文、Dense/MoE 与 reasoning/Agent 后训练；见 [Qwen 系列模型](../Qwen系列模型/Qwen系列模型.md)。
- **GLM**：预训练目标与 Llama 的纯 causal LM 不同；近期 GLM 系配置可使用 DSA + MLA 与 MoE，说明模型能力提升的架构侧仍聚焦稀疏化与效率。
- **DeepSeek**：MLA 主要减少 KV Cache，DeepSeek 稀疏注意力主要减少长上下文访问计算，DeepSeekMoE 通过细粒度 routed experts 与 shared experts 提高容量效率；这些是不同维度的优化，可同时出现。

## 面试回答模板

**问：从 Llama 到最新 MoE / DeepSeek 类模型，结构发生了什么本质变化？**

> 主干没有变：仍是 Decoder-only Transformer，token 经 Norm—Attention—残差—Norm—FFN/MoE—残差，最后由 LM Head 预测下一个 token。变化集中在效率：GQA/MLA 压 KV Cache，DSA 等稀疏机制少算不相关历史，MoE 让总参数容量大于每 token 激活计算。评价不应只看参数量或 token 数，而要同时看 active FLOPs、KV 显存、通信、训练收敛时间和最终端到端吞吐。

## 关联阅读

- [Attention 与 FlashAttention](../1.attention/1.attention.md)
- [MHA / MQA / GQA](../MHA_MQA_GQA/MHA_MQA_GQA.md)
- [LayerNorm 与 RMSNorm](../2.layer_normalization/2.layer_normalization.md)
- [Tokenizer](../4.tokenize分词/4.tokenize分词.md)
- [Llama 系列模型](../llama系列模型/llama系列模型.md)
- [DeepSeek V4 后训练：MLA、DSA、CSA、HCA](../../11.%20Agentic%20RL/DeepSeek%20V4%20后训练/DeepSeek%20V4%20后训练.md)
