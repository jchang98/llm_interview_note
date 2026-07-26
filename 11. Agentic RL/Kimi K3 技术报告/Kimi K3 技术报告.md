# Kimi K3 技术报告笔记：KDA、AttnRes 与 2.8T 稀疏 MoE

> **资料边界**：截至 2026-07-26，Moonshot 已发布 [Kimi K3 官方技术博客](https://www.kimi.com/blog/kimi-k3)，并明确表示完整架构、训练和评测技术报告将后续发布。本文只总结该博客明确披露的内容；路由损失、数据配比、训练 token、并行规模和 RL 超参等未公开信息不作推断。小红书已通过 `xhs-cli` 搜索并直读《Kimi-K3技术报告：16/896的稀疏度如何可能》等解读，主要用于定位官方资料。

## TL;DR

Kimi K3 是一个 2.8T 参数、1M context、原生视觉、面向开放权重发布的 MoE 模型。它的主线不是只做更大 MoE，而是将：

```text
KDA（跨序列高效注意力）
  + AttnRes（跨深度选择性残差读取）
  + Stable LatentMoE（896 选 16）
  + Quantile Balancing / Per-Head Muon
  + MXFP4 / MXFP8 QAT 与全均衡 EP
  → 长程代码、知识工作、多模态 Agent 服务
```

官方称这些架构与训练配方相对 Kimi K2 约提升 2.5× overall scaling efficiency；该数字是整体经验性表述，不应拆解为任一单项组件的独立增益。

## Fast Look

**实体关系**：`输入（文本/视觉）→ KDA + Gated MLA 建模序列 → AttnRes 跨层选择性取表示 → SiTU 控制激活 → Stable LatentMoE 路由 16/896 专家 → Quantile Balancing 保持负载 → Per-Head Muon 优化注意力 → MXFP4/MXFP8 QAT + 全均衡 EP 训练 → KDA prefill cache / Mooncake 服务`。模型能力还依赖 preserved thinking history 的 Agent 协议，不能只替换底座模型。

#### Kimi K3（2.8T、原生视觉、1M Context）

- **描述**：K3 是 Moonshot 的 2.8T 参数模型，具有 native vision 能力与 1M token context，目标是长程 coding、knowledge work 和 reasoning；官方博客称完整权重计划于 2026-07-27 发布。
- **与之前方法的区别**：相对 Kimi K2，官方披露 K3 引入 KDA、AttnRes，并把 MoE 稀疏度推进到 896 选 16；相对纯文本 LLM，视觉输入进入同一原生模型而非完全外接式流程。
- **优点**：将长上下文、视觉理解和工具型任务放在同一模型能力范围；高稀疏度使总容量可扩展到多万亿参数而不激活全部专家。
- **缺点/注意点**：2.8T 是总参数而非每 token 计算量；1M context 是能力上限，不代表长任务无需上下文管理。官方也承认整体用户体验仍落后于 Claude Fable 5 与 GPT-5.6 Sol。

#### Kimi Delta Attention（KDA）

- **描述**：KDA 是 K3 的注意力骨架，官方将其定义为面向大规模序列建模的高效 attention foundation；其目标是改善信息沿序列长度的流动和服务效率。
- **与之前方法的区别**：相对常规全量 Transformer attention，KDA 不是仅依赖标准 KV cache 的路径；官方特别说明它对传统 prefix cache 带来新挑战，因此需要专门的 prefill-cache 实现。
- **优点**：为 1M context 和超大模型提供更可扩展的注意力基础，并与 KDA prefill cache 配合降低服务成本。
- **缺点/注意点**：官方技术博客尚未公开完整公式、状态表示、复杂度与 kernel 细节；不能把 KDA简单等同于线性注意力或标准 MLA，部署需等待官方实现与兼容性说明。

#### Attention Residuals（AttnRes）

- **描述**：AttnRes 在模型深度方向选择性检索先前层的表示，而不是把所有残差一律累加；与 KDA 共同构成 K3 的架构主干。
- **与之前方法的区别**：相对标准 residual connection 的逐层固定相加，AttnRes 将跨层信息读取改为选择性机制，试图降低深层网络中无差别累积的干扰。
- **优点**：理论上可让深层按需复用有用表示，改善跨深度信息流，并支持更大模型的稳定扩展。
- **缺点/注意点**：博客未披露选择函数、额外计算、训练稳定性曲线及消融数据；其具体收益不能脱离 KDA、MoE 与训练配方单独归因。

#### Stable LatentMoE（16 / 896 Experts）

- **描述**：K3 的稀疏 MoE 每个 token 有效激活 896 名专家中的 16 名；Stable LatentMoE 是支撑这一极高稀疏度的稳定路由/表示框架。
- **与之前方法的区别**：相对低专家数或较高激活比例的 MoE，K3 将总专家数显著放大、激活比例压到约 1.8%；相对稠密 FFN，每个 token 只走少量专家。
- **优点**：在有限 active FLOPs 下获得更大参数容量与更细粒度的专家分工；是 K3 相对 K2 提高 scaling efficiency 的关键组成。
- **缺点/注意点**：16/896 的极端稀疏使路由错误、负载不均、all-to-all 通信和专家并行吞吐成为一等问题；官方未公开每 token active 参数量，不能仅由专家数量推算成本。

#### Quantile Balancing（分位数负载均衡）

- **描述**：从 router score 的分位数直接推导专家分配，避免依赖启发式更新与一个敏感的 balancing 超参数。
- **与之前方法的区别**：相对在辅助损失或手工阈值/动态规则中反复调节平衡项，Quantile Balancing 直接以路由分数分布决定分配边界。
- **优点**：减少调参敏感度，旨在让高专家数、低激活率下的专家利用率更稳定。
- **缺点/注意点**：博客未给出数学定义与和 auxiliary-loss-free/其他 routing 的对照；分位数估计本身也可能随 batch、数据分布和并行切分变化。

#### Per-Head Muon

- **描述**：将 Muon 优化器扩展到按 attention head 独立优化，使不同头可采用更自适应的更新。
- **与之前方法的区别**：相对对整个注意力投影或层使用统一更新尺度，Per-Head Muon 将优化粒度细化到 head。
- **优点**：可针对 head 间统计差异调整学习动态，服务于超大模型的稳定和高效训练。
- **缺点/注意点**：这不是对所有参数“逐头优化”；其精确作用范围、状态开销和超参仍未由 K3 博客完整公开，不能直接照搬为通用最优配置。

#### Gated MLA 与 SiTU（Sigmoid Tanh Unit）

- **描述**：Gated MLA 用门控增强注意力选择性；SiTU 是 K3 采用的激活控制单元。官方将二者列为提升 attention selectivity 与 activation control 的组件。
- **与之前方法的区别**：相对无额外门控的 MLA，Gated MLA 增加选择机制；相对 ReLU/GELU/SwiGLU 等常用激活，SiTU 使用 sigmoid 与 tanh 相关的门控/非线性设计。
- **优点**：将“读哪些 token”和“激活哪些特征”的控制变得更细，可能有助于高稀疏 MoE 与长序列场景的稳定性。
- **缺点/注意点**：博客没有给出 SiTU 的精确公式和 Gated MLA 的完整投影结构；不要仅凭名称把 SiTU 当作任意已有激活函数的同义词。

#### MXFP4 / MXFP8 Quantization-Aware Training（QAT）

- **描述**：K3 从 SFT 阶段开始进行 QAT，使用 MXFP4 权重和 MXFP8 激活，官方目标是获得更广泛硬件兼容性。
- **与之前方法的区别**：相对训练完成后再做 PTQ，QAT 让模型在训练期适应低精度误差；相对统一精度方案，权重与激活采用不同微缩浮点格式。
- **优点**：降低权重带宽和显存压力，缩小训练精度与部署精度之间的落差，为大规模 MoE 推理提供可部署性。
- **缺点/注意点**：QAT 增加训练复杂度，对硬件格式、kernel 和数值稳定性敏感；需要同时验证质量、通信、KV/prefill cache 与端到端时延，不能只比较单模型 FLOPs。

#### Fully Balanced Expert Parallel（静态 shape、无主机同步 EP）

- **描述**：为避免专家失衡在大 expert-parallel 规模下损害吞吐，K3 使用完全均衡的 EP 训练方法，保持 static shapes，关键路径上不进行 host synchronization。
- **与之前方法的区别**：相对动态 token dispatch 或依赖 CPU/host 协调的负载修正，该设计优先保证通信形状和关键路径稳定。
- **优点**：减少负载抖动与主机同步开销，使大规模 all-to-all 更可预测，配合 Quantile Balancing 支持极稀疏 MoE。
- **缺点/注意点**：静态 shape 可能带来 padding 或容量浪费；它解决的是训练吞吐与平衡，不替代推理时的动态流量控制。

#### KDA Prefill Cache、Mooncake 与 64+ Accelerator Supernode

- **描述**：KDA 需要专用 prefill cache 实现；官方建议在 64 个及以上加速器的 supernode 上部署，以获得更大的高带宽通信域。官方 API 使用 Mooncake 的 disaggregated inference 架构。
- **与之前方法的区别**：相对可直接复用传统 KV prefix cache 的 Transformer 服务，KDA 的缓存语义/实现需要适配；相对小规模节点部署，supernode 显式把高速互联域作为性能前提。
- **优点**：提高 cache-hit 价值和专家通信效率，降低大 MoE 服务的 token 成本；官方称其编码工作负载 cache hit rate 超过 90%。
- **缺点/注意点**：64+ 加速器是部署建议而非最低运行要求；缓存命中率强依赖请求分布与调度，Mooncake/KDA cache 的完整实现细节仍待开源，不能把官方 API 指标外推到任意自部署。

#### Preserved Thinking History（保留思维历史）

- **描述**：K3 使用 preserved thinking history 模式训练；Agent harness 需要把历史 thinking 内容完整回传给模型。
- **与之前方法的区别**：相对每轮丢弃或仅保留摘要的推理历史，K3 假定后续 turn 可看到此前完整思考状态，并将其纳入行为分布。
- **优点**：长程 Agent 可维持中间假设、工具计划与自检链条，减少重复推理和跨轮策略漂移。
- **缺点/注意点**：这是实际兼容性约束：缺失历史 thinking，或从其他模型中途切换到 K3，可能使质量明显不稳定；同时也增加上下文成本、隐私暴露与错误累积风险。

#### Thinking Effort 与长程 Agent 行为

- **描述**：发布时 K3 默认 max thinking effort，后续计划提供 low/high；训练偏向长程高难任务，官方展示其在代码、研究、知识工作与多模态创建中的 Agent 例子。
- **与之前方法的区别**：相对固定单一推理预算，effort 允许按任务选择测试时计算；相对短单轮助手，目标是多工具、长会话的自主执行。
- **优点**：复杂工程任务可使用更充分的推理与工具循环；默认最大努力有利于展示上限能力。
- **缺点/注意点**：更多 thinking 直接增加成本和时延，并可能导致过度主动：官方明确警告它可能在意图模糊时替用户做意外决定，生产中须以系统提示、`AGENTS.md`、权限和人工确认收紧边界。

## 版本与证据地图

| 结论 | 证据状态 |
| --- | --- |
| 2.8T、1M context、native vision、KDA、AttnRes、16/896 experts | 官方技术博客明确披露 |
| Stable LatentMoE、Quantile Balancing、Per-Head Muon、Gated MLA、SiTU | 官方技术博客明确披露名称与高层作用 |
| MXFP4 weights / MXFP8 activations QAT、全均衡 EP、KDA prefill cache | 官方技术博客明确披露 |
| 完整 KDA / AttnRes 公式、SiTU 定义、路由损失、训练数据/超参、完整 benchmark protocol | 完整技术报告尚未发布，不能补写为事实 |

## 面试回答框架

**Q：Kimi K3 的技术主线是什么？**

K3 将高稀疏 MoE 与系统化训练/服务协同推进：KDA 和 AttnRes 分别针对序列长度与网络深度的信息流，Stable LatentMoE 以 16/896 专家获得大总容量；Quantile Balancing、Per-Head Muon、QAT 与静态全均衡 EP 用于让极稀疏 MoE 能稳定训练和高效通信；服务端再用 KDA prefill cache、Mooncake 与大通信域 supernode 降低成本。它的代价是部署与 Agent 协议高度耦合，尤其需要完整 preserved thinking history。

**Q：16/896 意味着模型每 token 只计算 1.8% 参数吗？**

只能说明在路由专家这一层面，激活 16/896 名专家；总活跃参数还取决于共享专家、attention、embedding、专家宽度及路由实现。正确表述是“极高 MoE 稀疏度”，不能把 `16 ÷ 896` 直接当成整模型 active-parameter 比例或真实成本比例。

## 小红书检索对应与参考资料

- [Kimi-K3技术报告：16/896的稀疏度如何可能](https://www.xiaohongshu.com/explore/6a595a36000000000503a31a?xsec_token=ABXudNKJKEhtMEi8Px83ZnRfpXpiISB8UuTHsWrCZboLM%3D&xsec_source=pc_search)：已直读；其中列出的 K3 官方博客、Muon、KDA 与 Attention Residuals 资料可作为延伸阅读线索，本文事实以官方博客为准。
- [Kimi K3 官方技术博客：Open Frontier Intelligence](https://www.kimi.com/blog/kimi-k3)
- [MoonshotAI 官方组织](https://github.com/MoonshotAI)
