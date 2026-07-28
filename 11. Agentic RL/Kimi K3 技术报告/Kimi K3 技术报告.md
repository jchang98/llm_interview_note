# Kimi K3 技术报告笔记：KDA、AttnRes 与 2.8T-A104B 稀疏 MoE

> **资料边界（更新至 2026-07-28）**：Moonshot 已于 2026-07-27 同步公开 [Kimi K3 完整技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) 与开放权重。本页以官方报告、[官方仓库 README](https://github.com/MoonshotAI/Kimi-K3) 和技术博客为事实来源；小红书/知乎“报告解读”只用于发现阅读线索，不把二手推测写成事实。报告仍未完整披露原始语料配比、总训练 token、完整 RL 超参等细节。

## TL;DR

Kimi K3 是一个 `2.8T-A104B`、1M context、原生视觉、已开放权重的 MoE 模型。它的主线不是只做更大 MoE，而是将：

```text
KDA（跨序列高效注意力）
  + AttnRes（跨深度选择性残差读取）
  + Stable LatentMoE（896 个路由专家选 16 + 2 个共享专家）
  + Quantile Balancing / Per-Head Muon
  + MXFP4 / MXFP8 QAT 与全均衡 EP
  → 长程代码、知识工作、多模态 Agent 服务
```

文本骨干共 93 层：`69 KDA + 24 Gated MLA`，按 `3× KDA → 1× Gated MLA` 交错，并在最后额外插入一层 MLA；92 个 MoE 层与 1 个 dense 层组成宽度方向的计算。官方称这些架构与训练配方相对 Kimi K2 约提升 2.5× overall scaling efficiency；该数字是**整体**经验性表述，不应拆解为任一单项组件的独立增益。

## Fast Look

**实体关系**：`文本 / 图像 / 视频 → MoonViT-V2 + projector → KDA + Gated MLA 建模序列 → AttnRes 跨层选择性取表示 → SiTU-GLU 控制激活 → Stable LatentMoE 路由 16/896 专家 + 2 shared experts → Quantile Balancing 保持负载 → Per-Head Muon 优化注意力 → MXFP4/MXFP8 QAT + MoonEP / FlashKDA 训练 → KDA prefill cache / Mooncake 服务`。后训练则是 `任务合成 → AgentEnv 可分叉沙箱 → 百万 token RL → preserved thinking history 的 Agent 协议`；不能只替换底座模型而忽略运行时。

#### Kimi K3（2.8T-A104B、93 层、原生视觉、1M Context）

- **描述**：K3 是 Moonshot 已开放权重的 native multimodal MoE：总参数 2.8T、每 token 激活 104B、93 个 decoder 层、96 个 attention heads、160K vocabulary、1,048,576 context。文本与视觉共享一个端到端模型路径。
- **与之前方法的区别**：相对 Kimi K2，K3 用 KDA + AttnRes 改造序列/深度信息流，并把 MoE 推进到 896 routed experts 中 Top-16；相对纯文本 LLM，MoonViT-V2 的视觉特征经轻量 projector 接入同一 embedding 空间。
- **优点**：将长上下文、视觉理解和工具型任务放在同一模型能力范围；`A104B` 明确区分每 token 激活计算与 2.8T 总容量，便于部署估算。
- **缺点/注意点**：104B 仍是极大的 active 模型，且完整专家权重需驻留在 EP GPU 域；1M 是最大窗口而非免上下文管理保证，报告的 Agent 成绩也依赖具体 harness、effort 与工具配置。

#### Kimi Delta Attention（KDA）与 3:1 Hybrid Attention

- **描述**：KDA 是带细粒度门控的 delta-rule **线性注意力**，以固定大小的 recurrent state 承载长程 token mixing；K3 每 3 层 KDA 插入 1 层 Gated MLA（共 69/24），以 latent KV 的显式全局读取补足线性状态的选择性。
- **与之前方法的区别**：相对全层 dense attention，KDA 的状态不随序列长度线性增长；相对“全部改线性注意力”，K3 保留周期性 Gated MLA，因此是 hybrid attention 而非纯 KDA。
- **优点**：大部分层避免长度相关的 KV cache，适合百万上下文；MLA 层提供高容量全局检索，形成速度—表达能力折中。
- **缺点/注意点**：KDA 的 prefix/prefill cache 与传统 Transformer KV cache 语义不同，需专门实现；只有 24 层 MLA 保留 per-token latent KV，长上下文成本降低不等于归零。

#### Block Attention Residuals（AttnRes）

- **描述**：AttnRes 把“跨层残差”改为注意力读取：learned pseudo-query 对 embedding、当前 block 与此前 block 的表示生成权重，选择性混合跨深度信息。K3 使用 block 变体，将层聚为 block 后保存/读取 block representation。
- **与之前方法的区别**：相对标准 residual 的逐层固定相加，AttnRes 不把所有历史压进单一 residual stream；相对逐层 AttnRes，block 化将状态量和跨 stage 通信从按层保存降为按 block 保存。
- **优点**：可按需直接读取更早层的表示，改善深层梯度与信息流；报告给出 block 设计以降低 memory/通信开销。
- **缺点/注意点**：它带来额外 pseudo-query、归一化和跨 block 状态；其收益须与计算/内存代价、block size 和其他训练配方共同评估，不能将 2.5× 总效率全部归因于 AttnRes。

#### Stable LatentMoE（16 / 896 Routed + 2 Shared Experts）

- **描述**：K3 的每个 MoE 层先将 7,168 维 hidden state 投影到 3,584 维 latent，再做专家计算；每个 token 从 896 个 routed experts 选择 16 个，并恒定经过 2 个 shared experts。
- **与之前方法的区别**：相对低专家数或较高激活比例的 MoE，K3 将总专家数显著放大、激活比例压到约 1.8%；相对稠密 FFN，每个 token 只走少量专家。
- **优点**：在有限 active FLOPs 下获得更大参数容量与更细粒度的专家分工；是 K3 相对 K2 提高 scaling efficiency 的关键组成。
- **缺点/注意点**：16/896 的极端稀疏使路由错误、负载不均、all-to-all 通信和专家并行吞吐成为一等问题；`104B active` 是整模型口径，不能由 `16 ÷ 896` 或单层专家数自行反推。

#### Quantile Balancing（分位数负载均衡）与 Soft Drop

- **描述**：K3 采用 auxiliary-loss-free 路由：expert-specific bias 只用于 Top-k 的 dispatch 选择，不进入已选专家的混合权重。Quantile Balancing 以当前路由分数的分位数更新该 bias；溢出 token 使用 soft drop 机制处理。
- **与之前方法的区别**：相对 auxiliary-loss 路由，负载控制不通过额外损失直接干预 router 目标；相对固定步长更新 bias，分位数统计让更新幅度随当前分数分布自适应。
- **优点**：减少调参敏感度，旨在让高专家数、低激活率下的专家利用率更稳定。
- **缺点/注意点**：报告给出了路由/bias 的定义，但其稳定性仍依赖 batch、数据分布和并行切分；soft drop 是容量保护，不应被误解为“无损负载均衡”。

#### Per-Head Muon

- **描述**：将 Muon 优化器扩展到按 attention head 独立优化，使不同头可采用更自适应的更新。
- **与之前方法的区别**：相对对整个注意力投影或层使用统一更新尺度，Per-Head Muon 将优化粒度细化到 head。
- **优点**：可针对 head 间统计差异调整学习动态，服务于超大模型的稳定和高效训练。
- **缺点/注意点**：这不是对所有参数“逐头优化”；其精确作用范围、状态开销和超参仍未由 K3 博客完整公开，不能直接照搬为通用最优配置。

#### Gated MLA 与 SiTU-GLU（Sigmoid Tanh Unit）

- **描述**：Gated MLA 在 MLA 输出端加入输入相关的全秩 channel gate；SiTU-GLU 是 K3 MoE 的门控激活，以有界的 tanh 类门分支控制特征幅度。
- **与之前方法的区别**：相对无额外门控的 MLA，Gated MLA 让每个 token 选择性调制读取到的通道；相对 SwiGLU，SiTU-GLU 重点引入有界门控以抑制大激活。
- **优点**：将“读哪些 token”和“激活哪些特征”的控制变得更细，可能有助于高稀疏 MoE 与长序列场景的稳定性。
- **缺点/注意点**：Gated MLA 与 SiTU-GLU 是 K3 的特定组合，不能只因名称相似就直接替换进其他 checkpoint；实际收益仍需结合 kernel、量化和训练配方验证。

#### MoonViT-V2（从零训练的原生视觉编码器）

- **描述**：MoonViT-V2 是约 401M 参数、27 层的视觉 Transformer；图像与视频共用参数，空间 attention 与时间 attention 因子化，输出经轻量 MLP projector 映射到 LLM embedding 空间。
- **与之前方法的区别**：相对 Kimi K2.5/SigLIP 初始化视觉塔，K3 从零开始以 next-token prediction 联合训练，不使用对比预训练初始化。
- **优点**：报告显示它在视觉评测上达到 SigLIP 初始化基线，同时梯度范数更低、更少尖峰，联合优化更稳定；原生路径支持图像/视频与文本共同建模。
- **缺点/注意点**：约 0.4B 视觉塔和视频 token 仍会占用显存、上下文与预填充时间；“原生多模态”不代表所有视觉任务必然优于专用 VLM 或外部工具链。

#### MXFP4 / MXFP8 Quantization-Aware Training（QAT）

- **描述**：K3 从 SFT 阶段开始进行 QAT，使用 MXFP4 权重和 MXFP8 激活，官方目标是获得更广泛硬件兼容性。
- **与之前方法的区别**：相对训练完成后再做 PTQ，QAT 让模型在训练期适应低精度误差；相对统一精度方案，权重与激活采用不同微缩浮点格式。
- **优点**：降低权重带宽和显存压力，缩小训练精度与部署精度之间的落差，为大规模 MoE 推理提供可部署性。
- **缺点/注意点**：QAT 增加训练复杂度，对硬件格式、kernel 和数值稳定性敏感；需要同时验证质量、通信、KV/prefill cache 与端到端时延，不能只比较单模型 FLOPs。

#### MoonEP / Fully Balanced Expert Parallel（静态 shape、无主机同步 EP）

- **描述**：为避免专家失衡在大 expert-parallel 规模下损害吞吐，K3 使用 MoonEP 的完全均衡 EP 训练方法，保持 static shapes，关键路径上不进行 host synchronization。
- **与之前方法的区别**：相对动态 token dispatch 或依赖 CPU/host 协调的负载修正，该设计优先保证通信形状和关键路径稳定。
- **优点**：减少负载抖动与主机同步开销，使大规模 all-to-all 更可预测，配合 Quantile Balancing 支持极稀疏 MoE。
- **缺点/注意点**：静态 shape 可能带来 padding 或容量浪费；它解决的是训练吞吐与平衡，不替代推理时的动态流量控制。

#### FlashKDA（KDA 的高性能 Prefill Kernel）

- **描述**：FlashKDA 是 Kimi 为 KDA 实现的高性能 kernel，面向线性注意力的 prefill 阶段；官方已将其作为独立基础设施项目开放。
- **与之前方法的区别**：相对直接使用通用 linear-attention 实现，它针对 KDA 的状态更新与内存访问模式专门优化；相对只优化 MoE 的 MoonEP，它解决的是 attention 算子而非专家 all-to-all 通信。
- **优点**：官方在 NVIDIA H20 上报告，相比 `flash-linear-attention` 基线，prefill 速度提高约 1.72–2.22 倍；这使 KDA 的长前缀处理优势能真正落到服务吞吐上。
- **缺点/注意点**：该数字是特定硬件、版本和 **prefill** 场景的 kernel 对比，不能外推为整模型端到端加速比；部署仍需核对 GPU、编译链和推理引擎是否支持该后端。

#### AgentEnv（面向 Agent RL 的高保真可分叉沙箱）

- **描述**：AgentEnv 是 Kimi 与 KVCache.ai 合作的 Agent 运行环境，用于大规模后训练；它强调强隔离，并支持环境快照、恢复和 fork，以便并行运行与复用任务状态。
- **与之前方法的区别**：相对每个 rollout 从零初始化的普通容器/脚本环境，AgentEnv 把可恢复、可分叉的交互式工具环境作为训练基础设施；相对 MoonEP/FlashKDA，它服务于任务环境与 RL rollout，而非模型算子或 GPU 通信。
- **优点**：降低长程工具任务的环境重建成本，便于从同一中间状态探索多个分支，并让 Agent 训练更接近真实编码/工具工作流。
- **缺点/注意点**：快照和 fork 不会自动保证任务真实性、评测无泄漏或奖励正确；环境镜像、权限、网络、测试数据与成本上限仍需自行治理。

#### 大规模任务合成与百万 Token Agent RL

- **描述**：官方说明 K3 的后训练覆盖通用推理、通用 Agent、编程 Agent 三类任务，并进行大规模任务合成与百万 token 上下文的 RL 基建；AgentEnv 是其中的环境层。
- **与之前方法的区别**：相对以短问答偏好数据为主的后训练，重点转向多步工具调用、长轨迹验证与长上下文状态管理；相对只训练语言模型回答，训练对象包含“模型 + 工具环境 + rollout”的闭环。
- **优点**：更贴近代码、研究和知识工作中“执行—观察—修正”的任务形态，也解释了 K3 对 preserved thinking history 和 Agent harness 的依赖。
- **缺点/注意点**：报告/公开文章没有完整披露数据来源、任务配比、奖励设计、采样预算和各项独立增益；不能仅凭 benchmark 或案例反推出通用 Agent 成功率。

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

- **描述**：K3 始终返回 `reasoning_content`，通过顶层 `reasoning_effort` 设置 `low`、`high` 或 `max`（默认 `max`）；训练偏向长程高难任务，官方展示其在代码、研究、知识工作与多模态创建中的 Agent 例子。
- **与之前方法的区别**：相对固定单一推理预算，effort 允许按任务选择测试时计算；相对短单轮助手，目标是多工具、长会话的自主执行。
- **优点**：复杂工程任务可使用更充分的推理与工具循环；默认最大努力有利于展示上限能力。
- **缺点/注意点**：更多 thinking 直接增加成本和时延，并可能导致过度主动：官方明确警告它可能在意图模糊时替用户做意外决定，生产中须以系统提示、`AGENTS.md`、权限和人工确认收紧边界。

## 报告解读的统一视角：三维选择性计算

这是对 XHS / 知乎解读的**组织框架**，不是官方另行命名的单一模块。它有助于把看似分散的名词放回同一张图：

```text
序列维度：KDA 选择如何压缩、保留和更新历史
深度维度：Block AttnRes 选择读取哪些早期 block 表示
宽度维度：Stable LatentMoE 选择激活哪些专家
                 ↓
训练/系统维度：Quantile Balancing + MoonEP + FlashKDA + AgentEnv
                 ↓
长程 Agent：百万 token RL + 完整 thinking 历史 + 工具 harness
```

- **不要混淆**：KDA 与 AttnRes 都涉及“选择”，但前者处理 token 序列中的历史状态，后者处理网络深度中的历史表示；MoE 则是在同一层的宽度方向选择计算分支。
- **系统含义**：K3 的 2.5× scaling efficiency 是上述架构、优化器、量化、通信、算子和训练流程的联合结果，不是三条路径中任意一条的独立增益。
- **工程含义**：模型权重开放不等于开箱即得相同 Agent 效果。长上下文缓存、专家通信、运行环境、工具权限、thinking-history 协议和评测 harness 都是能力闭环的一部分。

## 版本与证据地图

| 结论 | 证据状态 |
| --- | --- |
| 2.8T-A104B、93 层、69 KDA + 24 Gated MLA、896 routed + 2 shared experts | 官方仓库与完整技术报告明确披露 |
| KDA 3:1 hybrid、Block AttnRes、Stable LatentMoE、Quantile Balancing、SiTU-GLU、MoonViT-V2 | 完整技术报告明确披露 |
| MXFP4 weights / MXFP8 activations QAT、MoonEP、KDA prefill cache、preserved thinking history | 官方仓库与完整技术报告明确披露 |
| 原始训练语料比例/总 token、全部 RL 超参、各单项架构改动的独立因果收益 | 报告仍未完整公开；不能从总 2.5× scaling efficiency 反推 |

## 报告评测：应如何阅读分数

官方表格显示 K3 在 `max` thinking effort 下取得 GPQA Diamond 93.5、DeepSWE 67.5、Terminal-Bench 2.1 88.3、BrowseComp 91.2、MCPMark-Verified 94.5 等分数。它们说明 K3 在 coding / agent / 长上下文任务上具备前沿竞争力，但不能把表格读作“模型本体的绝对排名”：

- **Harness 是变量**：DeepSWE、Terminal-Bench、SWE-Marathon、Agents' Last Exam 等分别使用 Kimi Code、Claude Code、Codex 或其他实现；同一底座模型换 harness 可能明显变分。
- **工具增强是变量**：HLE-Full、MMMU-Pro、CharXiv、MathVision、ZeroBench 的 `a / b` 分数分别对应不使用 / 使用工具，不能只摘取后者比较。
- **推理预算是变量**：K3 取 `reasoning_effort=max`、temperature 1.0；单步任务 top-p 0.95、Agent 任务 top-p 1.0。它反映的是最大努力配置，不是最低延迟配置。
- **部分对比来自外部榜单**：报告明确标注部分对手/任务来自 Artificial Analysis、Vals AI 或官方 leaderboard，时间点也不同；应优先比较相同 benchmark、版本、harness、预算与工具权限下的复现实验。

## 面试回答框架

**Q：Kimi K3 的技术主线是什么？**

K3 将高稀疏 MoE 与系统化训练/服务协同推进：KDA 和 AttnRes 分别针对序列长度与网络深度的信息流，Stable LatentMoE 以 16/896 专家获得大总容量；Quantile Balancing、Per-Head Muon、QAT 与静态全均衡 EP 用于让极稀疏 MoE 能稳定训练和高效通信；服务端再用 KDA prefill cache、Mooncake 与大通信域 supernode 降低成本。它的代价是部署与 Agent 协议高度耦合，尤其需要完整 preserved thinking history。

**Q：16/896 意味着模型每 token 只计算 1.8% 参数吗？**

只能说明在路由专家这一层面，激活 16/896 名专家；总活跃参数还取决于共享专家、attention、embedding、专家宽度及路由实现。正确表述是“极高 MoE 稀疏度”，不能把 `16 ÷ 896` 直接当成整模型 active-parameter 比例或真实成本比例。

## 小红书检索对应与参考资料

- [小红书：Kimi-K3技术报告：16/896的稀疏度如何可能](https://www.xiaohongshu.com/explore/6a595a36000000000503a31a?xsec_token=ABXudNKJKEhtMEi8Px83ZnRfpXpiISB8UuTHsWrCZboLM%3D&xsec_source=pc_search)：用于定位技术脉络；本文的架构、训练和评测事实均以以下官方材料为准。
- [小红书：Kimi K3 的三条高速路](https://www.xiaohongshu.com/explore/6a59b531000000001d0219d3)：将 KDA、AttnRes、LatentMoE 归纳为序列、深度、宽度三条信息路径；本文仅将它作为阅读框架，不将其未经报告支持的延伸说法当作事实。
- [小红书：Kimi K3 架构解读——把 attention 拆三轴](https://www.xiaohongshu.com/explore/6a68176e000000000503a53d)：用于定位 KDA / AttnRes / MoE 的三维解释与后训练线索。
- [知乎：Kimi K3 开放日（Kimi 官方）](https://zhuanlan.zhihu.com/p/2065221319797498039)：确认技术报告、权重及 MoonEP、FlashKDA、AgentEnv 的同步开放，并概述 K3 后训练和基础设施边界。
- [MoonshotAI / Kimi-K3 官方仓库、配置、评测与权重入口](https://github.com/MoonshotAI/Kimi-K3)
- [Kimi K3 完整技术报告 PDF](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
- [Kimi K3 官方技术博客：Open Frontier Intelligence](https://www.kimi.com/blog/kimi-k3)
- [Kimi Linear / KDA 技术报告](https://arxiv.org/abs/2510.26692)
