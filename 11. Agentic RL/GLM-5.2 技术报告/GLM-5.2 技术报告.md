# GLM-5.2 技术报告：长程 Agent、稀疏注意力与异步 RL

> **资料边界**：截至 2026-07，公开的完整论文是 [GLM-5 技术报告](https://arxiv.org/abs/2602.15763)，GLM-5.2 则以 [官方发布说明](https://z.ai/blog/glm-5.2) 与项目仓库为准。因此本文将“GLM-5 论文已披露的训练/后训练细节”与“GLM-5.2 已披露的产品与架构升级”分开陈述；没有公开依据的 5.2 训练细节不会补写为事实。

## Fast Look

**主线关系**：`MoE 基座（744B / 40B active）→ MLA / DSA 降低长上下文成本 → IndexShare 复用稀疏索引 → MTP 推测解码 → SFT 的思维协议 → 异步 Agentic RL → 10K+ 可验证环境 → 1M-token 长程执行`。前半段解决“算得起、读得长”，后半段解决“在工具环境中连续做对”。

#### GLM-5.2 与 GLM-5 / GLM-5.1

- **描述**：GLM-5.2 是面向长程 Agent 与代码任务的 744B-A40B 开放权重模型；官方称其稳定支持 1M token 上下文，并提供多档 reasoning effort。GLM-5 的完整报告给出了其 744B 总参数、40B 激活参数、256 experts、80 层及 28.5T 训练 token 的技术背景。
- **与之前方法的区别**：GLM-5 重点是从“vibe coding”走向可执行的 Agentic Engineering；GLM-5.1 强调跨数百轮工具调用维持策略迭代；5.2 将长上下文稳定性、代码能力与推理预算控制作为可用版本的主要升级。
- **优点**：把模型规模、长上下文和 Agent 轨迹训练合到一个基座；开放权重便于自部署、评测与微调。
- **缺点/注意点**：1M 是官方能力声明，不等于任意任务都能无损完成；端到端成本仍受 KV cache、工具延迟、环境可靠性和实际 agent loop 影响。

#### MoE（256 Experts，40B Active）

- **描述**：每个 token 仅路由到少数专家，GLM-5 以 256 专家、80 层形成 744B 总参数而每 token 激活约 40B 参数的稀疏模型。
- **与稠密模型的区别**：稠密 Transformer 每个 token 经过全部 FFN 参数；MoE 用路由器只计算少数专家，以较低的每-token FLOPs 换取更大总容量。
- **优点**：扩展参数容量而不同比例增加激活计算；适合代码、推理和 Agent 等多分布任务。
- **缺点/注意点**：路由失衡、专家并行通信和长序列 KV cache 都会成为系统瓶颈；总参数大也使部署和训练更复杂。

#### MLA-256 与 Muon Split

- **描述**：MLA（Multi-latent Attention）压缩 KV 为低维 latent cache；GLM-5 以 **Muon Split** 对分头投影分别做正交化更新，并通过增大 head dimension、减少 head 数形成 MLA-256，以降低解码计算。
- **与 GQA / 普通 MLA 的区别**：GQA 直接共享 KV 头；MLA 先压缩后重建 KV。论文发现原始 MLA 在 Muon 配方下弱于 GQA-8，Muon Split 用“各头独立尺度”的更新方式补回性能。
- **优点**：相比 GQA 更节省长上下文 KV 显存；MLA-256 兼顾 prefill 计算与 decode 阶段的点积成本。
- **缺点/注意点**：latent 压缩可能牺牲表示能力；MLA 解码点积仍可能高于小 head-dimension 的 GQA，硬件 roofline 会改变最优配置。

#### DSA（DeepSeek Sparse Attention）

- **描述**：Lightning Indexer 按 query 选择 top-k 相关 token，再执行稀疏注意力；将核心注意力从密集的 `O(L²)` 降为 `O(Lk)`。
- **与 Sliding Window Attention 的区别**：滑窗按距离固定丢弃远处 token；DSA 根据内容动态选择远距离 token，目标是保留长程依赖而非只看局部。
- **优点**：显著降低长序列注意力成本；GLM-5 通过从 dense 模型继续训练进行稀疏适配，减少从零训练的代价。
- **缺点/注意点**：Indexer 自身仍有 `O(L²)` 计算，且稀疏选择错误会损伤检索；它不是“上下文永远无损”的保证。

#### IndexCache / IndexShare

- **描述**：IndexCache 利用“跨层 token 选择稳定性”：相邻层的 DSA top-k 索引高度相似，少数 **Full layers** 运行 Indexer，多数 **Shared layers** 复用最近 Full layer 的索引。GLM-5.2 项目将对应升级称为 **IndexShare**：每四层共享一次 indexer。
- **与 DSA 的区别**：DSA 每层各算一次索引；IndexShare 不改变“按内容选 token”的稀疏注意力思想，而是跨层复用选择结果，专门削减 Indexer 成本。
- **优点**：论文在 30B DSA 模型上移除 75% indexer 计算且质量损失很小；官方称 5.2 在 1M 上下文将每-token FLOPs 降约 2.9×。这是支撑长程 prefill / decode 的关键工程改动。
- **缺点/注意点**：跨层注意力模式并非完全相同，过度共享会降低质量；“2.9×”是官方给定条件下的架构指标，端到端加速还取决于 KV、通信与工具等待。

#### MTP（Multi-token Prediction）与推测解码

- **描述**：MTP 预测未来多个 token，充当 draft model；主模型验证后一次接收多个 token。GLM-5 训练时共享 3 个 MTP layer 参数以控制显存，GLM-5.2 进一步提高 MTP 的接受长度。小红书解析称 5.2 的 MTP 还结合 IndexShare、KVShare 与 TV Loss 优化；在没有 5.2 单独技术报告前，这一实现组合仅作待核验线索。
- **与逐 token 解码的区别**：普通自回归每步只生成/验证一个 token；推测解码以少量额外草稿计算换取更少的主模型串行轮次。
- **优点**：降低解码时延、提高吞吐；官方称 GLM-5.2 接受长度最高提升 20%。
- **缺点/注意点**：收益依赖 draft 与 target 的一致性；接受率低时额外 MTP 计算可能抵消收益，并不改善模型本身的正确性。

#### 长程 RL、KVShare 与 Anti-Hack（社区解析，待官方核验）

- **描述**：直读的 GLM-5.2 社区解析称：长程 RL 从 group-wise 方法转向 critic-based PPO，并配合 token-level loss 适应上下文压缩；其还称采用两阶段 Anti-Hack（识别后在线拦截，以 dummy 结果替代直接终止轨迹）。同一来源把 KVShare、TV Loss 列为 MTP 优化组成。
- **与前述 GLM-5 异步 Agentic RL 的关系**：两者都服务于长轨迹训练；前者讨论的是奖励/优势估计和异常轨迹处理的可能升级，后者是 GLM-5 报告明确披露的异步采样、重要性修正与 stale-sample filtering。不能把前者反推为 GLM-5.2 已公开的完整训练配方。
- **潜在优点**：critic 可提供逐状态的价值基线，token-level 损失可能更适配被压缩或切分的长轨迹；不停训而返回 dummy 观察，可避免执行器异常直接截断整段 rollout；KVShare 有望进一步降低 draft 端缓存成本。
- **缺点/注意点**：这些均来自二手解析，尚缺官方技术报告佐证；critic 会引入价值估计偏差和额外训练成本，Anti-Hack 的误报/漏报以及 dummy 观察都可能改变策略学习分布，TV Loss 的具体定义也未由官方资料确认。

#### Interleaved Thinking、Preserved Thinking 与 Turn-level Thinking

- **描述**：SFT 模板显式训练三种思维协议：每次回复/工具调用前思考（Interleaved）、多轮保留已有思考块（Preserved）、按 turn 开关思考（Turn-level）。
- **与一次性 CoT 的区别**：普通 CoT 多在最终回答前推理一次；此处推理状态要跨工具调用与多轮会话持续衔接。
- **优点**：工具调用前可重估计划；保留思维减少重复推导与跨轮不一致；按需关闭思维可控制延迟。
- **缺点/注意点**：思维内容会占上下文和 token 预算；持久化错误假设也会持续传播，生产系统仍需外显状态、验证器和上下文压缩策略。

#### 异步 Agentic RL：TITO、双侧重要性采样与 stale-sample filtering

- **描述**：GLM-5 通过中央 Multi-Task Rollout Orchestrator 解耦 rollout 与训练。**TITO（Token-in-Token-out）** 保留 action 的原始 token，避免重分词错位；用 rollout log-prob 近似行为策略并以双侧阈值裁剪重要性比；过旧策略或环境崩溃造成的轨迹被丢弃。
- **与同步 PPO / 同步 RL 的区别**：同步方案须等待长轨迹结束，GPU 易空转；异步方案允许生成与更新并行，但天然带来 off-policy 偏差和轨迹版本不一致。
- **优点**：提高长程 Agent rollout 的吞吐与 GPU 利用率；能同时训练代码、终端和搜索任务。
- **缺点/注意点**：异步并不“免费更稳定”——裁剪阈值、版本陈旧度、失败样本处理与奖励噪声是关键超参；off-policy 偏差仍需严格控制。

#### SAO（Single-Rollout Asynchronous Optimization）

- **描述**：SAO 面向异步 Agentic RL：每个 prompt 只采样一条 rollout，而非 GRPO 的组内多条采样；再结合 value model 训练设计与严格的双侧、token-level clipping 控制离策略偏差和更新幅度。论文明确称其已部署在开放 GLM-5.2（论文写作 `750B-A40B`，与发布材料的 `744B-A40B` 为口径差异）Agentic RL 流水线中。
- **与之前方法的区别**：相对 GRPO 的 group-wise sampling，SAO 不等待同题完整组轨迹，因此更适合 rollout 完成时间不齐的异步系统；相对 GLM-5 已披露的 rollout-level 重要性修正，它将稳定性约束细化到 token 层，并重新引入经过专门设计的 value model。
- **优点**：单条轨迹可到达即训练，提高长程 Agent 任务的流水线利用率；双侧 token 裁剪降低策略版本漂移带来的极端梯度。论文报告其可稳定训练约 1,000 步，并在 Agent coding、推理与模拟在线学习中优于 GRPO 及若干变体。
- **缺点/注意点**：single-rollout 失去同 prompt 多样本的相对排序信号，效果依赖 critic/奖励质量；“部署于 GLM-5.2 pipeline”证明方法被采用，不等于公开了 GLM-5.2 的全部 SAO 超参、数据混合或训练时长。

#### CompactionRL（带上下文压缩的强化学习）

- **描述**：CompactionRL 将 context compaction 纳入 RL：模型既学习继续执行任务，也学习生成可供后续 rollout 使用的摘要；以 token-level loss normalization 和 cross-trajectory generalized advantage estimation（GAE）让被压缩、长度不一的轨迹能共同产生训练信号。论文称该方案已用于 GLM-5.2 的 RL pipeline。
- **与之前方法的区别**：普通 Agent RL 往往在 token 超窗时截断轨迹，或把摘要作为训练外的固定预处理；CompactionRL 将“何时/如何压缩”产生的状态转移放入同一优化闭环，并对压缩前后轨迹的 token 和优势估计做适配。
- **优点**：把有限 context window 延伸为可学习的长程任务能力；论文在 GLM-4.5-Air 与 GLM-4.7-Flash 的 SWE-bench Verified、Terminal-Bench 2.0 上报告一致增益，因而提供了迁移到 GLM-5.2 的直接训练证据。
- **缺点/注意点**：摘要不可逆，遗漏约束会导致后续 action 基于错误状态；cross-trajectory GAE 与 token normalization 会增加实现复杂度，且论文对小/中型 GLM 的收益不能定量外推为 GLM-5.2 的具体提升。

#### DP-aware Routing 与 KV Cache Locality

- **描述**：将同一 Agent rollout 固定路由到同一个数据并行 rank，以一致性哈希与轻量重平衡维持前缀 KV cache 的局部性。
- **与普通负载均衡的区别**：纯负载均衡可能把相邻 turn 分发到不同 rank，重复 prefill 相同历史；该机制把“缓存命中”视为调度目标。
- **优点**：长轨迹中 prefill 主要随新增 token 增长，降低重复计算并提高端到端吞吐。
- **缺点/注意点**：严格亲和性可能造成 rank 热点；需要在 cache locality 与负载均衡之间动态折中。

#### 可验证 Agent 环境与 On-policy Cross-stage Distillation（OPCSD）

- **描述**：GLM-5 构建超过 10K 个真实 SWE、终端与多跳搜索环境，以执行结果提供 grounded feedback；训练顺序为 Reasoning RL → Agentic RL → General RL，最后用在策略跨阶段蒸馏恢复早期能力。
- **与仅偏好数据的区别**：偏好优化主要比较文本输出；可验证环境可提供测试、命令、检索结果等可执行反馈。跨阶段蒸馏则防止后段对齐覆盖前段推理/Agent 能力。
- **优点**：奖励更贴近真实任务完成；减少多阶段 RL 的灾难性遗忘。
- **缺点/注意点**：环境覆盖决定能力上限，sandbox 故障会污染奖励；“可验证”不代表覆盖所有真实软件工程约束。

## 版本与证据地图

| 结论 | 对应版本 | 证据状态 |
| --- | --- | --- |
| 744B 总参数、40B active、256 experts、80 层、28.5T token | GLM-5 | 技术报告明确披露 |
| MLA-256、Muon Split、DSA 继续训练、三层共享 MTP | GLM-5 | 技术报告明确披露 |
| 异步 Agentic RL、TITO、双侧重要性采样、DP-aware routing | GLM-5 | 技术报告明确披露 |
| 1M 稳定上下文、多档 reasoning effort、IndexShare、改进 MTP | GLM-5.2 | 官方发布/项目仓库披露 |
| SAO：single-rollout、value model、双侧 token-level clipping | GLM-5.2 | SAO 论文明确称已部署于其 Agentic RL pipeline；完整训练配方未公开 |
| CompactionRL：token-level loss normalization、cross-trajectory GAE、上下文压缩 RL | GLM-5.2 | CompactionRL 论文明确称已部署于其 RL pipeline；GLM-5.2 的增益数字未单独披露 |
| 5.2 的完整预训练数据配比、RL 超参、环境统计 | GLM-5.2 | 未见单独技术报告，不能从 GLM-5 直接外推；社区所述 critic-based PPO、Anti-Hack、KVShare/TV Loss 均待核验 |

## 一页式面试回答

> GLM-5/5.2 的核心不是单纯把 MoE 做大，而是围绕长程 Agent 的系统协同：MoE 提供容量，MLA/DSA 降低长上下文成本，IndexShare 继续削减 DSA 的跨层重复索引，MTP 解决 decode 串行瓶颈；训练上用思维协议和可验证环境生成多步轨迹，再用异步 RL 提高长 rollout 的资源利用率。最新论文把这一训练侧进一步具体化为 SAO（让异步单轨迹 RL 可稳定更新）和 CompactionRL（让上下文压缩后的长轨迹仍可学习）。最大代价是系统复杂度：稀疏注意力质量、异步 off-policy 稳定性、压缩误差、KV cache 调度与环境可靠性必须一起处理。

## 与小红书参考的对应

- [GLM5 系列重要优化算法——IndexCache](https://www.xiaohongshu.com/explore/6a410b1d0000000021019eec?xsec_token=ABsikOhZAaBphw3NZ9Kykt6icNckRijcXATyogAjWtJ7Y=&xsec_source=pc_search&source=web_search_result_notes)：已直读正文，并确认其关联 7 页 `IndexCache.pdf`；补充了“跨层 token 选择稳定性”、免训练 greedy 选择与 training-aware 两类索引器选择思路。算法结论仍以论文为准。
- [GLM-5.2 模型技术解析速看](https://www.xiaohongshu.com/explore/6a339e54000000001503cc4e?xsec_token=ABSx1GE-Y9yCnUUMpd3CweQTO_IMKweD14Sb-KTTVCwvM=&xsec_source=pc_search&source=web_search_result_notes)：已直读正文。其对 1M 上下文、IndexShare、MTP 的表述可与官方发布交叉参照；critic-based PPO、Anti-Hack、KVShare 与 TV Loss 则已单列为社区解析、待官方核验，不作为主证据。
- [两篇论文看 GLM5.2 后训练：SAO、CompactionRL](https://www.xiaohongshu.com/explore/6a543b9800000000210140eb?xsec_token=CBBnIGiMs6g4v5GslysKeEx48CmVM3732kgFG0l8UW6Tw%3D&xsec_source=app_share)：已直读正文；该笔记指向 SAO 与 CompactionRL 两篇论文。本文已以两篇原论文补入方法实体，而非仅依据图片解析。

## 原始资料

1. [GLM-5.2 官方发布](https://z.ai/blog/glm-5.2)
2. [GLM-5 项目仓库与 5.2 模型说明](https://github.com/zai-org/GLM-5)
3. [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)
4. [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201)
5. [SAO: Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)
6. [CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents](https://arxiv.org/abs/2607.05378)
