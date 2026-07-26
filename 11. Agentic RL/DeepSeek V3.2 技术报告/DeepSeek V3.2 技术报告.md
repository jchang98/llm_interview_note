# DeepSeek-V3.2 技术报告：DSA、可扩展 RL 与 Agent 任务合成

> **资料边界**：本笔记以 DeepSeek 的 [V3.2 技术报告](https://arxiv.org/abs/2512.02556) 为主证据；小红书搜索并直读的《Deepseek V3.2技术报告解读》用于定位阅读要点。V3.2 是从 V3.1-Terminus 继续训练而来，其唯一架构改动是引入 DSA；不要把 V3 的所有原始训练配置直接当作 V3.2 的新贡献。

## TL;DR

V3.2 想解决的是开源模型在复杂任务中的三条短板：**长上下文注意力太贵、后训练算力投入不足、Agent 的泛化和指令遵循偏弱**。对应路线是：

```text
V3.1-Terminus（128K checkpoint）
  → DSA：Indexer 选 top-k KV，再做稀疏 MLA 注意力
  → continued pre-training + 专家蒸馏 + 混合 GRPO RL
  → 1,800+ 合成 Agent 环境 / 85,000 prompts
  → 复杂工具任务的泛化与长程推理
```

## Fast Look

**实体关系**：`MLA latent KV → Lightning Indexer 打分 → top-k Token Selection → 稀疏注意力 → continued pre-training → 领域专家蒸馏 → mixed GRPO RL → Agentic task synthesis → context management`。DSA 先降低长序列算力，RL 和合成环境再把省出的预算投入推理与工具能力。

#### DeepSeek Sparse Attention（DSA）

- **描述**：DSA 用轻量 Lightning Indexer 为每个 query 与历史 token 计算相关性分数，仅对 top-k 对应的 KV 条目执行主注意力；主注意力复杂度由 `O(L²)` 降为 `O(Lk)`。
- **与之前方法的区别**：V3.1-Terminus 使用稠密 MLA；DSA 不按固定窗口丢弃远处 token，而是按 query 内容动态选择全历史中的 token。V3.2 相比 V3.1 的唯一架构修改即是 DSA。
- **优点**：在长上下文中保留内容寻址能力，同时大幅缩减主注意力计算；官方在短序列 prefill 时改用 masked MHA 模拟 DSA，以避免稀疏机制的固定开销。
- **缺点/注意点**：Indexer 仍是 `O(L²)`，只是远比主 MLA 轻；top-k 选错会永久漏掉证据，因此不能理解为“长上下文无损”。

#### Lightning Indexer（闪电索引器）

- **描述**：用较少 head、较低维的 query/key 表示计算 `I(t,s)` 索引分数，并以 ReLU 作为激活；报告指出其可在 FP8 下高效实现。
- **与之前方法的区别**：它不是完整注意力的 QK 打分，而是先做廉价召回，为后续高成本注意力筛候选。
- **优点**：将“找哪些 token”与“精算这些 token 的注意力”拆开，适合长序列的粗排—精排式计算。
- **缺点/注意点**：它是 DSA 的召回瓶颈；低精度、低维与 ReLU 带来吞吐，但也可能降低索引质量。其 `O(L²)` 阶数并未消失。

#### Fine-grained Token Selection（细粒度 Token 选择）

- **描述**：对每个 query 在 Indexer 全部历史评分中取 top-k，只读取被选中 token 的 MLA latent KV，再计算最终注意力输出。
- **与之前方法的区别**：相对 Sliding Window Attention，滑窗固定保留附近 token；细粒度选择可命中很远的相关 token，选择粒度是 token 而不是连续块。
- **优点**：适合跨文档、跨段落的长程依赖；与大窗口相比，计算随 `k` 而非完整长度增长。
- **缺点/注意点**：top-k 是离散截断，未选 token 无法被主注意力纠正；选择、gather 和稀疏 kernel 的实现质量会显著影响实际加速。

#### MLA-MQA 模式下的 DSA

- **描述**：为沿用 V3.1-Terminus checkpoint，V3.2 在 MLA 上实例化 DSA；具体采用 MLA 的 MQA 模式，一个 latent KV 条目供该 token 的所有 query heads 共享。
- **与之前方法的区别**：相对 MLA 的 MHA 模式，MHA 可给不同 query head 更细的 KV 表示；MQA 共享 KV，使一个被选中的 latent 条目可被多头复用，利于 kernel 效率。
- **优点**：减少 KV/访存和重复 gather，更契合 DSA 的稀疏选取；可从既有 MLA 模型继续训练而非重建架构。
- **缺点/注意点**：KV 头共享限制了 head 间的表示自由度；必须严格匹配索引器与 MLA 的 RoPE layout，官方曾修复两者 interleaved / non-interleaved 不一致导致的性能问题。

#### Continued Pre-training（稀疏继续预训练）

- **描述**：从已扩展至 128K context 的 V3.1-Terminus checkpoint 出发，先做两阶段 continued pre-training，再进行同样采用 sparse attention 的后训练。
- **与之前方法的区别**：相对从零预训练，只把注意力模块稀疏化并继续适配，尽量继承原模型能力；而非为 DSA 重新训练一个基础模型。
- **优点**：较低成本验证新注意力结构；官方 parity 评测显示 V3.2-Exp 与 V3.1-Terminus 的短、长上下文质量总体相近。
- **缺点/注意点**：继承 checkpoint 也继承原先的能力边界；“基本持平”不表示每项任务都提升，迁移质量仍取决于继续训练数据与稀疏适配。

#### DeepSeek-V3.2-Exp（DSA 实验版）

- **描述**：V3.2-Exp 是正式 V3.2 前的实验发布：以 V3.1-Terminus 为基座引入 DSA，刻意对齐两者的后训练配置，用于隔离验证“稀疏注意力本身”是否在长上下文中节省成本且不明显伤害质量。官方仓库同时公开了推理 demo、Indexer logit kernel 与 sparse attention kernel 的对接线索。
- **与之前方法的区别**：相对 V3.1-Terminus，它只改变注意力为 DSA；相对正式 V3.2，它不承载后续扩大后训练计算、领域专家蒸馏和大规模 Agent 任务合成的全部最终能力主张，而是架构与效率的受控实验基线。
- **优点**：能把“DSA 带来的效率变化”与“后训练数据/RL 变化带来的能力变化”分开评估；官方基准显示其在多个推理与工具使用任务上总体与 V3.1-Terminus 接近，同时为服务端 sparse kernel 验证提供可复现入口。
- **缺点/注意点**：Exp 的能力持平不等于正式版能力，也不能直接拿其评测代替正式 V3.2；官方曾修复 Indexer RoPE 采用 non-interleaved layout、MLA RoPE 采用 interleaved layout 的实现不一致，复现实验必须使用修正后的代码。

#### Specialist Distillation（领域专家蒸馏）

- **描述**：从同一 V3.2 base 分别训练写作、通用问答，以及数学、编程、通用逻辑、通用 Agent、Agentic Coding、Agentic Search 等专家；专家以大规模 RL 训练后生成最终模型的领域数据，同时覆盖 thinking / non-thinking。
- **与之前方法的区别**：相对单一混合 SFT，先让不同目标独立达到较强策略，再把领域数据汇入最终模型；不同模式的数据生成器也分开处理长 CoT 和直接回答。
- **优点**：任务奖励、环境和推理长度可以按领域定制，缓解单模型多目标训练的相互干扰。
- **缺点/注意点**：专家能力会决定蒸馏上限；多教师数据配比和冲突处理困难，且报告没有把所有教师、数据规模与权重公开。

#### Mixed RL 与 GRPO（Group Relative Policy Optimization）

- **描述**：V3.2 仍以 GRPO 为 RL 算法，把 reasoning、Agent 与 human-alignment 合并进一个 RL 阶段；官方称后训练计算预算超过预训练成本的 10%。
- **与之前方法的区别**：相对多阶段串行 RL，不是先只训推理、再只训工具、最后训对齐的串行覆盖，而是混合任务共同更新，以减少阶段切换造成的遗忘。
- **优点**：组内相对奖励可减少对独立 critic 的依赖；混合训练同时维持多个能力面，且增加后训练算力可提升难题能力。
- **缺点/注意点**：算力扩大不自动等于泛化；GRPO 依然受奖励质量、组大小、KL 约束和 rollout 成本影响。报告中的“10%+”是计算预算比例，不能误读为数据比例或固定行业配方。

#### Agentic Task Synthesis Pipeline（大规模 Agent 任务合成）

- **描述**：先用 V3 系列式 cold start 将推理与工具调用统一为单条轨迹，再合成 1,800+ 交互环境和 85,000 个复杂 prompt，用于 Agentic RL。
- **与之前方法的区别**：相对只在代码/搜索环境 RL，目标不只是熟悉固定工具，而是构造更广泛的交互任务分布并训练可迁移的推理策略。
- **优点**：报告实验显示，仅使用合成通用 Agent 数据进行 RL，也能提升未见过的 Tau2Bench、MCP-Mark 和 MCP-Universe 任务，说明存在跨环境泛化潜力。
- **缺点/注意点**：合成环境的真实性、奖励漏洞和分布偏差会直接传递给策略；1,800+ 环境不等于覆盖真实生产工具链，必须做独立 held-out 评测。

#### Context Management（上下文管理）

- **描述**：V3.2 最大上下文为 128K；对超过上限的搜索 Agent 任务，报告使用上下文管理得到最终成绩。
- **与之前方法的区别**：相对无界保留 Agent 轨迹，主动压缩、筛选或重组织工具历史，以让长任务继续运行，而非因 token 超限终止。
- **优点**：将有限窗口用于最有价值的状态和证据；对工具轨迹尤为实用。
- **缺点/注意点**：压缩会丢失信息，也可能改变评测可比性；报告明确观察到冗余自检使轨迹膨胀并超过 128K，这仍是实际部署的主要风险。

#### DeepSeek-V3.2-Speciale

- **描述**：放宽输出长度约束、追求推理上限的高计算变体；官方报告给出其在 IMO、IOI、ICPC 等竞赛型评测上的成绩。
- **与之前方法的区别**：相对正式 V3.2，正式版在训练中施加更严格 token 约束，优化质量、延迟和成本的平衡；Speciale 以更长 reasoning token 换取更高难题表现。
- **优点**：展示扩大 test-time / post-training 计算对推理上限的收益，并提供能力上界参照。
- **缺点/注意点**：token 效率明显弱于 Gemini-3.0-Pro，且竞赛评测不等价于通用 Agent 可靠性；不能把 Speciale 的分数直接当成正式版的线上体验。

## 报告主线与面试回答

**一句话**：V3.2 不是重新设计整套 MoE 基座，而是在 V3.1-Terminus 上用 DSA 将长序列主注意力改为“轻量索引 + top-k 稀疏精算”，腾出预算后以专家蒸馏、混合 GRPO 和大规模 Agent 环境合成强化推理与工具泛化。

**面试常问：DSA 真正把 `O(L²)` 消掉了吗？**

没有完全消掉。最终主注意力从 `O(L²)` 降为 `O(Lk)`，但 Lightning Indexer 仍需要对 query—历史 token 打分，理论上仍是 `O(L²)`；它依靠更小的 head/维度、FP8 和高效 kernel 将常数压低。因而 DSA 是显著降低**主要注意力成本**，不是魔法式的严格线性复杂度。

**面试常问：为什么 DSA 与 Agent RL 要放在同一份报告里？**

长 Agent 轨迹同时受上下文、工具结果和反复推理约束。DSA 降低长序列训练/推理成本，才能使更多预算投入长 rollout；而合成环境、混合 RL 和 context management 决定这些省下来的 FLOPs 是否转化成真实工具任务成功率。

## 证据与边界

| 结论 | 证据状态 |
| --- | --- |
| V3.2 从 V3.1-Terminus 继续训练，唯一架构改动为 DSA | 官方报告明确披露 |
| V3.2-Exp 是对齐后训练配置的 DSA 验证版本，RoPE layout 修复影响推理复现 | 官方仓库明确披露 |
| DSA = Lightning Indexer + top-k 细粒度选择，主注意力为 `O(Lk)` | 官方报告明确披露 |
| 专家蒸馏、mixed GRPO、1,800+ 环境、85,000 prompts | 官方报告明确披露 |
| 后训练计算预算超过预训练成本的 10% | 官方报告明确披露 |
| 社区笔记称“复杂度变为线性级”“推理超 GPT-5” | 已直读，但已按报告修正为有条件表述；不能脱离 Indexer、模型变体与评测设置复述 |

## 参考资料

1. [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models（官方 arXiv）](https://arxiv.org/abs/2512.02556)
2. [DeepSeek-V3.2-Exp 官方仓库与推理实现](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
3. [小红书：Deepseek V3.2技术报告解读](https://www.xiaohongshu.com/explore/692de0ad000000000d0387a7?xsec_token=ABv7hlL7EKz5r-72AgdmnOjVbUcuRZ9Xc79t6jU_HUULI%3D&xsec_source=pc_search)
