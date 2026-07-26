# OPD（On-Policy Distillation）：在策略蒸馏、MOPD 与 RL 实现

> **资料边界**：OPD 是一类训练范式，不是一份固定算法。本文以 [MiMo-V2-Flash 技术报告](https://arxiv.org/abs/2601.02780)、[GLM-5 技术报告](https://arxiv.org/abs/2602.15763)、[DeepSeek-V4 报告](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/DeepSeek_V4.pdf) 和 Qwen3 技术报告为主要线索；用户提供的两篇小红书笔记已用 `xhs-cli` 直读，用于补充“RL 框架实现”的阅读路径。不同报告中 `OPD`、`MOPD`、`OPCSD` 的目标和教师构造并不相同，不能将它们混为一个公式。

## TL;DR

离线蒸馏让学生模仿教师预先生成的轨迹；但部署时模型走的是自己的轨迹，前缀分布会发生偏移。**OPD** 改为：

```text
student 在自身策略 πθ 下 rollout 前缀 y<t
    ↓
teacher 在同一前缀上给 token 分布 / dense 信号
    ↓
最小化学生与教师在 student-induced distribution 上的差异
    ↓
以 RL 框架实现时，将“教师—学生 token 差异”转为 token reward / advantage
```

核心不是“学生复制教师答案”，而是**在学生会实际到达的状态上，让教师纠正下一 token 分布**。

## Fast Look

**实体关系**：`student rollout → teacher 在同前缀评分 → per-token KL / log-ratio 信号 → token reward / advantage → PPO/GRPO-style update`。多领域时为 `domain teachers → MOPD → unified student`；多阶段训练时为 `earlier checkpoint → OPCSD → 恢复被遗忘能力`。所有变体都需处理教师可访问性、tokenizer 对齐、rollout 成本与 KL 方向。

#### OPD（On-Policy Distillation，在策略蒸馏）

- **描述**：学生从当前策略生成 rollout，教师在这些学生前缀上提供 token-level 分布或监督；训练目标在学生自身诱导的状态分布上拉近师生策略。
- **与之前方法的区别**：离线/行为克隆蒸馏通常对教师固定轨迹做 SFT，训练前缀来自 teacher；OPD 的前缀来自 student，因此更接近部署时会看到的错误、分支和上下文。
- **优点**：缓解 exposure bias / distribution mismatch；教师可在学生的真实错误状态给稠密 token 级纠正，特别适合长 CoT、工具轨迹和推理任务。
- **缺点/注意点**：每次 rollout 都需调用或运行教师，计算与通信昂贵；student 偏离教师支持集过远时，教师分布可能不再提供可学习信号，且需要可对齐的 tokenization 或映射方案。

#### Student-induced Distribution（学生诱导分布）

- **描述**：记学生策略为 `πθ`，从 prompt `x` 采样 `y ~ πθ(·|x)`；OPD 的期望和 teacher 评分都条件于学生已经生成的前缀 `y<t`。
- **与之前方法的区别**：teacher-forcing 蒸馏在教师前缀上计算损失；OPD 将训练上下文切换为 student prefix，直接面对模型自己的累积误差。
- **优点**：训练—推理分布更一致；可学习“教师轨迹中从未出现、但学生经常进入”的恢复动作。
- **缺点/注意点**：早期 student rollout 质量差会浪费教师查询；长序列还会使有效前缀迅速偏离，因而常需 warm start、长度控制、选择性 token 或质量过滤。

#### Reverse KL 与 Forward KL

- **描述**：在 student rollout 上常写作最小化 `D_KL(πθ || πT)`：对 student 采样 token，比较 `log πθ(a|s) - log πT(a|s)`；`πT` 为教师策略。Forward KL `D_KL(πT || πθ)` 则以教师分布/轨迹为期望。
- **与之前方法的区别**：传统 SFT 或教师轨迹蒸馏更自然接近 forward-KL / maximum likelihood；OPD 为匹配 student-generated state，通常采用可由 student 样本估计的 reverse KL，具体方向仍以每篇报告定义为准。
- **优点**：reverse KL 可直接复用 student rollout，并使蒸馏信号对部署分布对齐；全词表 logits 可保留教师对替代 token 的相对偏好。
- **缺点/注意点**：reverse KL 可能更偏 mode-seeking，过度压低 student 的探索；只用 sampled token 的近似会丢失全词表信息，数值稳定性和 teacher logprob 的可得性都是工程约束。

#### Token-level Distillation Reward / Advantage

- **描述**：若教师能给 `log πT(a_t|s_t)`，可构造 token 信号，例如与 `log πθ(a_t|s_t)` 的差或 KL 项；再经归一化、折扣/GAE 或奖励整形，作为策略梯度中的 advantage。小红书实现笔记指出，可在 GRPO/VeRL 管线中替换原本优势项来落地这一思路。
- **与之前方法的区别**：普通 RL 的奖励常来自终局 verifier、reward model 或环境；这里教师每个 token 的分布差提供 dense learning signal，RL 框架主要承担 rollout、掩码、优势估计和稳定更新。
- **优点**：不必等待终局 reward，长轨迹中 credit assignment 更细；便于复用成熟 PPO/GRPO 基础设施，而不是另写一套蒸馏训练循环。
- **缺点/注意点**：这是一种实现等价/近似，不意味着“OPD 就是 GRPO”；reward 的符号、基线、KL 系数和 stop/tool token mask 必须与目标函数一致，否则会优化到相反方向或引入偏差。

#### Full-vocabulary Logit Distillation（全词表 Logit 蒸馏）

- **描述**：在每个 student prefix 上获取教师完整 vocabulary logits，并计算分布级 KL，而不只读取已采样 token 的 logprob。DeepSeek-V4 将其用于多教师 OPD。
- **与之前方法的区别**：hard-label 或 sampled-token 蒸馏只告诉学生“已选 token”的好坏；全词表信号还保留教师对同义、替代推理步骤和不确定度的排序。
- **优点**：梯度更平滑、信息密度高，能更忠实传递专家策略；对非唯一正确的开放式输出更有价值。
- **缺点/注意点**：需要传输/存储巨大词表分布，teacher 推理和通信成本显著；不同词表不能直接逐项 KL，需要共用 tokenizer、投影或退化为 token-level 信号。

#### MOPD（Multi-Teacher On-Policy Distillation，多教师 OPD）

- **描述**：先为数学、代码、搜索、Agent 等领域训练专家教师；统一 student 在自身 rollout 上接收各领域教师的 dense token-level 信号，以整合专长。MiMo-V2-Flash 将该范式命名为 MOPD；DeepSeek-V4 也采用多教师 OPD。
- **与之前方法的区别**：单教师 OPD 将 student 拉向一个通用教师；MOPD 面对多位、可能互相冲突的专家，须决定任务路由、教师权重或损失加权。
- **优点**：把“培养专精能力”和“合并到统一模型”解耦；可用强领域 RL 教师提升 student，又避免每个能力都依赖同一通用 reward。
- **缺点/注意点**：教师间的分布冲突会造成 see-saw / 负迁移；教师选择、权重 `w_i`、数据配比和调用调度本身就是核心算法，不能只把多个 KL 简单相加。

#### Teacher Routing 与权重 `w_i`

- **描述**：多教师 OPD 通常按任务域、样本标签或教师置信度选择教师，并在目标中加入权重，例如 `L = Σ_i w_i · D_KL(πθ || πT_i)`；`w_i` 可随任务、token 或训练阶段变化。
- **与之前方法的区别**：混合 SFT 往往仅按数据比例隐式决定各任务权重；MOPD 显式控制“当前 student 更接近哪位教师”。
- **优点**：可把数学、代码、Agent 等异质能力按需要融合，并在冲突时保留可诊断的调度接口。
- **缺点/注意点**：错误路由会给出不相关甚至相矛盾的目标；固定权重容易让高频/高熵领域压制其他能力，需要独立的 domain eval 和权重消融。

#### OPCSD（On-Policy Cross-Stage Distillation，跨阶段在策略蒸馏）

- **描述**：GLM-5 的多阶段后训练在后续 RL 阶段后，用较早阶段 checkpoint 作为教师，在当前 student rollout 上进行蒸馏，以恢复先前阶段被覆盖的推理或 Agent 能力。
- **与之前方法的区别**：MOPD 横向融合不同领域教师；OPCSD 纵向连接同一模型谱系的不同训练阶段，目标是抗灾难性遗忘而非引入外部专家知识。
- **优点**：可保留早期 Reasoning RL 或 Agentic RL 的能力，同时继续进行 General RL；教师和学生往往同 tokenizer、同架构，logit 对齐更直接。
- **缺点/注意点**：早期 checkpoint 本身可能带有缺陷，过强蒸馏会阻止后阶段适应；需要权衡“恢复旧能力”与“允许新阶段突破”。

#### Qwen 式强—弱 OPD 与 Self-OPD

- **描述**：强—弱 OPD 用更大教师在小学生的自身 rollout 上提供监督；Self-OPD / OPSD 则让同一模型在不同条件信息下分饰教师与学生，例如教师可见经验证的特权 reasoning trace、学生只见问题。
- **与之前方法的区别**：前者的主要目的通常是能力迁移/压缩，后者减少独立大教师依赖；二者都不同于只用答案做 SFT，因为监督仍发生在 student-generated prefix 上。
- **优点**：强—弱模式可用较小模型接近大模型推理策略；self 模式降低教师部署成本，并可利用已有的正确轨迹作为特权上下文。
- **缺点/注意点**：强—弱模式受 tokenizer/容量鸿沟限制；self 模式依赖足够强的自身教师视角，若特权信息质量差可能强化错误。不同 Qwen 版本的具体配方应以其技术报告为准。

#### Selective / Teachability-Aware OPD

- **描述**：并非每个师生分歧 token 都值得蒸馏。Teachability-Aware OPD 以局部支持集是否重叠等指标选择“学生有可能学会”的 token 位置，而不是对全部 token 或仅高 KL token 施加损失。
- **与之前方法的区别**：全 token OPD 默认所有 KL 分歧同等重要；高熵/高分歧筛选又可能选到教师质量虽高、但学生容量/上下文无法兼容的位置。
- **优点**：减少无效 teacher 查询和梯度噪声；研究结果显示在 Qwen teacher-student 设置下，少量选中 token 也可优于全 token OPD。
- **缺点/注意点**：需要额外诊断或选择机制，可能错过长尾纠正信号；该方法是研究型改进，不能默认已被各家生产 OPD 管线采用。

## 与 SFT、RLHF / GRPO 的关系

| 方法 | 轨迹从哪里来 | 主要监督 | 解决什么问题 | 典型风险 |
| --- | --- | --- | --- | --- |
| Teacher SFT / 离线蒸馏 | 教师预先生成 | 硬标签或 forward-KL | 快速模仿教师行为 | 部署时 student prefix 偏移 |
| RLHF / GRPO | student rollout | 标量 reward / 组内相对 reward | 优化可验证结果或偏好 | 奖励稀疏、credit assignment 粗、rollout 昂贵 |
| OPD | student rollout | 教师 token 分布、KL 或 token reward | 对 student 的实际轨迹给稠密纠正 | 教师成本、tokenizer/支持集不匹配 |
| MOPD | student rollout | 多领域教师 token 信号 | 整合专家能力 | 教师冲突与权重调度 |
| OPCSD | 当前 student rollout | 早期 checkpoint 分布 | 缓解多阶段 RL 遗忘 | 约束过强会阻碍新能力 |

## 在 RL 框架中实现：最小心智模型

```text
1. 对 prompt 用当前 student 采样 action / CoT / tool trajectory。
2. 只在需要学习的 assistant action token 上对齐 mask；不要把环境 observation 当作 policy action。
3. 将同一前缀送给 teacher，取得 logits 或 action-token logprob。
4. 计算 distribution-level KL，或构造 token-level distillation reward。
5. 经过 value baseline / GAE / normalization 得到 advantage，再用 PPO/GRPO 风格稳定更新。
6. 多教师时做 task routing、权重归一化与冲突监控；持续做领域回归评测。
```

**常见错误**：

- 把 teacher rollout 当成 student rollout，退化回离线蒸馏；
- 对 user、tool observation 或 padding token 计算 OPD loss；
- 没有冻结/版本化 teacher，导致目标分布在一次训练中漂移；
- 只报告最终 reward，不检查 KL、教师置信度、token 覆盖与各领域能力是否被牺牲。

## 小红书检索对应与参考资料

- [On-Policy Distillation 使用 RL 框架实现？](https://www.xiaohongshu.com/explore/6a61be16000000001002b12d?xsec_token=CBl5xlBKMseDHsmvS4J3jWoplHbv14zRdbgP1xUK1RHAQ%3D&xsec_source=app_share)：已直读；提到 MiMo-V2-Flash MOPD、GLM-5，并讨论以替换 GRPO advantage 的方式在 VeRL/RL 框架实现 OPD。实现细节以各论文目标函数为准。
- [Qwen/GLM/DeepSeek 等基座模型 OPD 技术总结](https://www.xiaohongshu.com/explore/6a4ef71b0000000021022f37?xsec_token=CBD_Yan_aWaJxRsGcrygBokG1SBlIat3jTM6paiwwu5YE%3D&xsec_source=app_share)：已直读；作为多家模型采用 OPD/跨阶段蒸馏的导航材料，不将图片中的未交叉核验结论直接当作事实。
- [MiMo-V2-Flash Technical Report（MOPD）](https://arxiv.org/abs/2601.02780)
- [GLM-5 技术报告（OPCSD）](https://arxiv.org/abs/2602.15763)
- [DeepSeek-V4 技术报告（多教师 OPD）](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/DeepSeek_V4.pdf)
- [Qwen3 技术报告](https://arxiv.org/abs/2505.09388)
- [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)
- [Teachability-Aware OPD](https://arxiv.org/abs/2605.26844)
