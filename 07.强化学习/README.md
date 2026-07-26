# 07.强化学习

## Fast Look

**对齐链路**：`SFT policy → 偏好对 → RM 或直接偏好目标 → rollout → reward/advantage → PG/PPO 更新`。核心差别是：PPO 显式做在线 RL，DPO 将偏好优化化为监督式目标。

#### MDP、状态、动作、奖励、回报与价值函数

- **描述**：MDP 用状态 `s`、动作 `a`、转移、奖励 `r` 和折扣 `γ` 描述决策；回报是未来奖励和，价值函数估计期望回报。
- **与监督学习的区别**：监督学习有固定标签；RL 的动作会改变后续数据分布，奖励可能延迟或稀疏。
- **优点**：能优化多步、长期目标与交互式任务。
- **缺点/注意点**：探索、信用分配和高方差使训练难；LLM 场景尤其依赖可靠奖励。

#### REINFORCE / Policy Gradient（PG）

- **描述**：按 `∇ log π(a|s) · return` 增大高回报动作概率；常加 baseline 降方差。
- **与价值迭代/DQN 的区别**：直接优化随机策略，不必显式最大化离散动作的 Q 值。
- **优点**：适用于连续/巨大动作空间，天然匹配 token 概率策略。
- **缺点/注意点**：Monte Carlo 回报方差高、样本效率低；需要 advantage、critic 或更稳定约束。

#### Actor–Critic、Advantage 与 GAE

- **描述**：Actor 输出策略，Critic 估计价值；Advantage 衡量动作相对基线的好坏，GAE 在偏差与方差间折中。
- **与纯 REINFORCE 的区别**：用可学习 Critic 替代纯采样回报基线。
- **优点**：显著降低方差，支撑 PPO 等现代算法。
- **缺点/注意点**：Critic 估计偏差会误导 Actor；价值损失和归一化需要稳定训练。

#### TRPO 与 PPO

- **TRPO 描述**：在 KL 信赖域约束下更新策略；**PPO 描述**：用 clipped ratio 或 KL penalty 近似限制新旧策略差异。
- **与普通 PG 的区别**：不允许策略一步偏离过大，避免性能崩塌。
- **优点**：PPO 实现比 TRPO 简单、稳定性较好，是 RLHF 常用基线。
- **缺点/注意点**：仍是 on-policy，rollout 成本高；clip range、KL 系数、batch/epoch 不当会造成 reward hacking 或训练停滞。

#### RLHF：SFT、Reward Model、Actor、Reference 与 Critic

- **SFT model**：提供初始可用策略；**Reward Model**：从 chosen/rejected 偏好对学习标量奖励。
- **Actor**：当前被优化策略；**Reference**：冻结参考策略，用 KL 约束防止漂移；**Critic**：估计 value/advantage。
- **与单阶段 SFT 的区别**：RLHF 不只模仿答案，还根据相对偏好和在线生成持续优化。
- **优点**：可将帮助性、安全性和风格等人类偏好纳入目标。
- **缺点/注意点**：至少涉及多模型与多阶段数据，成本高；RM 偏差会被策略放大。

#### DPO（Direct Preference Optimization）

- **描述**：用偏好对和参考模型直接优化 chosen 相对 rejected 的对数概率差，无需显式训练 RM 或运行 PPO rollout。
- **与 PPO-RLHF 的区别**：把奖励最大化与 KL 正则的最优策略关系改写为监督式损失。
- **优点**：训练链路短、稳定、资源需求低，适合离线偏好数据。
- **缺点/注意点**：依赖固定偏好数据，缺少在线探索；`β`、参考模型与数据噪声会强烈影响结果。

#### DQN、DDPG、模仿学习与稀疏奖励

- **DQN**：学习离散动作 Q 值；**优点**是可用经验回放，**缺点**是不适合 LLM 巨大 token 动作空间。
- **DDPG**：确定性 Actor–Critic，面向连续动作；**优点**是样本效率，**缺点**是稳定性敏感。
- **模仿学习**：直接学习专家行为；**相对 RL** 无需奖励探索，**缺点**是分布偏移与专家数据依赖。
- **稀疏奖励**：只在终点给反馈；需要奖励塑形、过程监督或可验证环境，否则信用分配困难。

#### Claude Code SDK Skill 执行 RL：Adapter、Sandbox 与 Skill-aware Reward

- **描述**：复用 coding-agent RL 的 Claude Code SDK/harness、Anthropic-compatible adapter、trajectory tree 与 custom generate，把 SWE 的 `git diff + 测试` 改为 Skill bundle、任务工件和隐藏 validator；用任务正确性为主、Skill 选择/约束为辅的 reward 做 PPO/GRPO 类优化。
- **与之前方法的区别**：相对离线 SFT 模仿轨迹，RL 在干净 sandbox 中在线探索 Skill 选择、工具顺序与失败恢复；相对一般 RLHF，reward 来自可执行环境而非偏好模型。
- **优点**：直接优化 OpenCode/Claude Code 风格 agent 的端到端任务成功率；双沙箱隔离可降低测试作弊，token-level mask 仅优化模型实际采样 token。详见 [Claude Code SDK Skill执行RL](./Claude%20Code%20SDK%20Skill执行RL/Claude%20Code%20SDK%20Skill执行RL.md)。
- **缺点/注意点**：这是基于 SLIME coding-agent 示例的改造设计，不是现成内置功能；on-policy rollout、沙箱与 grader 成本高，过程奖励过强会导致无意义调用 Skill 的 reward hacking。

### 7.1 强化学习原理

[策略梯度（pg）](/07.强化学习/策略梯度（pg）/策略梯度（pg）.md "策略梯度（pg）")

[近端策略优化(ppo)](/07.强化学习/近端策略优化(ppo)/近端策略优化(ppo).md "近端策略优化(ppo)")

### 7.2 RLHF

[大模型RLHF：PPO原理与源码解读](/07.强化学习/大模型RLHF：PPO原理与源码解读/大模型RLHF：PPO原理与源码解读.md "大模型RLHF：PPO原理与源码解读")

[DPO](/07.强化学习/DPO/DPO.md "DPO")

[Claude Code SDK Skill执行RL](./Claude%20Code%20SDK%20Skill执行RL/Claude%20Code%20SDK%20Skill执行RL.md "Claude Code SDK Skill执行RL")

### 7.3 一些题目

[1.rlhf相关](/07.强化学习/1.rlhf相关/1.rlhf相关.md "1.rlhf相关")

[2.强化学习](/07.强化学习/2.强化学习/2.强化学习.md "2.强化学习")
