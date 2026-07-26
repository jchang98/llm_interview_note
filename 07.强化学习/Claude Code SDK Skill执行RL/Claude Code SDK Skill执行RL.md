# 用 Claude Code SDK 做 Skill 执行 RL：从 Coding-Agent RL 改造

> **定位**：本页将 SLIME 的 coding-agent RL 设计改造成“Skill 执行 RL”方案。它不是修改代码后的现成 SLIME 功能，而是一份可实现的设计：把 SWE 的“提交 diff、运行测试”替换为“在沙箱中按 Skill 完成任务、由确定性检查器评分”。

## TL;DR

```text
任务 + 可用 Skills + workspace snapshot
  → 新建隔离 sandbox / Claude Code SDK harness
  → 模型多轮采样：选 Skill → 读取 SKILL.md → 调工具 → 产生工件
  → 规则/测试/审计型 grader 计算 reward
  → adapter 记录精确 token、logprob、loss_mask
  → trajectory tree 展开 child session 分支
  → PPO/GRPO 类更新，只优化模型采样 token
```

**从 coding 到 Skill RL 的本质变化**：原例的成功信号是干净 sandbox 中的 `git diff` 通过测试；Skill RL 的成功信号是任务工件、命令执行结果和 Skill 约束同时通过验证。仅“调用了 skill”不是奖励。

## Fast Look

**实体关系**：`Skill task JSONL → sandbox → Claude Code SDK/harness → Anthropic-compatible adapter → trajectory tree → Skill-aware grader → reward/advantage → PPO/GRPO update`。系统、Skill 文本、工具回显是条件 token；模型的文本、thinking（若启用）和 tool call 才是可优化 token。

#### Claude Code SDK / Harness

- **描述**：用 Claude Code SDK 或兼容 CLI 作为 agent harness：创建会话、输入任务、调用工具、处理完成/超时，并把模型请求路由到本地 adapter。
- **与之前方法的区别**：相对直接让 LLM 输出一次答案，SDK/harness 管理真实多轮工具交互；相对 SLIME 原例固定的 SWE coding harness，本方案把任务准备与完成判定替换为 Skill 任务契约。
- **优点**：可复用成熟的工具执行循环，真实暴露 Read/Edit/Grep/Bash/Agent 等 observation，而非离线伪造轨迹。
- **缺点/注意点**：SDK/CLI 版本、系统提示、工具 schema 会改变轨迹分布；必须锁定版本并记录 prompt snapshot，不能将供应商内部隐藏行为当成可复现事实。

#### Anthropic-compatible Adapter 与精确 Token 记录

- **描述**：adapter 把 harness 的 Messages 请求按被训模型 chat template 编码为 `input_ids`，向推理服务器请求采样结果和 logprob，并记录每轮 prompt/output token 与 loss mask。
- **与之前方法的区别**：相对只保存字符串 transcript，RL 需要知道“哪些 token 是当前策略实际采样的”，才能正确计算策略梯度。
- **优点**：模型可在 Anthropic 风格 harness 下运行，同时仍对开源模型的精确 token/logprob 做训练；工具 observation 保持不可训练。
- **缺点/注意点**：chat template 或 parser 与服务模型不一致会使 token 对不上，训练目标失效；必须在 rollout 前做 tool-call、reasoning parser 的端到端测试。

#### Skill-aware Task Contract

- **描述**：每个任务指定 workspace/image、用户 query、可用 Skills、允许工具、完成工件和 grader；Skill 包含 `SKILL.md`、相对目录和必要的 references/scripts。
- **与之前方法的区别**：相对 SWE 问题只要求修复 issue 并跑测试，Skill 任务还要求模型正确发现/加载适用 Skill，并遵循其路径和流程约束。
- **优点**：将知识型工作流转换成可交互、可验证的 RL 环境，能单独量化 Skill 选择和执行质量。
- **缺点/注意点**：若 Skill 内容直接预加载给所有任务，会绕过选择能力；若 grader 只检查最终文本，模型可跳过 Skill 而投机成功。

#### Sandbox 双阶段隔离

- **描述**：rollout sandbox 用于模型查看文件、调用工具和产生工件；grader sandbox 从原始快照重建，只拿允许提交的工件/patch 评分。
- **与之前方法的区别**：沿用 SLIME SWE 示例的“生成与评测分离”，但评分对象从 `git diff + test harness` 扩展为 Skill 产物、命令记录和约束检查。
- **优点**：降低模型在 rollout 中篡改测试、读取隐藏答案或污染后续样本的风险。
- **缺点/注意点**：复制 workspace、安装依赖和启动沙箱成本高；工件传递白名单必须严格，否则仍可能带入隐藏状态。

#### Skill-aware Reward（任务、过程、约束）

- **描述**：总奖励由任务完成、Skill 正确选择/加载、关键步骤证据、安全与格式约束组成；主奖励应来自可验证结果，过程项只作小权重 shaping。
- **与之前方法的区别**：相对 SWE 的单一测试通过奖励，Skill RL 明确区分“完成任务”与“遵循工作流”，避免只奖励表面调用。
- **优点**：可诊断失败发生在选 Skill、路径解析、工具执行还是最终工件；对稀疏终局奖励提供有限的中间信号。
- **缺点/注意点**：过强过程奖励会导致 reward hacking（无意义地调用 Skill/工具）；隐藏 grader 规则和对抗性评测不可缺少。

#### Trajectory Tree 与 Subagent 信用分配

- **描述**：一个主会话可派发多个 child session；以消息前缀分叉的轨迹树保存每个根到叶路径，并导出可训练的叶子 Sample。
- **与之前方法的区别**：相对把并行子代理拍平为单链，trajectory tree 保留子会话独立上下文和分叉，契合 agent 实际运行。
- **优点**：可训练委派与并行；可将子任务结果同父任务成功关联，并分析哪条分支贡献结果。
- **缺点/注意点**：终局 reward 如何分给父/子是估计问题，不是天然真值；初版宜只对完整 root-to-leaf 轨迹给同一终局 reward，再逐步加入可验证子任务奖励。

#### PPO / GRPO 与 KL 约束

- **描述**：对同一 Skill 任务采样多个 rollout，根据 reward 计算 advantage，使用 PPO clipped objective 或 GRPO 类相对优势更新，并以 reference policy 的 KL 抑制漂移。
- **与之前方法的区别**：相对 SFT 模仿已有正确轨迹，RL 能探索不同 Skill 选择、工具顺序和恢复策略，并从 sandbox grader 得到新反馈。
- **优点**：能直接优化端到端任务成功率；组内相对比较适合有多个 rollout 的可验证任务。
- **缺点/注意点**：on-policy rollout 昂贵；奖励噪声、超长轨迹与工具失败会造成高方差，须控制长度、超时、KL 和无效样本比例。

## 1. 与 SLIME coding-agent RL 的一一映射

SLIME 的示例让 Claude Code 在每个新 sandbox 中完成 SWE 任务，收集 `git diff`，再在第二个干净 sandbox 运行测试评分；它通过 adapter 保存 token/logprob，通过 trajectory tree 表示子代理/自动压缩分支，并以 custom generate hook 把 Sample 交回训练器。Skill RL 可复用这条骨架：

| SLIME coding-agent RL | Skill 执行 RL 改造 |
| --- | --- |
| `swe.py::prepare_workspace` | `skill_task.py::prepare_workspace`：复制 Skill bundle、创建工作目录、写入 query 与允许工具 |
| `SWE_PROMPT` | `SKILL_TASK_PROMPT`：完成用户目标；按需选择 Skill；遵守 Skill 的 base directory 与约束 |
| coding harness（Claude Code CLI） | Claude Code SDK/CLI harness；保持 session 生命周期与 adapter 接口 |
| `git_diff()` | `collect_artifacts()`：白名单收集输出文件、patch、结构化报告和必要日志 |
| `evaluate()` + 测试 | `grade_skill_task()`：干净 sandbox 运行 test/validator，并检查工件与 Skill 约束 |
| SWE JSONL | Skill task JSONL：query、image、workdir、available_skills、artifact spec、grader spec |
| diff pass/fail reward | 多项 reward：任务正确性为主、Skill 选择/遵循为辅 |

## 2. Skill 任务数据格式

```jsonc
{
  "prompt": "根据给定数据生成指定指标报告。",
  "label": "ifind-edb-00017",
  "metadata": {
    "image": "registry/skill-rl:2026-07",
    "workdir": "/workspace/task",
    "available_skills": [
      {
        "name": "ifind-edb",
        "base_dir": "/workspace/.opencode/skills/ifind-edb",
        "description": "查询宏观与行业时间序列。"
      }
    ],
    "skill_bundle": "bundles/ifind-edb.tar.zst",
    "allowed_tools": ["Read", "Glob", "Grep", "Bash", "Write", "Skill"],
    "artifact_spec": {
      "allow": ["/workspace/task/result.json", "/workspace/task/report.md"],
      "deny": [".git", ".env", "/workspace/.opencode/skills/**/secrets/**"]
    },
    "grader": {
      "type": "command",
      "eval_cmd": "python /grader/check.py /submission"
    },
    "time_budget_sec": 900
  }
}
```

`available_skills` 只列名称和描述，适用于 non-slash 环境；完整 `SKILL.md` 只能由模型调用 `Skill` 后作为 tool result 返回。slash 环境可在任务 metadata 中设置 `forced_skill`，并由 harness 按该模式显式注入内容。

## 3. Rollout 与 Reward 设计

### 3.1 单样本 rollout

1. 从干净 image 启动 rollout sandbox，并解压对应 Skill bundle 到固定的虚拟路径。
2. 启动 Claude Code SDK/harness，设置目标系统提示、允许工具、时间和 token 预算。
3. adapter 将每一轮模型输入渲染给 SGLang/vLLM；记录 `prompt_ids`、`output_ids`、logprob、`loss_mask` 与 tool invocation。
4. 对调用的 Skill、文件操作、命令、child session 和最终工件写审计事件。
5. 超时/越权/解析失败时结束 rollout，返回可区分的失败标签，而不是伪造 tool result。
6. 把 artifact 白名单复制到全新的 grader sandbox；在其中执行隐藏验证器。

### 3.2 奖励函数（推荐起点）

```text
R = 1.00 * task_pass
  + 0.10 * skill_selection_valid
  + 0.10 * skill_constraints_pass
  - 0.05 * invalid_tool_call_count
  - 0.10 * policy_violation
  - 0.02 * normalized_excess_steps
```

- `task_pass` 必须由隐藏测试、结构化结果比对或独立验证命令得到，是主信号。
- `skill_selection_valid` 只能奖励“需要 Skill 时正确选择”；任务不需要 Skill 时，调用并不加分。
- `skill_constraints_pass` 只检查可观察的路径、命令/工件或要求步骤，不能奖励模型声称“我读过了”。
- 步数惩罚要很弱，并设任务依赖的最少合理步数，避免模型为了短而跳过验证。

## 4. 训练实现要点

### 4.1 Custom generate 与 Sample 导出

沿用 SLIME 的 `--custom-generate-function-path`：`generate_skill_task()` 负责 boot sandbox、启动 harness、收集 token segment、调用 grader，并导出包含 reward/metadata 的一个或多个 Sample。模型服务的 tool-call parser 和 reasoning parser 必须与被训 checkpoint 匹配；否则 harness 虽收到文本，训练器却无法恢复正确 action token。

### 4.2 Token mask

`loss_mask=1`：模型实际产生的 assistant 文本、thinking（若明确训练）与 tool-use token。\n\n`loss_mask=0`：system、user、Skill 列表、`SKILL.md` 内容、tool result、文件内容、grader 输出、模板标记。\n\n原则是：**不优化环境替模型说的话**。特别是 Skill 内容很长，若误训练它，loss 会虚低而 agent 能力不升。

### 4.3 课程学习

| 阶段 | 任务 | 通过门槛 |
| --- | --- | --- |
| A | 单 Skill、只读、确定性小任务 | tool parse 与 skill selection 稳定 |
| B | 单 Skill、读写工件、隐藏 validator | 端到端 pass 达到 SFT 基线 |
| C | 多 Skill 判别、失败恢复、长路径 | 不相关 Skill 的误调用下降 |
| D | child session、多工件任务 | 子代理不泄漏上下文，主任务成功率提升 |

先用 SFT 初始化，后做 RL。没有一个已具备基本工具语法与 Skill 选择能力的 SFT policy，直接做稀疏终局 RL 的样本效率通常很差。

## 5. 评测与常见失败

| 失败模式 | 症状 | 修复方向 |
| --- | --- | --- |
| Skill 作弊 | 模型不加载 Skill 也通过 | 加入只有遵循 Skill 才能通过的隐藏约束；过程奖励不替代终局测试 |
| 工具回显被训练 | 模型生成虚假的文件内容/命令输出 | 检查 token-level mask，所有 observation 必须为 0 |
| 沙箱污染 | 同一任务第二次异常变简单 | 每 rollout 和每 grader 都从干净快照启动 |
| 路径过拟合 | 模型背诵 `/Users/...` | 统一为 sandbox 虚拟路径，并做跨路径评测 |
| 子代理信息泄漏 | parent 直接“知道”child 私有文件 | 只将 harness 真实返回的 child summary 传回 parent |
| 奖励投机 | 无意义反复调用 Skill | 降低过程项权重，加入工具预算与隐藏反例 |

## 参考资料

1. [THUDM/SLIME：Coding-Agent RL 示例](https://github.com/THUDM/slime/tree/main/examples/coding_agent_rl)
2. [SLIME 示例 README：sandbox、adapter、trajectory tree、custom generate 与 token contract](https://raw.githubusercontent.com/THUDM/slime/main/examples/coding_agent_rl/README.md)
3. [Teich：Agent trace 标准化与 SFT masking](https://github.com/TeichAI/teich)
