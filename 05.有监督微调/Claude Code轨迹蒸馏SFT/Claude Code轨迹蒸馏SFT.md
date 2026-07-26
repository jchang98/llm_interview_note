# Claude Code 轨迹蒸馏为 SFT：适配 OpenCode 的数据规范

> **目标**：把 Claude Code（CC）执行任务时产生的多轮轨迹蒸馏给开源模型，使学生模型在 **OpenCode harness** 中也能正确调用工具、加载 Skill、处理 slash command 和派发 subagent。这里的“格式适配”不是简单替换工具名，而是将同一个任务重新表达为**目标 harness 在运行时真正会看到的消息与工具观察**。

## TL;DR

```text
CC 原始轨迹
  → 脱敏、去失败/重复、保留原始 provenance
  → 按 slash / non-slash / subagent 分桶
  → 工具 schema + tool result 映射
  → 把 Skill 注入位置改成 OpenCode 运行时语义
  → 目标模型 chat template 渲染 + assistant/tool response-only mask
  → 4B / 9B / 30B 训练与 OpenCode 在线回放评测
```

**核心结论**：模型学习的是“在某个 harness 下，看到什么上下文后该调用什么工具”。因此 CC 格式直接拿来训 OpenCode，最容易在 Skill、slash 和 subagent 三类样本上出现上下文错位。

## Fast Look

**实体关系**：`原始 CC session → Teich 标准化消息 → 数据桶（slash / non-slash / subagent）→ 工具映射 → OpenCode prompt 重放 → chat template → response-only SFT mask → OpenCode 回放评测`。

#### Teich：轨迹标准化与训练审计

- **描述**：Teich 可从 Claude 等本地 session 提取轨迹，保留 `messages`、`tools`、元数据与来源，并在 tokenizer chat template 渲染后记录监督 span、执行 response-only mask 和数据审计。
- **与之前方法的区别**：相对将日志拼成一段纯文本，Teich 在 tokenization 前保持 tool schema、tool result、reasoning 边界和 provenance 的结构化表示。
- **优点**：便于把 CC 原始格式与 OpenCode 目标格式分离；可统计超长、截断、损坏、完全被 mask 的样本，降低“能训但标签错位”的风险。
- **缺点/注意点**：Teich 只解决规范化和审计，不会自动证明两种 harness 的语义等价；转换规则仍须针对目标 OpenCode 版本回放验证。

#### Harness 语义归一化（Harness Normalization）

- **描述**：将轨迹拆成“系统指令、用户任务、工具 schema、assistant tool call、tool observation、子会话关系”六类事件，再按 OpenCode 的运行时协议重新渲染。
- **与之前方法的区别**：相对工具名的一对一替换，归一化先保留事件语义，再决定 system/user/tool 中的落位；尤其处理 Skill 内容注入和子代理边界。
- **优点**：同一 canonical trace 可派生 CC、OpenCode 或其他 agent 的训练格式，数据来源和目标格式可追溯。
- **缺点/注意点**：不可映射的工具、隐式系统提示和 CLI 内部行为必须显式标记；不能伪造一个目标 harness 实际不会提供的 observation。

#### Slash 数据（Slash-command Trace）

- **描述**：用户通过 `/skill-name args` 等 slash command 显式触发 Skill；训练样本应让模型学到该命令对应的 Skill 内容、基目录和相对路径约束如何进入上下文。
- **与之前方法的区别**：相对普通任务，Skill 的选择不是模型先调用 `skill` 工具的结果，而是由用户 slash command 直接指定；CC 与 OpenCode 的注入位置不同。
- **优点**：覆盖用户明确指定工作流的高价值场景；训练后可降低“明明点名 Skill 却忽略其说明”的概率。
- **缺点/注意点**：slash 数据不能伪装成 non-slash 数据；要删除机器特定绝对路径或替换为 sandbox path，防止模型背诵个人目录。

#### Non-slash 数据（模型自主加载 Skill）

- **描述**：用户只提出任务，模型从可用 Skill 列表判断匹配项并调用 `Skill/skill`，随后根据 tool result 中返回的 `SKILL.md` 和关联文件继续执行。
- **与之前方法的区别**：相对 slash 数据，Skill 不应预加载到用户消息；它应由模型的一次工具调用触发，且 Skill 内容以 tool observation 注入。
- **优点**：训练模型完成“任务识别 → Skill 选择 → 读取指令 → 执行”的完整闭环，更贴近真实 agent 行为。
- **缺点/注意点**：将 Skill 内容提前塞进 system/user 会泄漏 oracle，模型可能只会复读而学不会选择；需保留未调用 Skill 的负例或相近 Skill 的区分样本。

#### Subagent 数据（子会话与上下文隔离）

- **描述**：主代理把子任务交给 child session；子会话有自己的 system、用户任务、工具轨迹与结束结果，主会话只接收被 harness 回传的摘要/结果。
- **与之前方法的区别**：相对将所有消息拍平到单一对话，subagent 保留 parent/child session ID、分支关系和消息可见性；CC 可能预加载 slash Skill，OpenCode 子会话则按需加载。
- **优点**：模型能学习委派、并行和结果整合；保留分支后可避免把子代理私有上下文错误暴露给主代理。
- **缺点/注意点**：不能把 child 的完整 tool 轨迹直接拼给 parent 训练；否则训练分布与实际运行时不一致，并造成 token 成本和信息泄漏。

#### 工具 Schema 映射（Tool Mapping）

- **描述**：为每个 CC 工具定义目标 OpenCode 工具、参数转换、输出转换和不可映射策略；工具 schema 与调用参数应共同转换。
- **与之前方法的区别**：相对只把 `Grep` 字符串改为 `grep`，该映射同时约束参数名、路径范围、返回格式和副作用语义。
- **优点**：让 SFT 直接监督目标模型会生成的 tool-call 形状，减少 parser 错误与无效调用。
- **缺点/注意点**：`Task`、`TodoWrite`、`Skill` 的运行时语义通常没有完全一一对应；应保留 `unsupported` 标签并过滤或重写，而非静默猜测。

#### Response-only Mask 与可训练 Token

- **描述**：仅对学生模型生成的 assistant 文本、thinking（若策略允许）和 tool call token 计算 SFT loss；system、user、工具 schema、tool result 和注入的 Skill 内容只作条件上下文。
- **与之前方法的区别**：相对整段交叉熵，agent 轨迹有大量环境 observation；训练 observation 会使模型学习伪造工具回显。
- **优点**：监督目标与在线 rollout 的可控 token 一致，减少模型把“工具返回”当作应生成内容的风险。
- **缺点/注意点**：不同 tokenizer 的 chat template 会改变边界；必须 token 级检查 mask，而不能凭字符串角色标签假定正确。

## 1. 数据集目标与模型训练

### 1.1 训练目标

训练目标不是“复刻 Claude Code 的内部提示词”，而是：在 OpenCode 给定的工具、Skill 列表与工作目录下，让学生模型产生可解析的下一步 assistant 行为。每条记录至少应包含：

```json
{
  "trace_id": "stable-id",
  "source_harness": "claude-code",
  "target_harness": "opencode",
  "bucket": "slash | non_slash | subagent",
  "messages": [],
  "tools": [],
  "metadata": {
    "parent_trace_id": null,
    "skill_names": ["ifind-edb"],
    "tool_mapping_version": "v1",
    "conversion_notes": []
  }
}
```

保留原始 CC JSONL 作为不可变证据；转换后的 OpenCode JSONL 单独版本化。任何清洗、路径替换、工具降级、截断都写入 `conversion_notes`。

### 1.2 模型尺寸：4B、9B、30B

| 尺寸 | 推荐定位 | 数据策略 | 主要风险 |
| --- | --- | --- | --- |
| 4B | 工具格式、少步 Skill 执行、单文件任务 | 高质量短轨迹优先；限制工具集合与上下文 | 容量不足时容易遗漏多步状态或长 Skill 约束 |
| 9B | 主力迭代模型，多工具与中等长度任务 | 三桶均衡；加入路径、参数、失败恢复样本 | 若 long trace 占比过高，训练会被上下文截断主导 |
| 30B | 复杂多文件、长 Skill、subagent 编排 | 保留长轨迹和多轮回放；关注思考/工具比例 | 成本与吞吐压力大；不能只看离线 loss，应做真实 harness 成功率 |

`Qwen3.6-35B-A3B` 可作为较大 MoE 学生/基线来验证 agent 轨迹格式，但本页不把未经实验的效果写成结论。若用户所说的“30B-A3B”是模型名，需要先核对精确 checkpoint：`Qwen3-30B-A3B` 与 `Qwen3.6-35B-A3B` 是不同模型。

## 2. 先建立 canonical trace，再渲染目标 harness

### 2.1 推荐转换流程

1. 用 Teich 提取 CC session，先脱敏密钥、用户名、真实仓库 URL 和绝对路径。
2. 解析出 canonical event：`system`、`user`、`assistant_text`、`tool_call`、`tool_result`、`child_start`、`child_end`。
3. 按触发机制标记为 `slash`、`non_slash` 或 `subagent`；一个 parent trace 可引用多个 child trace。
4. 对每个工具执行映射：改 schema、参数与 observation；若语义不保真，标为 `unsupported`，不作为目标调用标签。
5. 将 canonical trace 渲染为 OpenCode 的 system/user/tool 顺序；最后用目标 tokenizer chat template tokenization。
6. 只对 assistant 的文本与 tool call 放开 loss mask；审计每条样本的有效监督 token、长度、映射版本和回放结果。

### 2.2 工具映射表（起始版本）

> 下表是**转换契约**，不是声称所有 OpenCode 安装都具有同名工具。实际名称以目标版本暴露的 tool schema 为准，并将 schema snapshot 存进数据集。

| Claude Code 工具 | 目标 OpenCode 语义 | 转换规则 | 注意点 |
| --- | --- | --- | --- |
| `Bash` | `bash` / shell 执行工具 | 保留命令、timeout、工作目录；移除 CC 专属字段 | 严格 sandbox；禁止将真实密钥或宿主路径写入样本 |
| `Read` | `read` | 文件路径与行范围映射到目标 schema | 保留文件不存在等失败 observation |
| `Write` | `write` | 映射 `file_path`、`content` | 写入副作用必须在独立任务 sandbox 回放 |
| `Edit` | `edit` / patch | 用目标的 old/new text 或 diff 结构表达 | 需验证编辑失败、模糊匹配和换行规则 |
| `Glob` | `glob` | 映射 pattern、path | 输出排序应稳定，否则训练观察噪声大 |
| `Grep` | `grep` / search | 映射 pattern、include、path | 正则引擎与输出上限可能不同 |
| `WebFetch` | `webfetch`（若启用） | 只保留允许联网任务；记录来源 | 数据隐私、时效与网页注入风险高 |
| `WebSearch` | `websearch`（若启用） | 把 query 和结果 schema 对齐 | 不可复现实例应做快照或从训练集剔除 |
| `TodoWrite` | `todowrite` / 计划状态 | 映射任务列表、状态枚举 | 若目标 harness 无此工具，不训练伪工具调用 |
| `Skill` | `skill` | 名称、base directory、Skill 内容作为 tool result | Skill 本身是指令，不应成为 assistant 生成目标 |
| `Task` | child session / delegate | 建 child trace，父会话只接收结果摘要 | 禁止把 child 的全部隐私上下文直接泄给 parent |

## 3. 三类数据的精确格式差异

### 3.1 Slash 数据：用户显式点名 Skill

**Claude Code 的语义（抽象）**：用户发出 `<command-args>query`；随后 CC 运行时将该命令关联的 `SKILL.md` 与 base directory 作为上下文预加载/传入。训练数据中应保留“这是用户显式指定”的信号。

```text
user: <command-args>query
user: Base directory: <cc-skill-base>\n\n<skill.md content>
```

**OpenCode 目标格式**：将同一个显式选择改写为目标 harness 可见的 Skill 内容和其基目录。例如：

```text
user: <skill.md content>
      Base directory for this skill: /workspace/.opencode/skills/ifind-edb
      Relative paths in this skill (e.g., scripts/, references/) are relative to this base directory.

      query
```

关键点是 base directory 必须是**任务 sandbox 内的稳定虚拟路径**，不是采集机器的 `/Users/...`。slash 样本不需要学生先调用 `skill`；用户已选择 Skill。

### 3.2 Non-slash 数据：模型自主调用 Skill

**Claude Code 的语义（抽象）**：系统提示给出“可用 Skills”；用户给 query；模型调用 `Skill`；tool result 返回“Launching skill”及对应 base directory/内容。

```text
user: <system-reminder>\nThe following skills are available; ...
user: query
assistant tool_call: Skill(name="ifind-edb")
tool: Launching skill: Base directory for this skill: <cc-skill-base> ...
```

**OpenCode 目标格式**：系统列出目标可用 Skills；用户只给任务；模型显式调用 `skill`；tool result 返回 Skill 内容及打包文件。

```text
system: <default.txt | gpt.txt | anthropic.txt>
        Skills provide specialized instructions and workflows for specific tasks.
        Use the skill tool to load a skill when a task matches its description.

        ## Available Skills
        - **ifind-edb**: ...
user: query
assistant tool_call: skill(name="ifind-edb")
tool: <skill_content name="ifind-edb">\n# Skill: ifind-edb\n...\n
      <skill_files><file>...</file></skill_files></skill_content>
```

这里 `skill_content` 是 observation，不训练为 assistant 输出；训练 assistant 的选择、参数和之后依据内容行动的 token。

### 3.3 Subagent 数据：child session 按需加载

**CC 采集时的现象**：slash Skill 可能已经预加载；系统提示和子代理指令会写入 CC/cowork 语境。不能因为 child 最终完成任务，就把它从一开始看到的所有上下文错当成 OpenCode 的默认行为。

**OpenCode 目标格式**：创建 child session；child 的 system 写入 lumi/目标 system 配置并列出 `available_skills`；用户发送 query；若任务需要，再由 child 调用 `skill` 并接收 `skill_content`。

```text
child system: <lumi system> + available_skills
child user: query
child assistant tool_call: skill(name="ifind-edb")
child tool: <skill_content name="ifind-edb">...</skill_content>
child assistant: result / summary
parent tool result: <child summary only>
```

父/子之间只传 harness 实际会传的结果。对于 CC 中“预加载 slash Skill”的 subagent，转换器要么保留为 OpenCode 的显式首轮 `skill` 调用，要么仅当 OpenCode 确实支持预加载时采用等价目标格式。

## 4. 清洗、切分与训练

### 4.1 必做过滤

- **安全与隐私**：删除 token、Cookie、邮箱、真实用户名、私有 URL、工作站绝对路径和未授权代码；Web observation 需防 prompt injection。
- **可执行性**：保留成功、有效失败恢复和真实错误处理；删除 harness 崩溃、缺 observation、工具 schema 不匹配的监督目标。
- **重复与泄漏**：按仓库、任务、Skill、近似文本和工具序列去重；同一仓库/Issue 的不同改写不可跨 train/eval。
- **长度**：优先裁剪无关观察，不要裁掉 tool call 与紧邻 result 的因果对；记录 `truncated=true`，并在评测集保留长样本。

### 4.2 训练与评测矩阵

| 评测维度 | 要回答的问题 | 最低记录 |
| --- | --- | --- |
| 工具语法 | tool call 能否被 OpenCode parser 接收？ | parse success、参数校验失败率 |
| Skill 选择 | non-slash 时是否选对 Skill？ | top-1 / top-k、误调用率 |
| Skill 遵循 | 加载后是否遵守相对路径和步骤？ | Skill-specific pass rate |
| Slash 兼容 | 显式指定 Skill 是否被正确使用？ | slash task success |
| Subagent | 是否创建 child、隔离上下文并整合结果？ | child success、父会话最终成功率 |
| 端到端 | 完成真实工作区任务吗？ | sandbox task success、token/步骤/时延 |

离线 loss 只能作为训练诊断。最终指标必须是用固定版本的 OpenCode、固定 tool schema、干净 sandbox 和隐藏任务进行回放的成功率。

## 参考资料

1. [Teich：agent trace 生成、规范化、mask 与审计](https://github.com/TeichAI/teich)
2. [Teich README：提取 Claude session 与转换为 OpenAI-style JSONL](https://github.com/TeichAI/teich#quickstart-extract-local-sessions)
3. [SLIME Coding-Agent RL：harness、trajectory 与 token supervision 的参考实现](https://github.com/THUDM/slime/tree/main/examples/coding_agent_rl)
