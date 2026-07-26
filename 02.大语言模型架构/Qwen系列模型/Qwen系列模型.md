# Qwen 系列开源权重：从初代到 Qwen3.6 的架构、训练与演进

> **范围（截至 2026-07-26）**：本文只记录可下载并自行部署的 Qwen 开源权重，不覆盖 API、Qwen Code 或 Plus 服务型号。Qwen、Qwen2 有正式技术报告，Qwen1.5、Qwen2.5、Qwen3 有官方发布说明；`Qwen3.5` 已在官方 Hugging Face 组织发布 4B、9B、`35B-A3B` 等权重，`Qwen3.6-35B-A3B` 也有官方权重仓库。3.5/3.6 尚未找到完整独立技术报告，未披露的参数、数据和训练算法不作推断。

## TL;DR

Qwen 主线并非每代都推翻 Transformer，而是沿着以下方向累积：

```text
Qwen 初代：中英双语 Decoder-only + SFT/RLHF
  → Qwen1.5：模型谱系扩展、32K、Transformers 原生兼容、首次 MoE
  → Qwen2：GQA、128K、更多语言、MoE 主线
  → Qwen2.5：18T 数据、结构化输出、Coder/Math/VL 与 1M 长上下文分支
  → Qwen3：36T、119 语言、dense + MoE、thinking/non-thinking 融合、Agent/MCP
  → Qwen3.5：开源多模态权重（4B、9B、35B-A3B 等）
  → Qwen3.6：公开 35B-A3B 权重
```

**面试记忆**：`Qwen1.5` 解决“好用与可部署”，`Qwen2` 把 GQA/128K/MoE 带入主线，`Qwen2.5` 把通用基座细分到代码、数学、视觉与百万上下文，`Qwen3` 把 reasoning budget、RL 和 Agent 能力合进同一系列。

## Fast Look

**实体关系**：`Tokenizer / multilingual data → Decoder-only Transformer → RoPE + GQA + SwiGLU → Dense 或 MoE → SFT / DPO / PPO / RL → thinking budget → tool/MCP Agent`。版本号表示一次公开发布主线；`Coder / Math / VL / Omni / Embedding` 是围绕同一代底座衍生的专项分支，不能和“通用文本基座”直接横向比较。

#### Qwen 初代（Qwen / 通义千问）

- **描述**：初代开源语言模型包含 1.8B、7B、14B、72B 等 Decoder-only 基座与 Chat 版本，使用 next-token prediction 预训练，并以 SFT、RLHF 完成助手对齐；训练数据以中英为强项并覆盖多语言。
- **与之前方法的区别**：相对以英文为中心的早期开源 Decoder-only 模型，Qwen 将中文/英文与多语言 tokenization、对话/工具使用放入同一开源谱系；相对 BERT，选择纯因果生成而非 MLM。
- **优点**：建立从小模型到 72B 的统一中文 LLM 生态，初代便提供 Base/Chat、量化、工具和多模态延伸的开发入口。
- **缺点/注意点**：早期各尺寸上下文长度并不统一，语言能力仍受训练数据覆盖影响；“Qwen”也是品牌总称，不能把后续 Qwen2/3 的架构特性倒灌给初代。

#### Qwen1.5：32K、生态兼容与早期 MoE

- **描述**：Qwen1.5 将 Base/Chat 扩展至 0.5B–110B 量级，并发布 Qwen1.5-MoE-A2.7B；官方强调全系 32K context、DPO/PPO 对齐，以及直接使用 Hugging Face `transformers` 的 chat template 与量化/推理生态。
- **与之前方法的区别**：相对初代，重点从“开放模型”转向“可直接部署”：不再依赖 `trust_remote_code`，对齐、量化与 vLLM/SGLang/llama.cpp 等框架集成更标准化；MoE 作为条件计算分支首次明确进入系列。
- **优点**：降低微调和推理接入成本；从小到大统一 32K，便于生产选型；MoE-A2.7B 展示以较少 active 参数匹配 7B 级能力的方向。
- **缺点/注意点**：统一最大长度不表示所有任务的长上下文质量一致；生态兼容是工程价值，不等于架构能力提升；MoE 的总参数和 active 参数必须分开看。

#### Qwen2：GQA、128K 与 57B-A14B MoE

- **描述**：Qwen2 发布 dense 0.5B/1.5B/7B/72B 与 MoE `57B-A14B` 等模型，强化代码、数学和多语言；部分 Instruct 模型将上下文扩展至 128K。技术报告披露其采用现代 Decoder-only 组件，包括 GQA、RoPE、RMSNorm 与 SwiGLU。
- **与之前方法的区别**：相对 Qwen1.5，Qwen2 更系统地采用 GQA 降低 KV Cache，并把 MoE 和 128K 长上下文纳入正式主线；相对 MHA 基座，多个 Query 头共享分组 KV。
- **优点**：GQA 使长上下文 Decode 更省显存/带宽；128K 与更多语言扩大应用范围；MoE 用较低激活量取得更大总容量。
- **缺点/注意点**：并非每个尺寸或版本都支持同样的 128K；GQA 和 MoE 都是质量—成本折中，不能只根据总参数判断推理速度。

#### Qwen2.5：18T 数据、结构化输出与专项模型

- **描述**：Qwen2.5 通用文本模型在约 18T token 上训练，强调知识、代码、数学、指令遵循、长文本与 JSON/结构化输出；主模型多数支持 128K context、8K generation。同期形成 `Qwen2.5-Coder`、`Qwen2.5-Math`、`Qwen2.5-VL` 等专项系列。
- **与之前方法的区别**：相对 Qwen2 的通用基座升级，2.5 将数据规模与任务专化并行推进：Coder 专注代码数据，Math 显式支持 CoT/PoT/TIR，VL 将视觉感知与推理继续做细。
- **优点**：在通用模型与专项模型之间提供更清晰的选型；强化 JSON、表格和系统提示鲁棒性，适合工具调用与生产输出协议。
- **缺点/注意点**：专项模型的分数不能代表通用模型；18T 是预训练数据量而非单一任务数据；JSON “能生成”不替代 schema 校验、重试和工具权限控制。

#### Qwen2.5-1M：RoPE 长度扩展与稀疏推理

- **描述**：Qwen2.5-1M 将 7B/14B Instruct checkpoint 扩展到 1M context；训练上从 4K 逐步提升到 256K，并将 RoPE base frequency 由 10,000 调至 10,000,000，随后再做分阶段 SFT；推理框架结合 sparse attention。
- **与之前方法的区别**：相对标准 128K Qwen2.5，采用渐进式长度扩展、调整 RoPE 基频与专用推理框架，而非只改配置中的 `max_position_embeddings`。
- **优点**：给出可开源部署的百万上下文路线；官方称其 vLLM-based 框架配合稀疏注意力可显著加速 1M input 的处理。
- **缺点/注意点**：1M 检索指标不代表任意复杂长文推理无损；KV、prefill 延迟和并发成本仍高，长度外推要同时评测短上下文回归与长上下文任务。

#### Qwen3 Dense / MoE：小模型效率与 235B-A22B

- **描述**：Qwen3 同时发布 dense（0.6B–32B）和 MoE（30B-A3B、235B-A22B）模型；MoE 使用 128 位专家、每 token 激活 8 位专家，较大模型提供 128K context。官方称较小 dense Qwen3 可达到更大 Qwen2.5 的基础能力，MoE 以约 10% active 参数接近 Qwen2.5 dense 表现。
- **与之前方法的区别**：相对 Qwen2.5，Qwen3 将“同代 dense + MoE + reasoning 后训练”作为统一发布，而不只围绕专项分支升级；重点是参数效率和训练/推理预算的协同。
- **优点**：覆盖从本地小模型到大 MoE 的部署范围；稀疏模型在保持容量的同时降低每 token active FLOPs，便于按成本选择。
- **缺点/注意点**：`235B-A22B` 的 `A22B` 是激活参数口径，不能和 235B dense 直接比；MoE 仍有专家通信、路由均衡和硬件并行要求。

#### Hybrid Thinking Modes：`/think`、`/no_think` 与 reasoning budget

- **描述**：Qwen3 将 Thinking 与 Non-Thinking 纳入同一模型；通过 chat template 的 `enable_thinking` 和 `/think`、`/no_think` 实现 turn-level 切换，以调节测试时推理 token 预算。
- **与之前方法的区别**：相对“思考模型”和“快速聊天模型”分开部署，Qwen3 尝试在同一模型中融合长 CoT 与直接回答；后训练流程为 long-CoT cold start → reasoning RL → thinking mode fusion → general RL。
- **优点**：按任务难度平衡延迟、成本与质量；同一模型在工具/Agent 任务中可按 turn 决定是否展开推理。
- **缺点/注意点**：开关只是预算与行为控制，不保证思考更长必然更正确；reasoning 内容会提高时延和上下文成本，生产系统仍应设置最大 token、超时和工具验证。

#### Qwen3 四阶段后训练：Cold Start → Reasoning RL → Fusion → General RL

- **描述**：Qwen3 的后训练依次进行：长 CoT cold start，使用规则奖励扩展 reasoning RL，将 long-CoT 与指令数据混合以融合 thinking / non-thinking，最后在 20+ 通用任务上做 General RL，覆盖指令、格式和 Agent 行为。
- **与之前方法的区别**：相对“仅 SFT 后直接偏好/RL 对齐”的单线流程，Qwen3 先单独建立长推理能力，再显式将其与快速回答融合，最后再做通用行为校正；这解释了同一模型为何能有 `/think` 与 `/no_think`。
- **优点**：将推理探索、直接回答和生产指令遵循分层处理，降低直接混训造成的能力冲突；规则奖励使数学、代码等可验证任务能扩展 RL 计算。
- **缺点/注意点**：阶段顺序和数据配比会决定最终取舍，融合阶段仍可能稀释推理深度或快速回答质量；20+ 任务不等于覆盖所有工具/Agent 现实环境，必须用独立 benchmark 与人工评测检查回归。

#### Qwen3 数据与多语言预训练

- **描述**：Qwen3 使用约 36T token、覆盖 119 种语言/方言；预训练分为 4K 的通用阶段、强化 STEM/代码/推理的数据阶段和长上下文阶段，并用 Qwen2.5-VL/专项模型辅助 PDF 文本提取和合成数据构造。
- **与之前方法的区别**：相对 Qwen2.5 约 18T token，数据规模近乎翻倍，并显式将 PDF-like documents、合成数学/代码材料和 119 语言纳入数据工程。
- **优点**：更强的知识、STEM、代码与跨语言覆盖；阶段化训练降低一开始用长序列训练的成本。
- **缺点/注意点**：token 数量不等于数据质量或去污染效果；合成数据会带来教师偏差，119 语言的覆盖深度并不必然均匀。

#### Qwen3 Agent、MCP 与 Qwen3-Coder

- **描述**：Qwen3 后训练的 general RL 覆盖指令、格式与 Agent 任务，并强化 MCP 支持；`Qwen3-Coder` 是面向代码/Agent 工作流的专项分支，Qwen Code 将其作为 coding agent 底座。
- **与之前方法的区别**：相对只生成文本/代码补全的基座模型，Agent 路线将工具 schema、环境 observation、多轮决策和执行结果纳入模型/运行时接口。
- **优点**：更适合 repository-level coding、MCP 工具调用和多步骤任务；通用 Qwen3 与 Coder 分支形成通用—专项的选择空间。
- **缺点/注意点**：Agent 成功率取决于 harness、工具、权限、上下文管理与评测环境，不是纯底座指标；不同 Coder 开源版本的能力、上下文窗口和部署要求会变动。

#### Qwen3.5：4B / 9B Dense 与 35B-A3B MoE

- **描述**：`Qwen3.5-4B`、`Qwen3.5-9B` 与 `Qwen3.5-35B-A3B` 均由 Qwen 官方 Hugging Face 组织发布，仓库含 Transformers 格式的后训练权重与配置。4B/9B 是 dense 尺寸；9B 的模型卡披露 Gated DeltaNet 与 Gated Attention 的混合结构、262K 原生上下文；35B-A3B 是图文到文本的 MoE，其中 35B 是总参数量、3B 是每 token 激活参数量。
- **与之前方法的区别**：相对 Qwen3 的文本 dense/MoE 发布，Qwen3.5 明确采用统一视觉—语言基座，并在 9B 中将线性注意力（Gated DeltaNet）和门控全注意力交错组合。`Qwen3-30B-A3B` 属于上一代 Qwen3；已核实的 Qwen3.5 MoE 名称是 `Qwen3.5-35B-A3B`。
- **优点**：4B 适合资源受限的本地推理，9B 提供更完整的中等规模能力，35B-A3B 以 3B active 参数换取更大总容量；公开权重可用 Transformers、vLLM、SGLang 等自行部署、量化和微调。
- **缺点/注意点**：`35B` 与 `A3B` 是不同口径，不能用总参数直接估算每 token 成本；9B 和 35B-A3B 都含视觉能力，和纯文本 Qwen3 的基准不应直接横比；完整数据配方和后训练细节尚未公开，不能从模型卡外推。

#### Qwen3.6-35B-A3B：已发布的开源 MoE checkpoint

- **描述**：Qwen 官方 Hugging Face 组织发布了 `Qwen3.6-35B-A3B` 权重仓库；按命名，35B 为总参数规模、A3B 为激活参数规模。本文仅记录已由官方仓库明确支持的这一开源型号。
- **与之前方法的区别**：相对 Qwen3.5 的多个 dense/MoE 开源尺寸，当前可核实的 Qwen3.6 开源记录聚焦在一个 `35B-A3B` MoE checkpoint；其不应与未纳入本文的托管产品名称混同。
- **优点**：提供一个可自部署的较大稀疏模型选择；A3B 的激活规模有利于在总容量和单 token 计算之间折中。
- **缺点/注意点**：目前未找到可与 Qwen2/Qwen3 技术报告同等完整的 Qwen3.6 架构、数据与训练报告；除官方模型卡明确内容外，不应推断专家数、注意力变体、上下文长度或性能结论。

## 版本时间线与证据等级

| 阶段 | 公开主线 | 代表变化 | 证据等级 |
| --- | --- | --- | --- |
| 2023 | Qwen 初代 | 1.8B–72B、中文/英文、Base/Chat、SFT/RLHF | 技术报告 |
| 2024-02 | Qwen1.5 | 0.5B–110B、32K、Transformers 原生兼容、MoE | 官方发布 |
| 2024-06 | Qwen2 | GQA、RoPE、RMSNorm、SwiGLU、128K、57B-A14B | 技术报告 |
| 2024-09 起 | Qwen2.5 | 18T、JSON/长文本、Coder/Math/VL、1M 分支 | 官方发布/专项报告 |
| 2025-04 | Qwen3 | 36T、119 语言、dense+MoE、hybrid thinking、Agent/MCP | 官方发布 |
| 2026 | Qwen3.5 | 官方开源 4B、9B、`35B-A3B` 等；视觉—语言、混合架构 | 官方模型卡；无完整独立技术报告 |
| 2026 | Qwen3.6 | 官方权重仓库可见 `35B-A3B` | 官方模型仓库；无完整独立技术报告 |

## 面试回答框架

**Q：Qwen 从 2.5 到 3 的关键变化是什么？**

Qwen2.5 更像“成熟的多专项 foundation model 平台”：扩大到 18T token，强调结构化输出，并拆出 Coder/Math/VL/1M。Qwen3 则进一步把 36T 多语言预训练、dense/MoE 参数效率、reasoning RL 和 thinking/non-thinking 融合到统一代际，并面向 MCP 与 Agent。它并非单纯增加参数，而是把训练时计算、测试时思考预算和工具任务结合起来。

**Q：Qwen3.5 是否开源？`30B-A3B` 又属于哪一代？**

是。官方 Qwen 组织已经发布 `Qwen3.5-4B`、`Qwen3.5-9B` 和 `Qwen3.5-35B-A3B` 等权重，能够按模型卡说明自部署。易混淆之处是名称：`Qwen3-30B-A3B` 是 Qwen3 的 MoE；已核实的 Qwen3.5 MoE 名称为 `Qwen3.5-35B-A3B`。回答时应先核对**精确模型 ID**，再谈架构和能力。

## 参考资料

1. [Qwen 技术报告与初代介绍](https://qwenlm.github.io/blog/qwen/)
2. [Qwen1.5 官方发布](https://qwenlm.github.io/blog/qwen1.5/)
3. [Qwen2 官方发布与技术报告入口](https://qwenlm.github.io/blog/qwen2/)
4. [Qwen2.5 官方发布](https://qwenlm.github.io/blog/qwen2.5/)
5. [Qwen2.5-1M 官方说明](https://qwenlm.github.io/blog/qwen2.5-1m/)
6. [Qwen3 官方发布](https://qwenlm.github.io/blog/qwen3/)
7. [Qwen3.5-9B 官方模型卡与权重](https://huggingface.co/Qwen/Qwen3.5-9B)
8. [Qwen3.5-35B-A3B 官方模型卡与权重](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
9. [Qwen3.6-35B-A3B 官方权重仓库](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

## 小红书报告解读（已直读，作交叉阅读）

- [过一遍 Qwen3 技术报告，增加知识储备](https://www.xiaohongshu.com/explore/69a83cb4000000001a01fbc1?xsec_token=ABWf9lIlPV9kRXjPdSomZ74lFlncXoUCr8LOTaLT3_AuA%3D&xsec_source=pc_search)：将 Qwen3 的核心概括为 unified thinking/non-thinking 与架构性价比改进；本文据官方报告落实为 hybrid thinking、四阶段后训练与 dense/MoE 的对应实体。
- [Qwen3 技术解析](https://www.xiaohongshu.com/explore/68fdbb7e000000000703a2e2?xsec_token=ABkhqcIt7UTure_s8mKuQB75DNLcGBlRiW-KjILdIXclk%3D&xsec_source=pc_search)：补充关注数据工程、分阶段训练、强—弱蒸馏与思考融合；其中“后续版本重新拆分思考模式”的表述未在本页作为官方技术事实采用。
