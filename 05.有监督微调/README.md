# 05.有监督微调

## Fast Look

**选择链路**：先判断知识缺口用 `CPT` 还是行为适配用 `SFT`；再在全参、BitFit、Prompt/Prefix/P-Tuning、Adapter、LoRA/QLoRA 中按显存、任务数和效果选择。

#### 全参数微调（Full Fine-tuning）与 SFT

- **描述**：更新全部模型参数；SFT 用指令—回答数据最小化 assistant token 的交叉熵。
- **与 PEFT 的区别**：PEFT 冻结主干只训练少量参数；全参微调改变所有权重。
- **优点**：适配能力上限高，不受特定低秩/模块结构限制。
- **缺点/注意点**：优化器状态与梯度显存巨大，多任务需保存完整副本，更易灾难性遗忘。

#### Continue Pre-training（CPT）

- **描述**：用领域原始文本继续 next-token 预训练，通常先于 SFT。
- **与 SFT 的区别**：CPT 注入术语、事实和写作分布；SFT 注入指令遵循和输出格式。
- **优点**：无需逐条人工问答标注，适合大量领域文档。
- **缺点/注意点**：会遗忘通用能力；应做数据配比、学习率控制和领域/通用双评测。

#### BitFit

- **描述**：冻结大部分参数，只训练 bias 项。
- **与全参微调的区别**：训练参数极少，不新增网络结构。
- **优点**：显存和存储开销极低，实现简单。
- **缺点/注意点**：容量最小，复杂生成任务和较大领域偏移下常不够。

#### Prefix Tuning、Prompt Tuning、P-Tuning 与 P-Tuning v2

- **Prefix Tuning**：为每层 Attention 注入可训练的 K/V 前缀；相对 Prompt Tuning 作用到深层，优点是调制能力更强，缺点是前缀带来额外推理开销。
- **Prompt Tuning**：只在输入 embedding 前添加 soft prompt；优点是参数最少，缺点是在小模型/复杂任务中可能效果弱。
- **P-Tuning/P-Tuning v2**：用 prompt encoder 生成连续提示，并在 v2 中扩展到深层；相对手写 prompt 更可学习，缺点是仍受虚拟 token 长度和任务形式限制。

#### Adapter Tuning、AdapterFusion、AdapterDrop、MAM 与 UniPELT

- **Adapter Tuning**：在 Transformer 层插入 bottleneck；相对 LoRA 新增显式模块，优点是任务模块可独立保存，缺点是推理有额外层开销。
- **AdapterFusion**：融合多个 Adapter；优点是组合任务知识，缺点是维护和冲突处理复杂。
- **AdapterDrop**：跳过部分 Adapter；优点是降计算，缺点是层选择影响效果。
- **MAM/UniPELT**：组合 prefix、adapter、LoRA；优点是灵活，缺点是训练/调参更复杂，简单任务常无必要。

#### LoRA、AdaLoRA 与 QLoRA

- **LoRA 描述**：冻结 `W`，学习低秩增量 `ΔW=BA`，可合并回主权重。
- **AdaLoRA 描述**：动态给不同层/矩阵分配秩；相对 LoRA 更聚焦重要模块。
- **QLoRA 描述**：4-bit 量化存储基座，在其上训练 LoRA，常配 NF4、双重量化和 paged optimizer。
- **优点**：LoRA 显著降低训练参数与 checkpoint；QLoRA 让单卡微调更大模型成为可能。
- **缺点/注意点**：rank、target modules、alpha 与数据格式决定效果；量化训练有精度/硬件限制。

#### Llama 2 与 ChatGLM 3 微调实践

- **描述**：将数据模板、tokenizer、训练参数和 PEFT 方法应用于具体基座模型。
- **与通用理论的区别**：需要遵守模型各自 chat template、EOS、角色 token 和目标字段。
- **优点**：提供从样本到 checkpoint 的可复现实操入口。
- **缺点/注意点**：配置不能跨模型机械复用；先做小样本过拟合与生成 sanity check。

#### Claude Code 轨迹蒸馏 SFT：Teich、Harness、Slash / Subagent

- **描述**：将 Claude Code 多轮工具轨迹用 Teich 规范化为 canonical trace，再按 OpenCode 的 system/user/tool 运行时语义渲染；数据拆分为 slash、non-slash 与 subagent 三桶，并对 Bash、Edit、Read、Skill、Task 等工具做 schema 级映射。
- **与之前方法的区别**：相对普通 instruction SFT 的“用户—答案”对，agent SFT 需保留工具调用、工具观察、Skill 注入位置和子会话边界；相对直接训练 CC transcript，目标是 OpenCode 的实际上下文协议。
- **优点**：可让 4B、9B、30B 等学生模型学会在目标 harness 中选择/执行 Skill 和工具；response-only mask 防止模型学习伪造工具回显。详见 [Claude Code轨迹蒸馏SFT](./Claude%20Code轨迹蒸馏SFT/Claude%20Code轨迹蒸馏SFT.md)。
- **缺点/注意点**：工具改名不等于语义等价；绝对路径、密钥、私有代码和子会话私有上下文必须清洗/隔离；最终以 OpenCode 沙箱回放成功率而非离线 loss 判断效果。

### 5.1 理论

[1.基本概念](/05.有监督微调/1.基本概念/1.基本概念.md "1.基本概念")

[2.prompting](/05.有监督微调/2.prompting/2.prompting.md "2.prompting")

[3.adapter-tuning](/05.有监督微调/3.adapter-tuning/3.adapter-tuning.md "3.adapter-tuning")

[4.lora](/05.有监督微调/4.lora/4.lora.md "4.lora")

[5.总结](/05.有监督微调/5.总结/5.总结.md "5.总结")

### 5.2 微调实战

[llama2微调](/05.有监督微调/llama2微调/llama2微调.md "llama2微调")

[ChatGLM3微调](/05.有监督微调/ChatGLM3微调/ChatGLM3微调.md "ChatGLM3微调")

[Claude Code轨迹蒸馏SFT](./Claude%20Code轨迹蒸馏SFT/Claude%20Code轨迹蒸馏SFT.md "Claude Code轨迹蒸馏SFT")

### 5.3 一些题目

[1.微调](/05.有监督微调/1.微调/1.微调.md "1.微调")

[2.预训练](/05.有监督微调/2.预训练/2.预训练.md "2.预训练")

参考资料：

-   [liguodongiot/llm-action](https://github.com/liguodongiot/llm-action#llm微调实战 "liguodongiot/llm-action")
