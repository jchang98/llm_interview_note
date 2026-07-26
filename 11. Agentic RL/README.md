# 11. Agentic RL

## Fast Look

**训练闭环**：`任务/网页/问答 → 合成 trajectory → Chat Template 序列化 → Agent rollout + 工具反馈 → rubric/pairwise reward → PPO/GRPO/DAPO/GSPO → 自蒸馏`。与传统 RLHF 的关键不同是：反馈可包含报错、测试输出和工具记录，而不只是单一标量。

#### Direct Distillation、CRAWLQA 与 E2HQA

- **Direct Distillation**：强模型 rollout 已知问题，仅保留正确轨迹；**优点**是数据干净直接，**缺点**是覆盖受已有问题限制。
- **CRAWLQA**：从网页集合和访问轨迹构造问答；**与直接蒸馏的区别**是引入真实检索/浏览过程，**优点**是环境更真实，**缺点**是网页变动与爬取噪声大。
- **E2HQA**：从简单 QA 提取实体并检索替换，反向构造更难问题；**优点**是可扩展难度，**缺点**是替换后可能语义失真或答案不可验证。

#### Short Think、Long Think 与 Atom Think

- **Short Think**：短步骤推理，合成时可保留既有上下文；**Long Think**：长推理链，常需清除历史思维避免重复/污染。
- **Atom Think**：将大段推理拆成原子步骤，再组合成 `<think>` 块。
- **与普通最终答案监督的区别**：显式监督过程轨迹，而非只监督最终结果。
- **优点**：便于过程奖励、错误定位和工具决策训练。
- **缺点/注意点**：思维轨迹可能是事后合理化；冗长 chain 会增加训练成本并携带教师偏差。

#### Chat Template、system/user/assistant/tool 与 Tool Call

- **描述**：Chat Template 将角色消息、`reasoning_content`、`<think>`、`<tool_call>`、`<tool_response>` 等序列化为模型 token。
- **与普通纯文本 SFT 的区别**：显式编码角色、工具参数、工具结果和可继续推理的状态。
- **优点**：训练、评测和部署共享严格协议，Agent 才能可靠调用工具。
- **缺点/注意点**：模板、停止词、转义和多工具调用任何一处不一致都会造成训练—推理失配或工具解析失败。

#### Agent 数据集与 Agent 基座：DeepDive、MiroThink、DR-Tulu、ASearcher、MiroThinker、WebExplorer、ArenaRL

- **描述**：DeepDive/MiroThink/DR-Tulu/ASearcher 提供不同的推理、检索、rubric 或研究型任务轨迹；MiroThinker、WebExplorer、DR-Tulu、ArenaRL 分别代表搜索、迭代浏览、rubric 评分和 pairwise 训练范式。
- **与通用指令数据的区别**：样本包含环境交互、工具观察和多步决策，而不是单轮问答。
- **优点**：更贴近真实 Agent 的长程规划与工具使用。
- **缺点/注意点**：数据许可证、环境可复现性、工具版本和最终可验证性必须单独审计。

#### Context Summarization 与 Sliding Window

- **Summarization 描述**：上下文达到阈值时压缩早期消息；**Sliding Window**：仅保留最近固定窗口消息。
- **与全量上下文的区别**：用信息损失换取固定上下文成本。
- **优点**：可支持长任务和持续运行，避免上下文无限增长。
- **缺点/注意点**：摘要可能遗漏约束，滑窗会遗忘早期目标；应保留结构化状态和原始日志索引。

#### Rubric Reward、Pairwise Reward 与排序赛制

- **Rubric Reward**：按明确评分规则给轨迹/答案打分；**优点**是可解释，**缺点**是规则覆盖有限。
- **Pairwise Reward**：比较两条轨迹优劣；相对单点分数可降低标尺漂移，但比较数可达 `O(N²)`。
- **循环赛、锚点排名、单/双淘汰、瑞士制**：是降低比较成本或提高排序可靠性的不同赛制；锚点/淘汰约 `O(N)`，瑞士制约 `O(N log N)`，都以部分排序信息换成本。
- **注意点**：排序系统会放大锚点、初始种子与 Judge 偏差，需定期人工校准。

#### Self-distillation、PPO、GRPO、DAPO 与 GSPO

- **Self-distillation**：将自身高质量 rollout 回流为训练数据；**优点**是持续扩充数据，**缺点**是易形成确认偏误和多样性坍缩。
- **PPO**：基于 critic/advantage 的受约束策略优化；**GRPO**：以组内相对奖励估计优势、可减少 value model 依赖；**DAPO/GSPO**：面向更稳定/高效的偏好或组策略优化变体。
- **与普通 SFT 的区别**：这些算法利用 rollout 奖励优化行为，而不仅模仿参考答案。
- **注意点**：具体算法名称不保证效果；奖励设计、rollout 质量、KL/裁剪与训练稳定性才是主变量。

#### OPD、MOPD 与 OPCSD：在策略蒸馏的三种组织方式

- **描述**：**OPD** 在 student 自身 rollout 前缀上匹配教师的 token-level 分布；**MOPD** 用多个领域教师整合专长；**OPCSD** 用较早阶段 checkpoint 在当前 rollout 上蒸馏以恢复遗忘能力。可将 teacher—student log-ratio / KL 转成 token reward 或 advantage，复用 RL 框架更新。
- **与之前方法的区别**：离线 SFT/蒸馏用固定 teacher trajectory；OPD 先让 student 走出自己的轨迹再求教师纠正。MOPD 是横向多领域融合，OPCSD 是纵向跨训练阶段保持能力。
- **优点**：降低训练—部署的前缀分布失配，并给稀疏终局 reward 补充稠密 token 信号；适合长 CoT、Agent 与多阶段后训练。完整笔记见 [OPD 在策略蒸馏](./OPD%20在策略蒸馏/OPD%20在策略蒸馏.md)。
- **缺点/注意点**：教师调用、全词表 logits 与多教师调度成本很高；reverse/forward KL、token mask、reward 符号和 tokenizer 对齐必须严格正确，OPD 不等同于把 GRPO 名称替换掉。

#### DeepSeek-V4 两阶段后训练

- **描述**：先按数学、代码、Agent、指令跟随等领域独立训练专家，再以在策略蒸馏（OPD）合并；专家训练结合 SFT、GRPO 与领域奖励，开放任务以生成式奖励模型（GRM）评价轨迹。
- **与传统 RLHF 的区别**：不以单一通用策略直接串行执行 SFT→RM→PPO 为中心，而是先专精、后用多教师全词表反向 KL 蒸馏整合。
- **优点**：将各领域优化解耦，并保留在线策略分布中的蒸馏信号；完整笔记见 [DeepSeek-V4 后训练](./DeepSeek%20V4%20后训练/DeepSeek%20V4%20后训练.md)。
- **缺点/注意点**：基于 V4 预览版技术报告及其中文翻译，应与官方后续报告交叉核验；多教师 rollout、全词表 logits 和容错服务均有高系统成本。

#### DeepSeek-V3.2：DSA、混合 GRPO 与 Agent 任务合成

- **描述**：V3.2-Exp 先以对齐的后训练配置验证 DSA；正式 V3.2 再在 V3.1-Terminus 上继续训练，只新增 DSA：Lightning Indexer 先给历史 token 打分，top-k Token Selection 再做稀疏 MLA 注意力；后训练结合领域专家蒸馏、mixed GRPO 与 1,800+ 环境/85,000 prompts 的 Agent 任务合成。
- **与前述 DeepSeek-V4 后训练的区别**：V3.2 的主线是“稀疏注意力降低长序列成本，再放大 RL 与 Agent 训练”；V4 更强调先训练领域专家、再以 OPD 合并，以及 GRM、QAT 和 rollout 基础设施。二者都有专家与 RL，但解决的主要瓶颈不同。
- **优点**：DSA 保留内容驱动的远程 token 选择而降低主注意力计算；合成 Agent 环境将训练从固定代码/搜索工具扩展到更广交互任务。完整笔记见 [DeepSeek-V3.2 技术报告](./DeepSeek%20V3.2%20技术报告/DeepSeek%20V3.2%20技术报告.md)。
- **缺点/注意点**：Indexer 的理论复杂度仍为 `O(L²)`，top-k 漏检会损失证据；模型最大上下文仍是 128K，冗余自检会令 Agent 轨迹超限。复现 Exp 时还须匹配 Indexer 的 non-interleaved 与 MLA 的 interleaved RoPE layout，不能把 DSA 简化为“完全线性、无损长上下文”。

#### Kimi K3：KDA、AttnRes 与 Stable LatentMoE

- **描述**：Kimi K3 是 2.8T、1M context、native vision、面向开放权重发布的 MoE。官方已披露 KDA、AttnRes、Stable LatentMoE（16/896）、Quantile Balancing、Per-Head Muon、QAT 与全均衡 EP；完整技术报告尚待发布。
- **与前述 DeepSeek-V3.2 的区别**：V3.2 用“Indexer + top-k”将主注意力稀疏化，并以 GRPO/Agent 合成强化后训练；K3 的公开主线是 KDA 处理序列、AttnRes 选择跨层表示，并将 MoE 稀疏度与 expert-parallel 系统推进到更高规模。
- **优点**：极高总容量下每 token 仅路由 16/896 专家，且将路由均衡、低精度 QAT、缓存和通信域一起考虑。完整笔记见 [Kimi K3 技术报告](./Kimi%20K3%20技术报告/Kimi%20K3%20技术报告.md)。
- **缺点/注意点**：KDA/AttnRes/SiTU 的公式和完整训练配方尚未公开；K3 对 preserved thinking history 有兼容性要求，缺失历史或中途换模型会造成不稳定，且可能有过度主动行为。

#### GLM-5.2：IndexShare、长程 Agent 与异步 RL

- **描述**：GLM-5.2 在 GLM-5 的 744B-A40B MoE、DSA 与异步 Agentic RL 基座上，公开升级了稳定 1M context、IndexShare（每四层复用 sparse indexer）、改进 MTP 与可调 reasoning effort。
- **与前述 Agent RL 的区别**：本章通用方法讨论轨迹、奖励和 PPO/GRPO；GLM-5/5.2 给出了一条模型—注意力—推理服务—异步 RL—可验证环境协同设计的具体实现路线。
- **优点**：同时针对长上下文 FLOPs、解码串行瓶颈、KV cache locality 和长 rollout 的 GPU 空转优化；完整笔记见 [GLM-5.2 技术报告](./GLM-5.2%20技术报告/GLM-5.2%20技术报告.md)。
- **缺点/注意点**：GLM-5 有完整技术报告，5.2 公开材料主要披露发布级更新，不能将 5 的训练超参与数据细节直接归因于 5.2。

#### SAO 与 CompactionRL：GLM-5.2 的后训练论文线索

- **描述**：**SAO** 以每个 prompt 一条 rollout、value model 与双侧 token-level clipping 解决异步 RL 的离策略稳定性；**CompactionRL** 将 context summary 也纳入 RL，通过 token-level loss normalization 与跨轨迹 GAE 训练压缩后的长程 Agent。两篇论文均明确称其方法部署在 GLM-5.2 的 RL / Agentic RL pipeline。
- **与之前方法的区别**：SAO 用 single-rollout 替换 GRPO 的 group-wise sampling，以适配异步到达的轨迹；CompactionRL 则不同于“超窗后训练外摘要/截断”，把压缩后的状态和优势估计放回训练闭环。
- **优点**：SAO 可在长 rollout 异步系统中提高训练连续性并抑制 token 级策略漂移；CompactionRL 让有限 context window 下的任务可继续学习，而非只在推理时临时压缩。完整机制见 [GLM-5.2 技术报告](./GLM-5.2%20技术报告/GLM-5.2%20技术报告.md)。
- **缺点/注意点**：两篇论文证明“方法被用于 GLM-5.2 pipeline”，不等于公开了该模型的完整 RL 配方或端到端增益；SAO 依赖 critic/奖励质量，CompactionRL 会承担摘要丢失与优势跨轨迹估计的误差。

## 专题报告

- [DeepSeek-V4 后训练：领域专家、GRM 与 OPD](./DeepSeek%20V4%20后训练/DeepSeek%20V4%20后训练.md)
- [DeepSeek-V3.2 技术报告：DSA、可扩展 RL 与 Agent 任务合成](./DeepSeek%20V3.2%20技术报告/DeepSeek%20V3.2%20技术报告.md)
- [OPD 在策略蒸馏：MOPD、OPCSD 与 RL 实现](./OPD%20在策略蒸馏/OPD%20在策略蒸馏.md)
- [Kimi K3 技术报告笔记：KDA、AttnRes 与 2.8T 稀疏 MoE](./Kimi%20K3%20技术报告/Kimi%20K3%20技术报告.md)
- [GLM-5.2 技术报告：IndexShare、长程 Agent 与异步 RL](./GLM-5.2%20技术报告/GLM-5.2%20技术报告.md)

## 数据合成
![](2026-01-22-15-54.png)

1. 直接蒸馏：已知问题，使用强model rollout；仅保留true answer
2. CRAWLQA 爬虫法: 针对网站集合 R；获得访问轨迹P；调用大模型得到QA = LLM（content（P））
3. E2HQA 逆向工程法：即针对简单问答对 （Q，A）；提取实体E（Q）；搜索S（E（Q））；用搜索信息替代原始Q； Q_n+1 = replace(Q_n, E(Q), S（E（Q））)

tips：
1. short think 可以通过gpt使用react方式生成，合成在步骤i的short think时保留之前的think； 但对于long think，使用qwq在合成步骤i的long think是要清除之前的think
2. 可以将think 拆分为原子think；形式上，我们在包含了思考内容的<think>块中，增加了<atom-think>子块，即多个原子思维构成了最后的思考


## 数据格式
1. 理解qwen chat-template; 
    - system 和 tools。如果有tools，则先拼接system，再拼接tools；如果没有tools，则直接使用system
    - user。如果是user，直接拼接content。
    - assistent。提取reason_content，直接有该字段，或者从content中分割出。添加<think></think>标签，添加<tool_call></tool_call>标签,可以有多个<tool_call>标签
    - tool。包装为 user 消息，并且添加<tool_response>标签


## 数据集
deepdive
mirothink
dr-tulu
aresearch


# Agentic RL infra
## agent repo
1. MiroThinker : 
2. WebExplorer : 根据搜索实体，然后迭代搜索问句
3. dr-tulu : rubric 打分
4. ArenaRL : 使用pair-wise打分


## agent context
1. resum： 达到context * threshold 就进行压缩；training-free
2. slid-window； 只看指定窗口的message


## agent tools



# RL算法

## reward function
1. 对于开放问题，可以使用rubric 打分的机制
2. 随着rl的训练，不同rollout的输出越来越像；信噪比降低;解决方法，不使用单点打分，使用pair-wise打分
- 循环赛，每条轨迹都22对比；o(N^2)
- 基于锚点排名；o(N)
- 种子单淘汰赛；* o(N)；先基于锚点排序；然后根据种子进行淘汰赛
- 双淘汰赛：包括败者组
- 瑞士制；o(nlogn); 分池子3-0池；2-1池等等
3. self-distillation 自蒸馏


## rl algorithm
1. ppo
2. grpo
3. dapo
4. gspo
5. 


# 教程
1. verl实战 https://zhuanlan.zhihu.com/p/1931076626940139506
2. 理论 https://zhuanlan.zhihu.com/p/1998517345900073326
3. 实践复线 [https://zhuanlan.zhihu.com/p/1987092986388038648](https://zhuanlan.zhihu.com/p/1987092986388038648)
4. rubric https://zhuanlan.zhihu.com/p/2004149762870502827

- 数据[https://huggingface.co/datasets/aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample](https://huggingface.co/datasets/aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample)



{
    "agent_name": "my_tool_agent",   
    "data_source":"nq",
    "prompt": [
        {
            "content": "You are a helpful and harmless assistant.",
            "role": "system"
        },
        {
            "content": "Answer the given question. You must conduct reasoning inside <think> and <\/think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <tool_call> query <\/tool_call> and it will return the top searched results between <tool_response> and <\/tool_response>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and <\/answer>, without detailed illustrations. For example, <answer> Beijing <\/answer>. Question: total number of death row inmates in the us?",
            "role": "user"
        }
    ],
    "ability": "fact-reasoning",
    "reward_model": {
        "ground_truth": {
            "target": [
                "2,718"
            ]
        },
        "style": "rule"
    },
    "extra_info": {
        "index": 0,
        "question": "who got the first nobel prize in physics?",
        "split": "train"
    }
}


https://github.com/volcengine/verl/tree/main/verl
verl_project_root/
├── run_train_main.sh                                     # 开发自己的训练总控脚本
├── verl/
│   ├── tools/
│   │   ├── web_search.py                                 # 模拟API搜索工具（替换成自己真实的API）
│   │   ├── my_web_search_tool.py                         # 自定义工具
│   ├── experimental/
│   │   ├── agent_loop/
│   │   │   ├── tool_parser.py                            # 自定义tool parser
│   │   │   ├── my_tool_agent_loop.py                     # 自定义AgentLoop
│   │   ├── reward_loop/ 
│   │   │   ├── reward_manager/
│   │   │   │   ├── my_web_search_reward_manager.py       # 自定义reward manager
│   ├── utils/
│   │   ├── reward_score/
│   │   │   ├── my_reward_function.py                     # 自定义Reward Function
├── examples/
│   ├── sglang_multiturn/
│   │   ├── config/
│   │   │   ├── tool_config/
│   │   │   │   ├── my_web_search_tool_config.yaml        # 自定义工具配置化描述


TOOL_CONFIG = "${ROOT_DIR}/examples/sglang_multiturn/config/tool_config/my_web_search_tool.py"
REWARD_PATH = "${ROOT_DIR}/verl/utils/reward_score/my_reward_function.py"
python3 -m verl.trainer.main_ppo \ 
    ......   
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=3 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG} \  # 配置步骤（3）的工具
    actor_rollout_ref.rollout.multi_turn.format=hermes \                    # 配置步骤（4）的ToolParser
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    ......   

    reward_model.reward_manager=my_web_search \              #配置步骤（6）自定义的RewardMeneger
    custom_reward_function.path=${REWARD_PATH} \             #配置步骤（7）自定义的reward fucntion 文件
    custom_reward_function.name=compute_score \              #配置步骤（7）自定义的reward fucntion 函数名
2. mlsys https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main
