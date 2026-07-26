# 06.推理

## Fast Look

**请求链路**：`tokenize → Prefill → KV Cache → Decode → 调度/批处理 → 采样 → 服务输出`；优化应先区分 TTFT、TPOT、吞吐、显存与尾延迟，而非只看单一 benchmark。

#### Prefill 与 Decode

- **Prefill 描述**：并行处理完整 prompt、建立 KV Cache，通常计算密集；**Decode 描述**：逐 token 读取历史 KV，通常带宽/延迟敏感。
- **与一次性前向的区别**：自回归生成将首轮与后续 token 拆成资源特性不同的阶段。
- **优点**：分阶段观测可针对性优化 TTFT 与 TPOT。
- **缺点/注意点**：长 prompt 可能主导 TTFT，长输出可能主导成本；不能用单一 tokens/s 描述所有体验。

#### Static、Dynamic 与 Continuous/In-flight Batching

- **Static batching**：固定凑 batch；**Dynamic batching**：等待窗口内合并请求；**Continuous batching**：每个 decode step 都插入/移除请求。
- **与静态 batch 的区别**：Continuous batching 不必等待同批全部请求结束。
- **优点**：提高 GPU 利用率和总体吞吐，连续批处理对长度不均请求尤其有效。
- **缺点/注意点**：调度更复杂，过度追求吞吐会牺牲 TTFT/尾延迟；需设置公平性与最大等待时间。

#### KV Cache

- **描述**：缓存每层历史 token 的 K/V；新 token 只计算自身 Q/K/V，再读取历史缓存。
- **与不缓存的区别**：避免每生成一个 token 都重算整个历史前缀。
- **优点**：显著降低自回归重复计算，是在线生成的基础优化。
- **缺点/注意点**：显存随上下文、层数和并发线性增长；需结合 GQA、分页缓存和调度处理容量问题。

#### PagedAttention 与 vLLM

- **PagedAttention 描述**：将 KV Cache 切为固定 block，用页表映射逻辑 token 到物理 block，减少碎片并支持共享前缀。
- **vLLM 描述**：以 PagedAttention 和连续批处理为核心的推理/服务引擎。
- **与连续 KV buffer 的区别**：不需为每个请求预分配连续最大长度缓存。
- **优点**：高并发下 KV 利用率高，吞吐表现强。
- **缺点/注意点**：block size、预留空间和调度策略影响碎片/性能；模型特性和硬件支持需实测。

#### TGI、FasterTransformer、TensorRT-LLM、CTranslate2、DeepSpeed-MII、Ray Serve 与 MLC LLM

- **TGI**：Hugging Face 服务端，集成连续批处理与多模型生态；**优点**是易部署，**注意**是版本/硬件特性需核对。
- **FasterTransformer**：NVIDIA 高性能 Transformer 内核库；相对通用 PyTorch 更靠近 CUDA 优化，性能高但集成灵活性较低。
- **TensorRT-LLM**：基于 TensorRT 的图、内核和量化优化；在 NVIDIA GPU 上性能强，但构建与硬件绑定更强。
- **CTranslate2、DeepSpeed-MII、Ray Serve、MLC LLM**：分别偏转换推理、DeepSpeed 服务、分布式服务编排和端侧编译部署；按目标硬件与运维边界选择。

#### 量化、稀疏与蒸馏

- **量化**：以 INT8/INT4 等低比特表示权重或激活；**优点**是降显存/带宽，**缺点**是精度和 kernel 支持风险。
- **稀疏**：跳过零或结构化无效计算；**优点**是理论降 FLOPs，**缺点**是硬件不支持时不一定更快。
- **蒸馏**：用教师软标签/轨迹训练小学生；**优点**是从根源降成本，**缺点**是能力上限受教师和数据限制。

#### Speculative Decoding

- **描述**：小 draft model 先提出多个 token，大 target model 并行验证并接受/拒绝。
- **与普通 Decode 的区别**：不改变 target 分布，但用草稿模型减少 target 的串行步数。
- **优点**：在高接受率时显著降低生成延迟。
- **缺点/注意点**：需额外模型与调度；接受率低、短输出或 target 很小时收益有限。

#### Greedy、Beam Search、Temperature、Top-k、Top-p 与 Repetition Penalty

- **Greedy**：每步最大概率，稳定但多样性低；**Beam Search** 保留多条候选，适合受控生成但成本高且易产生通用答案。
- **Temperature/Top-k/Top-p**：控制采样随机性与候选集，详见第 02 章。
- **Repetition Penalty**：降低已出现 token 的 logit；**优点**是减少复读，**缺点**是过强会破坏专有名词、代码或必要重复。

### 6.1 推理框架

[0.llm推理框架简单总结](/06.推理/0.llm推理框架简单总结/0.llm推理框架简单总结.md "0.llm推理框架简单总结")

[1.vllm](/06.推理/1.vllm/1.vllm.md "1.vllm")

[2.text_generation\_inference](/06.推理/2.text_generation_inference/2.text_generation_inference.md "2.text_generation_inference")

[3.faster_transformer](/06.推理/3.faster_transformer/3.faster_transformer.md "3.faster_transformer")

[4.trt_llm](/06.推理/4.trt_llm/4.trt_llm.md "4.trt_llm")

### 6.2 推理优化技术

[llm推理优化技术](/06.推理/llm推理优化技术/llm推理优化技术.md "llm推理优化技术")

### 6.3 量化


### 6.4 vLLM


### 6.5 一些题目

[1.推理](/06.推理/1.推理/1.推理.md "1.推理")
