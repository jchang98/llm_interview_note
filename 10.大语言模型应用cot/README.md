# 10.大语言模型应用

## Fast Look

**应用链路**：`Prompt Template → LLM → Parser/Validator → Chain 或 Agent → Memory/Retriever/Tool → Observability`；CoT 是提示策略，LangChain 是组件编排框架，二者不等价。

#### Zero-shot CoT、Few-shot CoT 与 Self-Consistency

- **Zero-shot CoT**：用“让我们一步步思考”等提示诱导中间推理；**Few-shot CoT**：提供带推理步骤的示例；**Self-Consistency**：多次采样不同推理链并投票答案。
- **与直接回答的区别**：显式引入中间步骤或多条路径，以换取复杂推理表现。
- **优点**：数学、逻辑和多步任务中常提升正确率；Self-Consistency 可降低单条轨迹偶然错误。
- **缺点/注意点**：增加 token、延迟和成本；可见推理文本不等于真实内部因果过程，也可能泄露不必要信息。

#### Prompt Template、System Prompt 与 Output Parser

- **Prompt Template 描述**：参数化组织指令、上下文和示例；**System Prompt** 定义高优先级角色/边界；**Output Parser** 将输出约束/解析为 JSON、字段或结构化对象。
- **与自由文本调用的区别**：把输入输出协议显式化、可测试化。
- **优点**：便于复用、版本控制和下游系统集成。
- **缺点/注意点**：模板本身会注入偏差；Parser 必须处理不合法 JSON、截断和提示注入，不能假设模型永远遵循格式。

#### Chain、Router 与 Runnable

- **Chain 描述**：将多个步骤串联；**Router** 按输入选择不同链/模型；**Runnable** 是可组合、可执行组件抽象。
- **与单次 LLM 调用的区别**：显式表达多阶段数据流与分支。
- **优点**：模块复用、单元测试和观测更清晰。
- **缺点/注意点**：简单任务过度编排会增加依赖、状态与调试成本。

#### Retriever、Memory 与 Tool Calling

- **Retriever**：为链提供外部文档；**Memory**：维护对话/任务状态；**Tool Calling**：让模型选择结构化函数/API。
- **与 CoT 的区别**：CoT 只改变推理提示，这三者真正扩展可获取信息和可执行能力。
- **优点**：可构建知识问答、多轮协作和业务自动化。
- **缺点/注意点**：Memory 会膨胀或污染；Tool 参数必须 schema 校验与权限隔离；Retriever 质量决定事实依据。

### 10.1 思维链提示

[1.思维链（cot）](/10.大语言模型应用/1.思维链（cot）/1.思维链（cot）.md "1.思维链（cot）")

### 10.2 LangChain框架

[1.langchain](/10.大语言模型应用/1.langchain/1.langchain.md "1.langchain")
