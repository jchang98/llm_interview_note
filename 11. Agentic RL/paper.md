# 富文本反馈，不仅仅通过标量奖励 （忽略了编译器报错、测试用例输出等富文本反馈）

1. Reinforcement Learning via Self-Distillation (SDPO)
- 通过将包含了反馈信息的 Teacher 的 token 级概率分布蒸馏回原始 Student 策略，SDPO 实现了无需外部教师模型、无需显式奖励模型的稠密信用分配。
- Student (当前策略) ：在未看到反馈f 之前生成的y ，其分布为pai（y｜x） 。
- self-teacher(事后策略)：在看到反馈 f 之后，重新评估y的合理性，其分布为 pai（y| x,f）
- 使用kl散度拟合两个分布