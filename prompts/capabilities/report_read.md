使用报告的 name 和 id 读取当前回合已完成的专业 Agent 报告；结果按字段路径标注，包含报告原文、来源 Agent、来源任务及会话引用。文本报告的 content 文本块是报告原文，不带外层 JSON 包装产生的转义；合规审核报告的 content 则展开 approved 和 reason 字段，reason 按原文展示。多份报告按引用顺序分别提供。
