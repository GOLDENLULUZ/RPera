只允许故事总结 Agent 修改当前存档的 story_summary.json。old_text 必须在原文中唯一出现，替换后的完整文件必须是合法 JSON 摘要，且不能包含当前尚未完成的 turn；校验通过后才原子写入。
