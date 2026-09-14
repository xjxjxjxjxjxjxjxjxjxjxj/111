# OpenCode GitHub步骤（占位模板）

当前目录未发现用户所说的GitHub流程文件。收到该文件后，OpenCode必须优先按用户文件执行并更新本页；GitHub相关操作始终由OpenCode负责。

默认安全流程：

1. 确认`workflow_config.json -> github.repository_dir`和`push_branch`已由用户填写。
2. 把`READY_FOR_OPENCODE.json`指定的新dog发布包复制到GitHub仓库，不提交视频、评价原文中的隐私信息、密钥或设备地址。
3. 运行发布包的全部测试，检查`git diff`只包含本轮代码、配置、测试、原因报告摘要。
4. 问题修复提交信息建议：`dog12: 修复返程主线误锁（来自dog11实测）`。
5. 用户评价`1 完赛`时，给本次真正运行的旧版本写明路牌颜色与实际耗时，例如：`dog11 完赛 黄牌 88.0秒 2026-09-13 18:30 CST`，并创建带注释标签`dog11-complete-yellow-20260913-183000`。
6. Codex随后产生的`dog12`是提速候选，提交信息写`dog12: 基于dog11完赛视频提速（待实测）`，不得沿用完赛标签。
7. push成功并取得commit SHA后，才调用控制器的`ack-deploy`。
