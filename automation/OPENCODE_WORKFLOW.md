# OpenCode 机械狗闭环工作流

## 职责边界

OpenCode负责：部署代码到机械狗、启动程序、下载录像与遥测、询问用户评价、接收Codex发布包、覆盖部署、按GitHub流程提交和推送。

Codex负责：读取评价、视频与遥测，给出有时间点证据的原因，修改独立工作副本，运行测试，把`dogN`严格增加为`dogN+1`，生成诊断报告和发布清单。评价1时优化完赛速度，评价2时修复问题。Codex不执行GitHub推送，也不直接连接机械狗。

## 用户输入“启动”后的步骤

初始化时先运行Codex自检；配置值`auto`会先查PATH，再自动寻找Codex Desktop内置CLI，因此不要求npm全局安装：

```powershell
python automation/workflow_controller.py doctor --config D:\Desktop\dog11_closed_loop\XGO-AutoLoop\workflow_config.json
```

1. 只在用户明确输入完整词语`启动`时调用：

   ```powershell
   python automation/workflow_controller.py prepare-run --config D:\Desktop\dog11_closed_loop\XGO-AutoLoop\workflow_config.json --trigger 启动
   ```

2. 读取命令返回的`run_request`。先把`current_project`原子部署到机械狗的临时目录，校验SHA-256清单后再替换`remote_project_dir`。保留上一版本作为可回退副本。
3. 在机械狗上启动同一个控制程序并录像。不要另开ffmpeg或第二个OpenCV进程抢摄像头。路牌测试命令格式：

   ```text
   python3 sign_line_closed_loop.py --mode run --start-delay 5 --max-seconds <maximum_run_seconds> --record-video <remote_video> --telemetry <remote_telemetry> --record-fps 20
   ```

4. 程序结束或达到安全超时后，下载视频、JSONL遥测和终端日志到`D:\Desktop\dog11_closed_loop\XGO-AutoLoop\downloads\<run_id>`。校验下载文件非空后才删除机械狗端临时视频。
5. 先分类停止来源：状态机出现路牌测试的`YELLOW/BLACK COMPLETE`或完整赛程“完全进入出发区”证据时为`goal_complete`，这种自行停车不算强制终止；用户在OpenCode输入停止为`opencode_stop`。超时、中断、安全停车、无完成标志退出、崩溃、被杀死均为非正常完成并自动评价2。
6. 只有`goal_complete`才让用户选择：

   - `1 完赛`：完成所有比赛目标；
   - `2 有问题`：再让用户选择或填写问题，例如：路牌误识别、20 cm偏差、巡线偏移、黑牌动作、黄牌绕行、抓球、投杯、返程、其他。

7. 评价1必须从终端/遥测确认实际路牌为黄或黑，并记录实际总耗时。无法确认颜色时不得登记完赛。把评价和完整备注交给控制器。示例：

   ```powershell
   python automation/workflow_controller.py ingest --config D:\Desktop\dog11_closed_loop\XGO-AutoLoop\workflow_config.json --video <视频> --telemetry <JSONL> --terminal-log <日志> --rating 2 --comment "黄牌在约25 cm提前报警" --termination goal_complete --sign-color yellow --elapsed-seconds 45.2
   ```

8. 评价1和2都会由Codex处理。评价1的目标是分析分段耗时、在保持完赛行为与安全条件的前提下提速；评价2的目标是解释并修复问题。成功后读取`D:\Desktop\dog11_closed_loop\XGO-AutoLoop\state\READY_FOR_OPENCODE.json`，部署其中的`release_path`，覆盖机械狗上一版本并再次校验文件清单。
9. GitHub操作严格按`OPENCODE_GITHUB.md`以及用户后续提供的GitHub流程文件执行。OpenCode负责commit、push和tag，Codex不得代替。
10. 部署和GitHub均成功后调用：

   ```powershell
   python automation/workflow_controller.py ack-deploy --config D:\Desktop\dog11_closed_loop\XGO-AutoLoop\workflow_config.json --release <READY清单中的release_path> --github-commit <真实SHA>
   ```

   然后询问：`新版本已部署并上传GitHub，是否启动？准备好后请输入“启动”。`
11. 评价为1时，OpenCode先在GitHub给实际跑完的旧版本写入`完赛`、黄/黑路牌、实际耗时、北京时间和带注释标签；Codex产生的新版本只能标为`提速候选，待实测`，不能提前标记完赛。

## 禁止事项

- 不使用`--auto`或绕过审批参数启动OpenCode。
- 不把SSH密码、GitHub token或私钥写入JSON、日志、评价文件或提交。
- 下载视频未校验前不得删除机械狗端录像。
- Codex测试失败、未写原因/提速报告或dog序号未恰好加一时不得部署。
- 完整比赛入口仍有安全锁时，只能运行当前获准的独立测试入口。
