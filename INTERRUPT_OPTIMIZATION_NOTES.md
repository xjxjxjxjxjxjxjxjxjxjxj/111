# CPU1 轮询可靠版说明

## 为什么关闭中断唤醒

实车已经确认此前两个 `InterruptSafe_20sStop` 包均无法让电机启动。两个包只在 VOFA 开关上不同，共同的新增运行路径是 CPU1 IDLE 与 GPSR00 唤醒，因此本版不再依赖该链路。

默认配置为：

```c
#define LOW_COMPUTE_CPU1_INTERRUPT_WAKE (0)
```

编译后不会包含 CPU1 IDLE、GPSR00 触发、GPSR00 唤醒 ISR、4 ms 状态检查或强制 RUN 代码。

## 实际运行路径

1. CPU0 沿用原流程等待相机帧完成、复制图像并计算阈值。
2. CPU0 设置 `frame_ready=1`。
3. CPU1 在原主循环中连续检查 `frame_ready`，执行原版 `get_xian()`。
4. CPU1 发布识别结果并设置 `result_ready=1`。
5. CPU0 继续原来的 PID 与电机控制。

这条路径与原 NoRing 15 的双核握手一致，不需要软件中断唤醒。代价是 CPU1 等待新帧时仍会空转，空闲算力和功耗不如中断版，但可靠性优先。

## 仍保留的保护

- 相机等待超过 100 ms 后清零两个 PWM。
- CPU1 结果等待超过 50 ms 后清零两个 PWM并复位 PID。
- t2500 当前的 20 秒计时开关为关闭状态；若以后手动启用，计时保护逻辑仍保留。

50 ms 安全停车不会结束等待；CPU1 后续发布结果后，CPU0仍可继续运行。若轮询版仍从不产生电机命令，应继续检查相机完成标志、`frame_ready/result_ready` 和赛道线有效判定，而不是再次启用 GPSR00。
