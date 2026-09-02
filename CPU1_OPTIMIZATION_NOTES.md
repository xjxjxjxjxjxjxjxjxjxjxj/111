# CPU1 低风险优化说明

## 修改范围

本版只修改 `user/cpu1_main.c`，并保持以下内容不变：

- 屏幕关闭；本交付版另在 CPU0 低频启用 VOFA，不改变本文件的 CPU1 优化。
- CPU1 使用原 NoRing 15 的连续轮询，`LOW_COMPUTE_CPU1_INTERRUPT_WAKE=0`。
- 电机引脚、方向、PID、圆环状态机、相机、DMA 和计时开关保持 t2500 基线不变（计时默认关闭）。
- `frame_ready/result_ready` 握手、图像快照和 50 ms 安全停车不变。

## 优化 1：只优化 CPU1 文件

`CPU1_PROCESSING_OPTIMIZE=1` 时：

- TASKING 使用 `#pragma optimize 2` 和 `#pragma tradeoff 0`。
- AURIX GCC 使用 `#pragma GCC optimize ("O2")`。
- 这些设置仅覆盖 `cpu1_main.c`，不会改变 CPU0、ISR、电机或停车代码的编译方式。

把宏改成 0 可以关闭文件级 O2，便于快速回退排查。

## 优化 2：等价减少图像扫描计算

原梯度判断包含变量整数除法：

```c
((first + second) * 10) / (abs(first - second) + 1) < 400
```

现在使用严格等价的乘法比较：

```c
(first + second) * 10 < 400 * (abs(first - second) + 1)
```

分母始终大于 0，输入是 0～255 的图像像素，因此两种判断完全等价且不会溢出。全部 65,536 种像素组合已穷举比较，差异为 0。

纵向扫描中，旧代码每轮重新读取 `current` 和 `above`。新代码把本轮的 `above` 直接作为下一轮的 `current`，并用指针每次向上一行移动。循环范围、短路顺序、首次命中的行号和无边界默认值均不改变。最坏情况下纵向读取由 44,368 次降为 22,372 次，减少约 49.6%。

## 已完成检查

- 梯度判断 65,536 种输入组合：0 个差异。
- 309 个边界与随机噪声测试帧、58,092 列纵向扫描：0 个差异。
- 在验证副本中仅屏蔽原工程的 TASKING 专用内存段指令后，AURIX GCC 在整体 O0 条件下以 `-Wall -Wextra -Werror` 编译通过，确认 CPU1 源文件局部 O2 生效。
- 优化目标代码中未发现整数除法指令。
- `cpu0_main.c`、`isr.c`、`isr_config.h` 和 `cpu0_main.h` 与实车可运行基线逐文件哈希一致。

TASKING 编译器在本机禁止脱离 AURIX Development Studio 独立运行，因此交付后仍应在 AURIX Development Studio 中执行 Clean + Build。源码中的 TASKING 优化指令已按该编译器内置帮助支持的语法设置。
