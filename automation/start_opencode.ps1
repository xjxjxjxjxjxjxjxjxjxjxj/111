$ErrorActionPreference = 'Stop'
$workflowRoot = 'D:\Desktop\dog11_closed_loop\XGO-AutoLoop'
$config = Join-Path $workflowRoot 'workflow_config.json'

if (-not (Test-Path -LiteralPath $config)) {
    throw "请先填写配置文件：$config"
}

Set-Location -LiteralPath $workflowRoot
opencode . --prompt '请先完整读取 OPENCODE_WORKFLOW.md、OPENCODE_GITHUB.md 和 workflow_config.json。你只负责部署、启动、录像交接、询问评价和GitHub，不直接修改机械狗算法。等待我输入“启动”。'
