param(
    [ValidateSet('ask', 'serve', 'index', 'dev', 'test', 'grid')]
    [string]$Command = 'ask',
    [string]$Question = '',
    [string[]]$Sku = @(),
    [string]$RunName = ('a-manual-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [int]$Port = 8765
)

$ErrorActionPreference = 'Stop'
if ($env:DOC_VQA_WORKSPACE) { $TaskRoot = $env:DOC_VQA_WORKSPACE } else { $TaskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path }
$env:DOC_VQA_WORKSPACE = $TaskRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$ProjectPython = Join-Path $TaskRoot '.venv\Scripts\python.exe'
$StageCode = Join-Path $TaskRoot 'upstream\VisRAG\personal\ecommerce_rag'
$AppFile = Join-Path $StageCode 'stage_a_app.py'
$RunnerFile = Join-Path $StageCode 'stage_a.py'

if ($Command -eq 'ask') {
    if ([string]::IsNullOrWhiteSpace($Question)) { throw '请使用 -Question 输入商品问题。' }
    $AppArguments = @($AppFile, 'ask', '--question', $Question)
    foreach ($ProductSku in $Sku) { $AppArguments += @('--sku', $ProductSku) }
    & $ProjectPython @AppArguments
} elseif ($Command -eq 'serve') {
    & $ProjectPython $AppFile serve --port $Port
} elseif ($Command -eq 'index') {
    & $ProjectPython $RunnerFile index
} elseif ($Command -eq 'grid') {
    & $ProjectPython $RunnerFile grid --run-name $RunName
} else {
    $SettingsPath = Join-Path $TaskRoot 'data\ecommerce_rag\abo-cards-v1\frozen-test-config.json'
    $FrozenSettings = (Get-Content -LiteralPath $SettingsPath -Raw | ConvertFrom-Json).settings
    & $ProjectPython $RunnerFile run --split $Command --run-name $RunName --strategy $FrozenSettings.strategy --overlap $FrozenSettings.overlap --top-k $FrozenSettings.top_k --max-pixels $FrozenSettings.max_pixels
}
if ($LASTEXITCODE -ne 0) { throw ('A阶段命令失败，退出码：' + $LASTEXITCODE) }
