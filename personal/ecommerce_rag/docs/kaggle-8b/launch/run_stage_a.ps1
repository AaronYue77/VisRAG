param(
    [ValidateSet('chat', 'ask', 'serve', 'check', 'search', 'index', 'dev', 'test')]
    [string]$Command = 'chat',
    [ValidateSet('full')][string]$Mode = 'full',
    [string]$Question = '',
    [string[]]$Sku = @(),
    [int]$Port = 8765,
    [string]$Domain = '',
    [string]$Type = '',
    [switch]$LegacyEvidence,
    [ValidateSet('8b')][string]$Model = '8b',
    [string]$ModelDir = ''
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
$StageCode = Join-Path $TaskRoot 'upstream/VisRAG/personal/ecommerce_rag'
if ($Command -in @('dev', 'test')) {
    $RunnerArguments = @((Join-Path $StageCode 'full_evaluate.py'), 'qa', '--split', $Command, '--stage', 'A')
    if ($ModelDir) { $RunnerArguments += @('--model-dir', $ModelDir) }
    & $ProjectPython @RunnerArguments
} elseif ($Command -eq 'index') {
    & $ProjectPython (Join-Path $StageCode 'full_catalog.py')
} else {
    & (Join-Path $PSScriptRoot 'run_full_rag.ps1') -Stage A -Command $Command -Question $Question -Sku $Sku -Domain $Domain -Type $Type -Port $Port -LegacyEvidence:$LegacyEvidence -Model $Model -ModelDir $ModelDir
}
if ($LASTEXITCODE -ne 0) { throw ('Stage A failed. Exit code: ' + $LASTEXITCODE) }
