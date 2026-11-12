param(
    [ValidateSet('A', 'B')][string]$Stage = 'B',
    [ValidateSet('chat', 'ask', 'serve', 'check', 'search', 'build')][string]$Command = 'chat',
    [ValidateSet('auto', 'bm25', 'dense', 'hybrid', 'rerank')][string]$Mode = 'auto',
    [string]$Question = '', [string[]]$Sku = @(), [string]$Domain = '', [string]$Type = '',
    [switch]$Visual, [switch]$Retry, [switch]$LegacyEvidence, [int]$Port = 8768,
    [ValidateSet('8b')][string]$Model = '8b', [string]$ModelDir = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
$FullCode = Join-Path $TaskRoot 'upstream\VisRAG\personal\ecommerce_rag'
if ($Command -eq 'build') {
    & $ProjectPython (Join-Path $FullCode 'full_catalog.py')
    if ($LASTEXITCODE -ne 0) { throw 'Full catalog build failed.' }
    if ($Stage -eq 'B') { & $ProjectPython (Join-Path $FullCode 'full_dense.py') }
} else {
    $FullArguments = @((Join-Path $FullCode 'full_app.py'), $Command, '--stage', $Stage, '--model', $Model)
    if ($ModelDir) { $FullArguments += @('--model-dir', $ModelDir) }
    if ($Mode -ne 'auto') { $FullArguments += @('--mode', $Mode) }
    if ($Command -in @('ask', 'search')) {
        if ([string]::IsNullOrWhiteSpace($Question)) { throw 'Please provide -Question.' }
        $FullArguments += @('--question', $Question)
    }
    foreach ($ProductSku in $Sku) { $FullArguments += @('--sku', $ProductSku) }
    if ($Domain) { $FullArguments += @('--domain', $Domain) }
    if ($Type) { $FullArguments += @('--type', $Type) }
    if ($Visual) { $FullArguments += '--visual' }
    if ($Retry) { $FullArguments += '--retry' }
    if ($LegacyEvidence) { $FullArguments += '--legacy-evidence' }
    if ($Command -eq 'serve') { $FullArguments += @('--port', $Port) }
    & $ProjectPython @FullArguments
}
if ($LASTEXITCODE -ne 0) { throw ('Full RAG command failed. Exit code: ' + $LASTEXITCODE) }
