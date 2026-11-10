param(
    [ValidateSet('chat', 'ask', 'serve', 'check', 'search', 'dev', 'test')]
    [string]$Command = 'chat',
    [ValidateSet('auto', 'bm25', 'dense', 'hybrid', 'bge', 'text_rrf', 'visual', 'multimodal_rrf', 'rerank', 'retry')]
    [string]$Mode = 'auto',
    [string]$Question = '',
    [string[]]$Sku = @(),
    [switch]$Retry,
    [int]$Port = 8767,
    [ValidateSet('full', 'sample')][string]$Corpus = 'full',
    [string]$Domain = '', [string]$Type = '', [switch]$Visual, [switch]$LegacyEvidence,
    [ValidateSet('8b', '2b')][string]$Model = '8b', [string]$ModelDir = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
$StageCode = Join-Path $TaskRoot 'upstream\VisRAG\personal\ecommerce_rag'
if ($Corpus -eq 'full') {
    if ($Command -in @('dev', 'test')) {
        & $ProjectPython (Join-Path $StageCode 'full_evaluate.py') retrieval --split $Command
    } else {
        if ($Mode -notin @('auto', 'bm25', 'dense', 'hybrid', 'rerank')) {
            throw 'For full metadata use auto/bm25/dense/hybrid/rerank and optional -Visual. Old modes remain with -Corpus sample.'
        }
        & (Join-Path $PSScriptRoot 'run_full_rag.ps1') -Stage B -Command $Command -Mode $Mode -Question $Question -Sku $Sku -Domain $Domain -Type $Type -Port $Port -Visual:$Visual -Retry:$Retry -LegacyEvidence:$LegacyEvidence -Model $Model -ModelDir $ModelDir
    }
    if ($LASTEXITCODE -ne 0) { throw ('Full Stage B failed. Exit code: ' + $LASTEXITCODE) }
    return
}
if ($Command -in @('dev', 'test')) {
    $TaskArguments = @((Join-Path $StageCode 'b_evaluate.py'), 'run', '--split', $Command)
    if ($Mode -ne 'auto') { $TaskArguments += @('--mode', $Mode) }
} else {
    if ($Mode -eq 'retry') { throw 'Use -Retry to enable bounded retry with an actual retrieval mode.' }
    $TaskArguments = @((Join-Path $StageCode 'b_app.py'), $Command)
    if ($Mode -ne 'auto') { $TaskArguments += @('--mode', $Mode) }
    if ($Retry) { $TaskArguments += '--retry' }
    if ($Command -eq 'ask') {
        if ([string]::IsNullOrWhiteSpace($Question)) { throw 'Please provide -Question.' }
        $TaskArguments += @('--question', $Question)
    }
    foreach ($ProductSku in $Sku) { $TaskArguments += @('--sku', $ProductSku) }
    if ($Command -eq 'serve') { $TaskArguments += @('--port', $Port) }
}
& $ProjectPython @TaskArguments
if ($LASTEXITCODE -ne 0) { throw ('Stage B failed. Exit code: ' + $LASTEXITCODE) }
