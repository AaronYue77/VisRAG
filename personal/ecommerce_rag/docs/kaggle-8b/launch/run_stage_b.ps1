param(
    [ValidateSet('chat', 'ask', 'serve', 'check', 'search', 'dev', 'test')]
    [string]$Command = 'chat',
    [ValidateSet('auto', 'bm25', 'dense', 'hybrid', 'rerank')]
    [string]$Mode = 'auto',
    [string]$Question = '',
    [string[]]$Sku = @(),
    [switch]$Retry,
    [int]$Port = 8767,
    [ValidateSet('full')][string]$Corpus = 'full',
    [string]$Domain = '', [string]$Type = '', [switch]$Visual, [switch]$LegacyEvidence,
    [ValidateSet('8b')][string]$Model = '8b', [string]$ModelDir = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
$StageCode = Join-Path $TaskRoot 'upstream\VisRAG\personal\ecommerce_rag'
if ($Command -in @('dev', 'test')) {
    & $ProjectPython (Join-Path $StageCode 'full_evaluate.py') retrieval --split $Command
} else {
    & (Join-Path $PSScriptRoot 'run_full_rag.ps1') -Stage B -Command $Command -Mode $Mode -Question $Question -Sku $Sku -Domain $Domain -Type $Type -Port $Port -Visual:$Visual -Retry:$Retry -LegacyEvidence:$LegacyEvidence -Model $Model -ModelDir $ModelDir
}
if ($LASTEXITCODE -ne 0) { throw ('Stage B failed. Exit code: ' + $LASTEXITCODE) }
