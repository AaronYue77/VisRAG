param(
    [ValidateSet('chat', 'ask', 'serve', 'check', 'search', 'index', 'dev', 'test', 'grid')]
    [string]$Command = 'chat',
    [ValidateSet('full', 'enhanced', 'baseline')]
    [string]$Mode = 'full',
    [string]$Question = '',
    [string[]]$Sku = @(),
    [string]$RunName = ('a-manual-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [int]$Port = 8765,
    [string]$Domain = '',
    [string]$Type = '',
    [switch]$LegacyEvidence,
    [ValidateSet('8b', '2b')][string]$Model = '8b', [string]$ModelDir = ''
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
$StageCode = Join-Path $TaskRoot 'upstream\VisRAG\personal\ecommerce_rag'
if ($Mode -eq 'full') {
    if ($Command -in @('dev', 'test')) {
        & $ProjectPython (Join-Path $StageCode 'full_evaluate.py') qa --split $Command --stage A
    } elseif ($Command -eq 'index') {
        & $ProjectPython (Join-Path $StageCode 'full_catalog.py')
    } elseif ($Command -eq 'grid') {
        throw 'The old grid is retained with -Mode enhanced. Full retrieval comparisons use run_stage_b.ps1 -Command dev.'
    } else {
        & (Join-Path $PSScriptRoot 'run_full_rag.ps1') -Stage A -Command $Command -Question $Question -Sku $Sku -Domain $Domain -Type $Type -Port $Port -LegacyEvidence:$LegacyEvidence -Model $Model -ModelDir $ModelDir
    }
    if ($LASTEXITCODE -ne 0) { throw ('Full Stage A failed. Exit code: ' + $LASTEXITCODE) }
    return
}
$AppFile = Join-Path $StageCode 'stage_a_app.py'
if ($Mode -eq 'enhanced') { $AppFile = Join-Path $StageCode 'dialogue_app.py' }
$RunnerFile = Join-Path $StageCode 'stage_a.py'

if ($Command -in @('ask', 'chat')) {
    if ($Command -eq 'ask' -and [string]::IsNullOrWhiteSpace($Question)) { throw 'Please provide a product question with -Question.' }
    $AppArguments = @($AppFile, $Command)
    if ($Command -eq 'ask') { $AppArguments += @('--question', $Question) }
    foreach ($ProductSku in $Sku) { $AppArguments += @('--sku', $ProductSku) }
    & $ProjectPython @AppArguments
} elseif ($Command -eq 'check') {
    if ($Mode -eq 'baseline') { throw 'Use enhanced mode for startup check; original baseline entry is preserved.' }
    & $ProjectPython $AppFile check
} elseif ($Command -eq 'serve') {
    & $ProjectPython $AppFile serve --port $Port
} elseif ($Command -eq 'index') {
    & $ProjectPython $RunnerFile index
} elseif ($Command -eq 'grid') {
    & $ProjectPython $RunnerFile grid --run-name $RunName
} else {
    $SettingsPath = Join-Path $TaskRoot 'data\ecommerce_rag\abo-cards-v1\frozen-test-config.json'
    $FrozenSettings = (Get-Content -LiteralPath $SettingsPath -Raw -Encoding UTF8 | ConvertFrom-Json).settings
    & $ProjectPython $RunnerFile run --split $Command --run-name $RunName --strategy $FrozenSettings.strategy --overlap $FrozenSettings.overlap --top-k $FrozenSettings.top_k --max-pixels $FrozenSettings.max_pixels
}
if ($LASTEXITCODE -ne 0) { throw ('Stage A command failed. Exit code: ' + $LASTEXITCODE) }
