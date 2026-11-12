param(
    [ValidateSet('cloud-status')][string]$Command = 'cloud-status',
    [string]$KaggleKernel = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
if (-not $KaggleKernel) {
    $Active = Join-Path $TaskRoot 'data/ecommerce_rag/stage-c-active.json'
    if (Test-Path -LiteralPath $Active) {
        $KaggleKernel = (Get-Content -LiteralPath $Active -Raw -Encoding UTF8 | ConvertFrom-Json).kernel
    } else {
        $KaggleKernel = 'aaronyue77/abo-rag-8b-120-example-candidate'
    }
}
$Kaggle = Join-Path $TaskRoot '.tools/kaggle-cli/Scripts/kaggle.exe'
& $Kaggle kernels status $KaggleKernel
if ($LASTEXITCODE -ne 0) { throw 'Kaggle 8B status query failed.' }
