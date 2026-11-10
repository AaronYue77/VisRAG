param(
    [ValidateSet('cloud-status', 'setup', 'prepare', 'train', 'evaluate', 'diagnose', 'extended')]
    [string]$Command = 'cloud-status',
    [string]$RunName = 'stage-c-lora-20261109',
    [string]$KaggleKernel = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
if ($Command -eq 'cloud-status') {
    if (-not $KaggleKernel) {
        $Active = Join-Path $TaskRoot 'data/ecommerce_rag/stage-c-active.json'
        if (Test-Path -LiteralPath $Active) {
            $KaggleKernel = (Get-Content -LiteralPath $Active -Raw -Encoding UTF8 | ConvertFrom-Json).kernel
        } else {
            $KaggleKernel = 'aaronyue77/abo-rag-qwen3-vl-8b-formal'
        }
    }
    $Kaggle = Join-Path $TaskRoot '.tools/kaggle-cli/Scripts/kaggle.exe'
    & $Kaggle kernels status $KaggleKernel
    if ($LASTEXITCODE -ne 0) { throw 'Kaggle 8B status query failed.' }
    return
}
$TrainingEnv = Join-Path $TaskRoot '.venv-stage-c'
$TrainingPython = Join-Path $TrainingEnv 'Scripts\python.exe'
if ($Command -eq 'setup') {
    if (-not (Test-Path -LiteralPath $TrainingPython)) {
        & $ProjectUv venv $TrainingEnv --python $ProjectPython
        if ($LASTEXITCODE -ne 0) { throw 'Stage C environment creation failed.' }
    }
    # Reuse the pinned CUDA runtime without installing training packages into the inference environment.
    $RuntimeLink = Join-Path $TrainingEnv 'Lib\site-packages\stage_a_runtime.pth'
    [IO.File]::WriteAllText($RuntimeLink, '..\..\..\.venv\Lib\site-packages' + [Environment]::NewLine)
    & $ProjectUv pip install --python $TrainingPython --no-deps 'peft==0.21.0'
    if ($LASTEXITCODE -ne 0) { throw 'Stage C dependency setup failed.' }
    & $TrainingPython -c 'import peft, torch, transformers; print(peft.__version__, torch.__version__, transformers.__version__, torch.cuda.is_available())'
} else {
    if (-not (Test-Path -LiteralPath $TrainingPython)) { throw 'Run with -Command setup first.' }
    $env:HF_HUB_OFFLINE = '1'
    $Runner = Join-Path $TaskRoot 'upstream\VisRAG\personal\ecommerce_rag\stage_c.py'
    if ($Command -in @('diagnose', 'extended')) {
        $Entry = if ($Command -eq 'diagnose') { 'c_diagnostics.py' } else { 'c_extended.py' }
        if (-not $PSBoundParameters.ContainsKey('RunName')) { $RunName = "stage-c-$Command-" + (Get-Date -Format 'yyyyMMdd-HHmmss') }
        & $TrainingPython (Join-Path (Split-Path -Parent $Runner) $Entry) --run-name $RunName
    } else {
        & $TrainingPython $Runner $Command --run-name $RunName
    }
}
if ($LASTEXITCODE -ne 0) { throw ('Stage C failed. Exit code: ' + $LASTEXITCODE) }
