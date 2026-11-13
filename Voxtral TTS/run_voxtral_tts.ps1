param(
    [switch]$ReadOneToThousand
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelsDir = Join-Path $projectRoot 'models'
$outputDir = Join-Path $projectRoot 'outputs'

New-Item -ItemType Directory -Force -Path $modelsDir, $outputDir | Out-Null

$env:CRISPASR_CACHE_DIR = $modelsDir
$env:CRISPASR_MODELS_DIR = $modelsDir

$binary = Join-Path $projectRoot 'crispasr.exe'
if (-not (Test-Path $binary)) {
    $binary = (Get-ChildItem -Path $projectRoot -Filter 'crispasr.exe' -Recurse -File |
        Select-Object -First 1 -ExpandProperty FullName)
}

if (-not $binary -or -not (Test-Path $binary)) {
    throw "crispasr.exe was not found under $projectRoot. Extract the prebuilt runtime into this folder first."
}

function Invoke-VoxtralSpeech {
    param(
        [string]$Text,
        [string]$OutputPath
    )

    & $binary --backend voxtral-tts --cache-dir $modelsDir -m auto --auto-download `
        --accept-license cc-by-nc-4.0 `
        --tts $Text --voice neutral_female --seed 42 --tts-output $OutputPath

    if ($LASTEXITCODE -ne 0) {
        throw "Voxtral TTS failed with exit code $LASTEXITCODE."
    }

    Write-Host "Created: $OutputPath"
}

if ($ReadOneToThousand) {
    $chunkSize = 50
    $chunkNumber = 0

    for ($start = 1; $start -le 1000; $start += $chunkSize) {
        $end = [Math]::Min($start + $chunkSize - 1, 1000)
        $chunkNumber++
        $text = (($start..$end) -join ', ') + '.'
        $outputPath = Join-Path $outputDir ("numbers-{0:D2}-{1:D4}-to-{2:D4}.wav" -f $chunkNumber, $start, $end)
        Invoke-VoxtralSpeech -Text $text -OutputPath $outputPath
    }

    exit 0
}

do {
    $text = Read-Host 'Enter text (leave blank to exit)'
    if ([string]::IsNullOrWhiteSpace($text)) {
        break
    }

    $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $outputPath = Join-Path $outputDir "voxtral-$timestamp.wav"

    Invoke-VoxtralSpeech -Text $text -OutputPath $outputPath
} while ($true)
