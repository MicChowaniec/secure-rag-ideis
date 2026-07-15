param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$Host.UI.RawUI.WindowTitle = 'Bezpieczny Chatbot RAG IDEIS'
Set-Location -LiteralPath $ProjectRoot

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Find-SystemPython {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "$env:ProgramFiles\Python313\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command -and $command.Source -notlike '*WindowsApps*') {
        try {
            & $command.Source -c 'import sys; assert sys.version_info >= (3, 11)' 2>$null
            if ($LASTEXITCODE -eq 0) { return $command.Source }
        } catch {}
    }
    return $null
}

function Install-Python {
    Write-Step 'Instalowanie Python 3.12 (dla bieżącego użytkownika)'
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        & $winget.Source install --id Python.Python.3.12 -e --scope user `
            --accept-package-agreements --accept-source-agreements --silent
        if ($LASTEXITCODE -ne 0) { throw "winget nie zainstalował Pythona (kod $LASTEXITCODE)." }
    } else {
        $installer = Join-Path $env:TEMP 'python-3.12.10-amd64.exe'
        Invoke-WebRequest `
            -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' `
            -OutFile $installer -UseBasicParsing
        $process = Start-Process -FilePath $installer -Wait -PassThru -ArgumentList @(
            '/quiet', 'InstallAllUsers=0', 'PrependPath=1', 'Include_launcher=1',
            'Include_test=0', 'Shortcuts=0'
        )
        if ($process.ExitCode -ne 0) { throw "Instalator Pythona zwrócił kod $($process.ExitCode)." }
    }
}

function Find-Ollama {
    $command = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:LOCALAPPDATA\Ollama\ollama.exe",
        "$env:ProgramFiles\Ollama\ollama.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

function Test-OllamaApi([string]$Url = 'http://127.0.0.1:11434') {
    try {
        Invoke-RestMethod -Uri "$Url/api/tags" -TimeoutSec 3 | Out-Null
        return $true
    } catch { return $false }
}

function Test-OllamaGeneration([string]$Url) {
    try {
        $body = @{
            model = 'qooba/bielik-1.5b-v3.0-instruct:Q8_0'
            prompt = 'Odpowiedz: OK'
            stream = $false
            options = @{ temperature = 0; num_predict = 3 }
        } | ConvertTo-Json -Depth 4
        $response = Invoke-RestMethod -Uri "$Url/api/generate" -Method Post `
            -ContentType 'application/json; charset=utf-8' `
            -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 180
        return -not [string]::IsNullOrWhiteSpace($response.response)
    } catch {
        Write-Warning "Test generacji Ollama nie powiódł się: $($_.Exception.Message)"
        return $false
    }
}

function Ensure-Ollama([string]$OllamaExe) {
    if (-not (Test-OllamaApi 'http://127.0.0.1:11434')) {
        Write-Step 'Uruchamianie serwera Ollama'
        Start-Process -FilePath $OllamaExe -ArgumentList 'serve' -WindowStyle Hidden
        foreach ($attempt in 1..20) {
            Start-Sleep -Milliseconds 500
            if (Test-OllamaApi 'http://127.0.0.1:11434') { return }
        }
        throw 'Ollama nie odpowiada pod http://127.0.0.1:11434.'
    }
}

function Ensure-OllamaModel([string]$OllamaExe, [string]$Model) {
    $tags = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 5
    $names = @($tags.models | ForEach-Object { $_.name })
    if ($names -notcontains $Model -and $names -notcontains "$Model`:latest") {
        Write-Step "Pobieranie modelu Ollama: $Model"
        & $OllamaExe pull $Model
        if ($LASTEXITCODE -ne 0) { throw "Nie udało się pobrać modelu $Model." }
    }
}

function ConvertFrom-Secure([Security.SecureString]$Secure) {
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
}

function Read-EnvMap([string]$Path) {
    $map = [ordered]@{}
    if (Test-Path -LiteralPath $Path) {
        foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
            if ($line -match '^\s*([^#][^=]*)=(.*)$') {
                $map[$matches[1].Trim()] = $matches[2].Trim()
            }
        }
    }
    return $map
}

function Save-EnvMap([string]$Path, $Map) {
    $lines = foreach ($key in $Map.Keys) { "$key=$($Map[$key])" }
    [IO.File]::WriteAllLines($Path, $lines, [Text.UTF8Encoding]::new($false))
}

if ($SelfTest) {
    Write-Host 'Autotest instalatora: skrypt odczytany poprawnie.' -ForegroundColor Green
    Write-Host "Katalog projektu: $ProjectRoot"
    exit 0
}

try {
    Write-Host 'Bezpieczny Chatbot RAG — Uniwersytet DSW Ideis Kraków' -ForegroundColor Green
    Write-Host "Folder projektu: $ProjectRoot"

    $venvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        $systemPython = Find-SystemPython
        if (-not $systemPython) {
            Install-Python
            $systemPython = Find-SystemPython
        }
        if (-not $systemPython) { throw 'Python został zainstalowany, ale nie można odnaleźć python.exe.' }
        Write-Step 'Tworzenie izolowanego środowiska .venv'
        & $systemPython -m venv (Join-Path $ProjectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Nie udało się utworzyć .venv.' }
    }

    Write-Step 'Instalowanie i aktualizowanie zależności Pythona'
    & $venvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw 'Aktualizacja pip nie powiodła się.' }
    & $venvPython -m pip install -e "$ProjectRoot[models]"
    if ($LASTEXITCODE -ne 0) { throw 'Instalacja zależności projektu nie powiodła się.' }

    $ollama = Find-Ollama
    if (-not $ollama) {
        Write-Step 'Instalowanie Ollama'
        $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        if (-not $winget) { throw 'Brak Ollama i winget. Zainstaluj Ollama z https://ollama.com/download/windows.' }
        & $winget.Source install --id Ollama.Ollama -e --scope user `
            --accept-package-agreements --accept-source-agreements --silent
        $ollama = Find-Ollama
    }
    if (-not $ollama) { throw 'Nie można odnaleźć ollama.exe po instalacji.' }
    Ensure-Ollama $ollama
    Ensure-OllamaModel $ollama 'qooba/bielik-1.5b-v3.0-instruct:Q8_0'
    Ensure-OllamaModel $ollama 'nomic-embed-text'

    $ollamaUrl = 'http://127.0.0.1:11434'
    $ollamaBackend = 'auto'
    Write-Step 'Test generacji z automatycznie wykrytym backendem Ollama'
    if (-not (Test-OllamaGeneration $ollamaUrl)) {
        Write-Warning 'Automatycznie wybrany backend nie generuje odpowiedzi. Uruchamiam przenośny fallback CPU na porcie 11435.'
        Write-Host 'Launcher nie instaluje sterowników GPU: dobór CUDA, ROCm, Vulkan lub CPU pozostaje zadaniem Ollamy.'
        $ollamaUrl = 'http://127.0.0.1:11435'
        $ollamaBackend = 'cpu_avx2-fallback'
        if (-not (Test-OllamaApi $ollamaUrl)) {
            $oldHost = $env:OLLAMA_HOST
            $oldLibrary = $env:OLLAMA_LLM_LIBRARY
            $env:OLLAMA_HOST = '127.0.0.1:11435'
            $env:OLLAMA_LLM_LIBRARY = 'cpu_avx2'
            Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden
            $env:OLLAMA_HOST = $oldHost
            $env:OLLAMA_LLM_LIBRARY = $oldLibrary
            foreach ($attempt in 1..30) {
                Start-Sleep -Milliseconds 500
                if (Test-OllamaApi $ollamaUrl) { break }
            }
        }
        if (-not (Test-OllamaGeneration $ollamaUrl)) {
            throw 'Model nie generuje odpowiedzi ani przez backend automatyczny, ani przez bezpieczny fallback CPU.'
        }
    }

    $envFile = Join-Path $ProjectRoot '.env'
    $config = Read-EnvMap $envFile
    $config['OLLAMA_URL'] = $ollamaUrl
    $config['OLLAMA_BACKEND'] = $ollamaBackend
    $config['OLLAMA_CHAT_MODEL'] = 'qooba/bielik-1.5b-v3.0-instruct:Q8_0'
    $config['OLLAMA_EMBED_MODEL'] = 'nomic-embed-text'
    $config['PRIVACY_MODEL'] = 'openai/privacy-filter'
    $config['BIELIK_GUARD_MODEL'] = 'speakleash/Bielik-Guard-0.1B-v1.1'
    $config['USE_PRIVACY_MODEL'] = 'true'
    $config['PII_POLICY'] = 'mask'
    $config['BLOCK_PROMPT_INJECTION'] = 'true'
    $config['RAG_TOP_K'] = '3'
    $config['HF_HOME'] = "$env:LOCALAPPDATA\SecureRagBot\hf_cache"

    if (-not $config.Contains('TELEGRAM_BOT_TOKEN') -or [string]::IsNullOrWhiteSpace($config['TELEGRAM_BOT_TOKEN'])) {
        Write-Host "`nDo testów Telegram potrzebny jest token z @BotFather." -ForegroundColor Yellow
        Write-Host 'Token zostanie zapisany tylko lokalnie w ignorowanym pliku .env.'
        $secureToken = Read-Host 'Wklej token Telegram (Enter = skonfiguruj później)' -AsSecureString
        $token = ConvertFrom-Secure $secureToken
        if ($token) {
            try {
                $me = Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/getMe" -TimeoutSec 10
                if (-not $me.ok) { throw 'Telegram odrzucił token.' }
                Write-Host "Token poprawny — bot: @$($me.result.username)" -ForegroundColor Green
                $config['TELEGRAM_BOT_TOKEN'] = $token
            } catch {
                Write-Warning 'Nie udało się potwierdzić tokena. Nie został zapisany.'
            }
        }
    }

    Write-Host "`nBielik Guard jest modelem z dostępem warunkowym na Hugging Face." -ForegroundColor Yellow
    Write-Host 'Najpierw zaakceptuj warunki: https://huggingface.co/speakleash/Bielik-Guard-0.1B-v1.1'
    if (-not $config.Contains('HF_TOKEN') -or [string]::IsNullOrWhiteSpace($config['HF_TOKEN'])) {
        $secureHf = Read-Host 'Wklej token HF z uprawnieniem read (Enter = użyj fallbacku)' -AsSecureString
        $hfToken = ConvertFrom-Secure $secureHf
        if ($hfToken) { $config['HF_TOKEN'] = $hfToken }
    }
    $config['USE_BIELIK_GUARD'] = if ($config.Contains('HF_TOKEN') -and $config['HF_TOKEN']) { 'true' } else { 'false' }
    Save-EnvMap $envFile $config

    foreach ($key in $config.Keys) { [Environment]::SetEnvironmentVariable($key, $config[$key], 'Process') }

    Write-Step 'Budowanie lokalnej bazy RAG z regulaminu IDEIS'
    & $venvPython -m secure_rag_bot.ingest
    if ($LASTEXITCODE -ne 0) { throw 'Indeksowanie PDF nie powiodło się.' }

    Write-Step 'Kontrola modeli i konfiguracji'
    & $venvPython -m secure_rag_bot.preflight
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Co najmniej jeden pełny model nie został załadowany. Szczegóły są powyżej; fallback nadal umożliwia testy.'
    }

    while ($true) {
        Write-Host "`n--- MENU ---" -ForegroundColor Green
        Write-Host '1 — Uruchom bota Telegram'
        Write-Host '2 — Uruchom 3 scenariusze demonstracyjne'
        Write-Host '3 — Testy szybkie (jednostkowe + 40 przypadków granicznych)'
        Write-Host '4 — Testy live: fakty regulaminowe + prawdziwy Bielik przez Ollama'
        Write-Host '5 — Diagnostyka konfiguracji i modeli'
        Write-Host 'Q — Zakończ'
        $choice = (Read-Host 'Wybór').Trim().ToUpperInvariant()
        switch ($choice) {
            '1' {
                if (-not $config.Contains('TELEGRAM_BOT_TOKEN') -or -not $config['TELEGRAM_BOT_TOKEN']) {
                    Write-Warning 'Brak TELEGRAM_BOT_TOKEN. Usuń .env i uruchom launcher ponownie albo wpisz token do .env.'
                } else {
                    & $venvPython -m secure_rag_bot.telegram_app
                }
            }
            '2' { & $venvPython -m secure_rag_bot.demo }
            '3' {
                & $venvPython -m unittest discover -s tests -v
                if ($LASTEXITCODE -eq 0) { & $venvPython -m secure_rag_bot.evaluation }
            }
            '4' {
                & $venvPython -m secure_rag_bot.evaluation --live `
                    --cases evals/live_factual_cases.json `
                    --output outputs/evaluation-live-factual-results.json
                if ($LASTEXITCODE -eq 0) {
                    & $venvPython -m secure_rag_bot.evaluation --live `
                        --cases evals/ollama_smoke_case.json `
                        --output outputs/evaluation-ollama-smoke.json
                }
            }
            '5' { & $venvPython -m secure_rag_bot.preflight }
            'Q' { return }
            default { Write-Warning 'Nieznana opcja.' }
        }
    }
} catch {
    Write-Host "`nBŁĄD: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Skopiuj ten komunikat, jeśli potrzebna będzie pomoc.'
    Read-Host 'Naciśnij Enter, aby zamknąć'
    exit 1
}
