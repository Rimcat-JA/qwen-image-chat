# Qwen Image Chat を起動します（画像エンジン sd-server + チャットアプリ）。
# 使い方: start.bat をダブルクリック、または  powershell -ExecutionPolicy Bypass -File start.ps1
param([switch]$NoBrowser, [string]$LogFile)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

function Resolve-RootPath([string]$p, [string]$default) {
    if (-not $p) { $p = $default }
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return (Join-Path $Root $p)
}

$cfg = @{}
if (Test-Path "$Root\local.json") {
    $json = Get-Content "$Root\local.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($p in $json.PSObject.Properties) { $cfg[$p.Name] = $p.Value }
}
$ModelRoot = Resolve-RootPath $cfg["model_root"] "models\qwen-image-2.1"
$SdDir     = Resolve-RootPath $cfg["sdcpp_dir"] "bin\sdcpp"
$AppPort   = if ($cfg["app_port"]) { [int]$cfg["app_port"] } else { 7860 }
$SdPort    = if ($cfg["sd_port"]) { [int]$cfg["sd_port"] } else { 1234 }
$Python    = "$Root\app\.venv\Scripts\python.exe"
$LogDir    = "$Root\data\logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

if (-not (Test-Path $Python)) { throw "Python 環境がありません。先に setup.bat を実行してください。" }
if (-not (Test-Path "$SdDir\sd-server.exe")) { throw "sd-server.exe が見つかりません ($SdDir)。setup.bat を実行してください。" }

function Test-Port([int]$port) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $port); $c.Close(); return $true
    } catch { return $false }
}

# ---- 1) image engine: the app starts and supervises sd-server itself
if (-not (Test-Port $SdPort)) { Write-Host "画像エンジン (sd-server) はアプリが自動で起動します…" }

# ---- 2) chat app (listens on all interfaces so phones on the same Wi-Fi can connect)
$ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and $_.PrefixOrigin -ne 'WellKnown' } |
    Select-Object -ExpandProperty IPAddress
Write-Host ""
Write-Host "==============================================="
Write-Host " Qwen Image Chat"
Write-Host "  このPC:   http://localhost:$AppPort"
foreach ($ip in $ips) { Write-Host "  スマホ等: http://${ip}:$AppPort" }
Write-Host " (スマホから繋がらない場合は allow-lan.bat を一度実行)"
Write-Host " 終了するにはこのウィンドウで Ctrl+C"
Write-Host "==============================================="
if (-not $NoBrowser) {
    Start-Job -ScriptBlock { param($u) Start-Sleep 3; Start-Process $u } -ArgumentList "http://localhost:$AppPort" | Out-Null
}
# uvicorn writes its normal logs to stderr. Under "Stop", Windows PowerShell 5.1 turns
# redirected stderr lines into errors and aborts, so relax it from here on.
$ErrorActionPreference = "Continue"
$uv = "-m uvicorn server:app --app-dir `"$Root\app`" --host 0.0.0.0 --port $AppPort --ws none --log-level warning"
if ($LogFile) {
    # Autostart (no window): append everything to the log file as UTF-8.
    $env:PYTHONUTF8 = "1"
    $urls = (@("http://localhost:$AppPort") + ($ips | ForEach-Object { "http://${_}:$AppPort" })) -join "  "
    Add-Content -Path $LogFile -Encoding UTF8 -Value "`r`n===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') start  $urls"
    & cmd.exe /c "`"$Python`" $uv >> `"$LogFile`" 2>&1"
} else {
    & $Python -m uvicorn server:app --app-dir "$Root\app" --host 0.0.0.0 --port $AppPort --ws none --log-level warning
}
