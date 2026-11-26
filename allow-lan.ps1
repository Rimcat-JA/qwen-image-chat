# 同じネットワーク(LAN)のスマホから Qwen Image Chat に接続できるよう、
# Windows ファイアウォールに受信規則を追加します（TCP 7860・同一サブネットからのみ）。
# 管理者権限が必要です。allow-lan.bat から実行してください。
$Root = $PSScriptRoot
$port = 7860
if (Test-Path "$Root\local.json") {
    $cfg = Get-Content "$Root\local.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($cfg.app_port) { $port = [int]$cfg.app_port }
}
$name = "Qwen Image Chat (LAN $port)"
Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port `
    -RemoteAddress LocalSubnet -Profile Any | Out-Null
Write-Host "ファイアウォール規則を追加しました: $name （同一サブネットからのみ許可）"
Write-Host "削除する場合: Remove-NetFirewallRule -DisplayName '$name'"
Start-Sleep -Seconds 4
