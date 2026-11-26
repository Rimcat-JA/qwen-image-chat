# Windows にログインしたとき、Qwen Image Chat（Web UI・Telegram Bot・画像エンジン）を
# ウィンドウなしで自動起動するよう、タスク スケジューラに登録します。管理者権限は不要です。
# 解除するには uninstall-autostart.bat を実行します。
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Name = "QwenImageChat"
$User = "$env:USERDOMAIN\$env:USERNAME"
$Log  = "$Root\data\logs\app.log"
New-Item -ItemType Directory -Force "$Root\data\logs" | Out-Null

# start.ps1 の出力（URL 表示やエラー）は data\logs\app.log に追記します。
$arg = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Root\start.ps1`" -NoBrowser -LogFile `"$Log`""
$action    = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $Root
$trigger   = New-ScheduledTaskTrigger -AtLogOn -User $User
$trigger.Delay = "PT30S"   # ログイン直後はネットワークや Ollama の準備を待つ
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "Qwen Image Chat ($Root)" -Force | Out-Null

Write-Host "自動起動を登録しました（タスク名: $Name）。次回ログインから、約30秒後に起動します。"
Write-Host "ログ: $Log"
Write-Host "今すぐ起動するには:  Start-ScheduledTask -TaskName $Name"
