# install-autostart.ps1 で登録した自動起動を解除し、動いているアプリも止めます。
$Name = "QwenImageChat"
if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $Name -Confirm:$false
    Write-Host "自動起動を解除しました（タスク名: $Name）。"
} else {
    Write-Host "自動起動は登録されていません。"
}
& "$PSScriptRoot\stop.bat" nopause
