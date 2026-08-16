param(
    [Parameter(Mandatory=$true)]
    [string]$Username,

    [ValidateSet("wsl", "native")]
    [string]$Type = "wsl"
)

$taskName = "WordGPT-Training"
$executePath = $null
$arguments = $null

if ($Type -eq "wsl") {
    $executePath = "C:\Windows\System32\wsl.exe"
    $arguments = "-d Ubuntu-24.04 -u $Username bash /home/$Username/source/ai/word-gpt-mini/services/start.sh"
}
else {
    $executePath = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    $psScript = "C:\Users\$Username\source\ai\word-gpt-mini\services\start.ps1"
    $arguments = "-ExecutionPolicy Bypass -File `"$psScript`""
}

$action = New-ScheduledTaskAction -Execute $executePath -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force -ErrorAction Stop
    Write-Host "Task '$taskName' registered successfully (type: $Type)" -ForegroundColor Green
}
catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Run PowerShell as Administrator and try again." -ForegroundColor Yellow
    exit 1
}