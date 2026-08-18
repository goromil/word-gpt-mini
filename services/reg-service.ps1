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
    $wslRoot = "/home/$Username/source/ai/word-gpt-mini"
    $logDir = Join-Path $wslRoot "logs"
    $arguments = "-d Ubuntu-24.04 -u $Username bash -c `"cd $wslRoot; mkdir -p $logDir; bash services/start.sh 1> $logDir/wordgpt.log 2> $logDir/wordgpt-error.log`"
}
else {
    $executePath = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    $wordgptRoot = "C:\Users\$Username\source\ai\word-gpt-mini"
    $arguments = "-NoProfile -ExecutionPolicy Bypass -Command `"$wordgptRoot='$wordgptRoot'; `$logDir=Join-Path `$wordgptRoot 'logs'; New-Item -ItemType Directory -Path `$logDir -Force | Out-Null; & (Join-Path `$wordgptRoot 'services\start.ps1') 1>> (Join-Path `$logDir 'wordgpt.log') 2>> (Join-Path `$logDir 'wordgpt-error.log')`"
}

$action = New-ScheduledTaskAction -Execute $executePath -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force -ErrorAction Stop
    Write-Host "Task '$taskName' registered successfully (type: $Type)" -ForegroundColor Green
    Write-Host "" -ForegroundColor White
    Write-Host "Config: C:\Users\$Username\.config\wordgpt\gpt_train.json" -ForegroundColor Yellow
    Write-Host "Logs:   $wordgptRoot\logs\" -ForegroundColor Yellow
}
catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Run PowerShell as Administrator and try again." -ForegroundColor Yellow
    exit 1
}
