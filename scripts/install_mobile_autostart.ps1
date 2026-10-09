# Install + start the mobile/anywhere GPU stack so it runs hidden at every login.
#
# Keeps this PC ready for: phone / other PC -> Vercel -> Render -> Cloudflare -> Comfy here.
# Starts (and re-heals): ComfyUI :8188, Comfy tunnel, gpu_agent :8799, agent tunnel,
# Prowler Control :8010 + Prowler tunnel (prowler_url in tokens&cmd).
# Updates Render COMFYUI_URL / GPU_AGENT_URL when a tunnel URL changes.
#
# One-time setup (PowerShell, this GPU PC):
#   powershell -ExecutionPolicy Bypass -File scripts/install_mobile_autostart.ps1
#
# Options:
#   -NoStart           only install the keepalive task (do not start now)
#   -Status            print whether the keepalive task + stack look healthy
#   -Uninstall         remove the keepalive task only (does not kill running processes)
#
# Re-run after changing wan_stack_watchdog.py / gpu_agent.py (the watchdog also
# re-syncs its runtime copy on every start while the repo is unlocked).
#
# Requires: python on PATH, cloudflared, gitignored tokens&cmd with render=<key>
# Leave this PC on; disable sleep while you need remote gens.

param(
    [switch]$NoStart,
    [switch]$Status,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$watchdog = Join-Path $repo "scripts\wan_stack_watchdog.py"
$tokens = Join-Path $repo "tokens&cmd"
# Runtime copy lives outside the repo (hidden) so the stack restarts while the repo is locked.
$runtime = Join-Path (Split-Path $repo -Parent) "svc"
$vbs = Join-Path $runtime "start.vbs"
$logFile = Join-Path $runtime "svc.log"
$taskName = "UserSvcKeepalive"
$startup = [Environment]::GetFolderPath("Startup")
$legacyLnk = Join-Path $startup "WanStudioGpuWatchdog.lnk"
$legacyTask = "WanTrainerKeepalive"
$cloudflared = "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe"
if (-not (Test-Path $cloudflared)) {
    $cloudflared = "$env:ProgramFiles\cloudflared\cloudflared.exe"
}

function Test-PortOpen([int]$Port) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.ReceiveTimeout = 1000
        $c.SendTimeout = 1000
        $c.Connect("127.0.0.1", $Port)
        $c.Close()
        return $true
    } catch {
        return $false
    }
}

function Show-Status {
    Write-Host "=== Mobile stack status ==="
    Write-Host "Repo:     $repo"
    Write-Host "Runtime:  $runtime"
    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        Write-Host "Keepalive task: INSTALLED - $taskName"
    } else {
        Write-Host "Keepalive task: MISSING"
    }
    if (Test-Path $tokens) {
        Write-Host "tokens:   OK"
    } else {
        Write-Host "tokens:   MISSING - need tokens&cmd with render="
    }
    if (Test-Path $cloudflared) {
        Write-Host "cloudflared: OK - $cloudflared"
    } else {
        Write-Host "cloudflared: MISSING"
    }
    if (Test-PortOpen 8188) { Write-Host "Comfy :8188: UP" } else { Write-Host "Comfy :8188: down" }
    if (Test-PortOpen 8799) { Write-Host "gpu_agent :8799: UP" } else { Write-Host "gpu_agent :8799: down" }
    $wd = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match "wan_stack_watchdog\.py|svc_main\.py" -and $_.Name -match "python" }
    if ($wd) {
        Write-Host ("watchdog: RUNNING (pid {0})" -f ($wd.ProcessId -join ","))
    } else {
        Write-Host "watchdog: not running"
    }
    if (Test-Path $logFile) {
        Write-Host "Log tail ($logFile):"
        Get-Content $logFile -Tail 5 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" }
    }
    try {
        $h = Invoke-RestMethod -Uri "https://wan-studio-api.onrender.com/api/health" -TimeoutSec 20
        Write-Host ("Render:   ok={0} comfyui={1} url={2}" -f $h.ok, $h.comfyui, $h.comfyui_url)
    } catch {
        Write-Host ("Render:   unreachable ({0})" -f $_.Exception.Message)
    }
    Write-Host "Frontend: https://frontend-six-chi-37.vercel.app"
}

if ($Status) {
    Show-Status
    exit 0
}

if ($Uninstall) {
    foreach ($t in @($taskName, $legacyTask)) {
        if (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $t -Confirm:$false
            Write-Host "Removed task: $t"
        }
    }
    if (Test-Path $legacyLnk) { Remove-Item $legacyLnk -Force }
    exit 0
}

if (-not (Test-Path $watchdog)) { throw "Missing $watchdog" }
if (-not (Test-Path $tokens)) {
    throw "Missing tokens&cmd - add render=<Render API key> (keep file gitignored)"
}
$hasRender = Select-String -Path $tokens -Pattern "^render=.+" -Quiet
if (-not $hasRender) {
    throw "tokens&cmd has no render= line - Render cannot get tunnel URL updates"
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python not on PATH"
}
if (-not (Test-Path $cloudflared)) {
    Write-Warning "cloudflared not found - install from Cloudflare, then re-run. Watchdog will fail tunnels until then."
}

# Copy watchdog + gpu_agent + encrypted token subset into the runtime folder
& python $watchdog --sync-runtime
if ($LASTEXITCODE -ne 0) { throw "runtime sync failed" }
if (-not (Test-Path $vbs)) { throw "Missing $vbs after sync" }

# Keepalive: at sign-in and every 5 min; start.vbs exits at once if the loop is running
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`""
$triggers = @(
    (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME),
    (New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5))
)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $triggers -Settings $settings -Force | Out-Null

# Legacy starters ran from inside the repo (fail with a popup while it is locked)
if (Get-ScheduledTask -TaskName $legacyTask -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $legacyTask -Confirm:$false
}
foreach ($name in @("WanStudioGpuWatchdog.lnk", "WanStudioGpuWatchdog.cmd", "WanStudioCloudflared.cmd")) {
    $p = Join-Path $startup $name
    if (Test-Path $p) { Remove-Item $p -Force }
}

Write-Host "Installed keepalive task: $taskName (sign-in + every 5 min)"
Write-Host "The stack starts hidden from $runtime and heals in the background, even with the repo locked."
Write-Host "Logs: $logFile"
Write-Host "Disable sleep while you need phone gens from elsewhere."

if ($NoStart) {
    Write-Host "Skipped start (-NoStart). Will run at next login."
    exit 0
}

Write-Host "Starting stack now (hidden)..."
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$vbs`"" -WindowStyle Hidden
Start-Sleep -Seconds 4
Show-Status
Write-Host ""
Write-Host "Done. Generate from phone/other PC at https://frontend-six-chi-37.vercel.app"
