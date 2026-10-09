' Hidden mobile/anywhere GPU stack.
' Refreshes the runtime copy outside the repo (wan_stack_watchdog.py --sync-runtime),
' then starts the heal loop from there so it keeps running and restarting while the
' repo folder is locked. Watchdog brings up: Comfy :8188, Cloudflare tunnels,
' gpu_agent :8799, and pushes new tunnel URLs to Render so phone gens keep working.
'
' Normally started by the keepalive task from scripts\install_mobile_autostart.ps1.
Option Explicit
Dim sh, fso, repo, runtime, wd
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
repo = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
runtime = fso.GetParentFolderName(repo) & "\svc"
wd = repo & "\scripts\wan_stack_watchdog.py"
If Not fso.FileExists(wd) Then
  WScript.Quit 1
End If
sh.Run "cmd /c cd /d """ & repo & """ && python """ & wd & """ --sync-runtime", 0, True
If fso.FileExists(runtime & "\start.vbs") Then
  sh.Run "wscript.exe """ & runtime & "\start.vbs""", 0, False
End If
