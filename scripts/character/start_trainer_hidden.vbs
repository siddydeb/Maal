' Hidden launcher for the character-LoRA trainer worker.
' Run every few minutes by the WanTrainerKeepalive scheduled task; the worker exits
' at once when another instance already holds E:\LoraTraining\trainer_worker.lock.
Option Explicit
Dim sh, fso, repo, worker, logFile
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
repo = fso.GetParentFolderName(fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName)))
worker = repo & "\scripts\character\trainer_worker.py"
logFile = repo & "\tmp_test\trainer_worker.log"
If Not fso.FileExists(worker) Then
  WScript.Quit 1
End If
sh.Run "cmd /c cd /d """ & repo & """ && python """ & worker & """ >> """ & logFile & """ 2>&1", 0, False
