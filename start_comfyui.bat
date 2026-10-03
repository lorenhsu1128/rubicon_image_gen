@echo off
chcp 65001 > nul
title ComfyUI (rubicon_image_gen)
rem Runs in the default WSL distro; set RUBICON_WSL_DISTRO to pick another (e.g. Ubuntu-24.04)
if defined RUBICON_WSL_DISTRO (
  wsl.exe -d %RUBICON_WSL_DISTRO% --cd "%~dp0." -- bash -lc "bash ./scripts/start_comfyui.sh"
) else (
  wsl.exe --cd "%~dp0." -- bash -lc "bash ./scripts/start_comfyui.sh"
)
pause
