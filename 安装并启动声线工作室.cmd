@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_CMD=py -3"
where py >nul 2>nul
if errorlevel 1 set "PYTHON_CMD=python"

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] 正在创建本地 Python 环境...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto :failed
)

echo [2/3] 正在安装界面依赖（首次运行需要网络）...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[ui]"
if errorlevel 1 goto :failed

if not exist "timbreforge.local.toml" (
  copy /y "timbreforge.example.toml" "timbreforge.local.toml" >nul
  echo.
  echo 已创建 timbreforge.local.toml。模型未随整合包分发，请按文档填写本机模型路径。
)

echo [3/3] 正在启动本地工作室...
.venv\Scripts\python.exe -m timbreforge.launcher --config timbreforge.local.toml
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo 安装或启动失败。请查看上面的错误信息，或阅读 docs\WINDOWS_BUNDLE.md。
pause
exit /b 1

