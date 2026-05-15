@echo off
chcp 65001 >nul
title SFT Docker 镜像构建推送 - ACR

REM ═══════════════════════════════════════════════════════════════
REM  凭证自动从 ..\secret.json 的 "acr" 段加载。
REM  也可用命令行参数覆盖:
REM    push_to_acr.cmd -u <用户> -p <密码> --tag v3
REM ═══════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

REM ── 从 secret.json 加载 ACR 配置 ──────────────────────────────
set "SECRET_FILE=%~dp0..\secret.json"
if exist "!SECRET_FILE!" (
    for /f "tokens=1,* delims=:" %%a in ('python -c "import json,sys;d=json.load(open(r'!SECRET_FILE!',encoding='utf-8'))['acr'];print('USERNAME:',d['username']);print('PASSWORD:',d['password']);print('REGISTRY:',d['registry_public']);print('NAMESPACE:',d['namespace']);print('REPO:',d['repo'])" 2^>nul') do (
        set "line=%%a:%%b"
        set "key=%%a"
        set "val=%%b"
        rem 去除值首尾空格
        for /f "tokens=*" %%v in ("!val!") do set "val=%%v"
        if "!key!"=="USERNAME"  if "!val!" neq "" set "SECRET_USERNAME=!val!"
        if "!key!"=="PASSWORD"  if "!val!" neq "" set "SECRET_PASSWORD=!val!"
        if "!key!"=="REGISTRY"  if "!val!" neq "" set "SECRET_REGISTRY=!val!"
        if "!key!"=="NAMESPACE" if "!val!" neq "" set "SECRET_NAMESPACE=!val!"
        if "!key!"=="REPO"      if "!val!" neq "" set "SECRET_REPO=!val!"
    )
)

REM ── 默认配置 (secret.json 优先，命令行参数可覆盖) ─────────────
if not defined REGISTRY  set "REGISTRY=!SECRET_REGISTRY!"
if not defined NAMESPACE set "NAMESPACE=!SECRET_NAMESPACE!"
if not defined REPO      set "REPO=!SECRET_REPO!"
if not defined USERNAME  set "USERNAME=!SECRET_USERNAME!"
if not defined PASSWORD  set "PASSWORD=!SECRET_PASSWORD!"

REM ── 兜底硬编码 ─────────────────────────────────────────────────
if not defined REGISTRY  set "REGISTRY=crpi-32rssczyu25r10yu.cn-beijing.personal.cr.aliyuncs.com"
if not defined NAMESPACE set "NAMESPACE=zyz25"
if not defined REPO      set "REPO=sft-base"
set "TAG=v1"
set "DOCKERFILE_DIR=%USERPROFILE%\docker-sft-base"

REM ── 解析命令行参数 ────────────────────────────────────────────
:parse
if "%~1"=="" goto :end_parse
if /i "%~1"=="-u"          set "USERNAME=%~2"     & shift & shift & goto :parse
if /i "%~1"=="-p"          set "PASSWORD=%~2"      & shift & shift & goto :parse
if /i "%~1"=="--registry"  set "REGISTRY=%~2"      & shift & shift & goto :parse
if /i "%~1"=="--namespace" set "NAMESPACE=%~2"     & shift & shift & goto :parse
if /i "%~1"=="--repo"      set "REPO=%~2"          & shift & shift & goto :parse
if /i "%~1"=="--tag"       set "TAG=%~2"           & shift & shift & goto :parse
if /i "%~1"=="--dockerfile" set "DOCKERFILE_DIR=%~2" & shift & shift & goto :parse
echo [错误] 未知参数: %~1
exit /b 1
:end_parse

set "IMAGE=%REGISTRY%/%NAMESPACE%/%REPO%:%TAG%"

REM ── 步骤 0：检查 docker ─────────────────────────────────────
echo ╔══════════════════════════════════════════════════════════════╗
echo ║        SFT Docker 镜像 — 构建 ^& 推送到 ACR                  ║
echo ║        %IMAGE%                                    ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

docker version >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [错误] Docker 不可用。请先启动 Docker Desktop。
    exit /b 1
)
echo [OK] Docker 运行正常
echo.

REM ── 步骤 1：登录 ACR ────────────────────────────────────────
echo =================================================================
echo  步骤 1/3 : 登录阿里云 ACR
echo =================================================================

if "!USERNAME!"=="" set /p "USERNAME=ACR 登录用户名: "
if "!PASSWORD!"=="" set /p "PASSWORD=ACR Registry 密码: "

echo [凭证] 用户名=!USERNAME!  Registry=!REGISTRY!

echo docker login !REGISTRY! --username=!USERNAME! [密码已隐藏]
docker login !REGISTRY! --username="!USERNAME!" --password="!PASSWORD!" 2>&1
if !ERRORLEVEL! neq 0 (
    echo [错误] 登录失败，请检查用户名和密码。
    exit /b 1
)
echo [OK] ACR 登录成功
echo.

REM ── 步骤 2：构建镜像 ────────────────────────────────────────
echo =================================================================
echo  步骤 2/3 : 构建 Docker 镜像
echo =================================================================
echo  目录: !DOCKERFILE_DIR!
echo  镜像: !IMAGE!
echo.

docker build -t !IMAGE! "!DOCKERFILE_DIR!" 2>&1
if !ERRORLEVEL! neq 0 (
    echo [错误] 构建失败，请检查 Dockerfile 和网络连接。
    exit /b 1
)
echo [OK] 镜像构建成功
echo.

REM ── 步骤 3：推送镜像 ────────────────────────────────────────
echo =================================================================
echo  步骤 3/3 : 推送镜像到 ACR
echo =================================================================
echo.

docker push !IMAGE! 2>&1
if !ERRORLEVEL! neq 0 (
    echo [错误] 推送失败，请检查网络连接和 ACR 配额。
    exit /b 1
)

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  全部完成！                                                 ║
echo ║                                                              ║
echo ║  镜像地址:                                                   ║
echo ║  docker://!IMAGE!                                          ║
echo ║                                                              ║
echo ║  Magnus 蓝图中使用:                                          ║
echo ║  container_image = "docker://!IMAGE!"                      ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

exit /b 0
