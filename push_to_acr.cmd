@echo off
chcp 65001 >nul
title SFT Docker 镜像构建推送 - ACR

REM ═══════════════════════════════════════════════════════════════
REM  用法:
REM    push_to_acr.cmd                          ← 交互式输入密码
REM    push_to_acr.cmd -p <密码>                ← 命令行指定密码
REM    push_to_acr.cmd -u <用户> -p <密码>      ← 完整参数
REM
REM  可选参数:
REM    --registry     REGISTRY    默认 registry.cn-hangzhou.aliyuncs.com
REM    --namespace    NS          默认 zhangyuanzheng
REM    --repo         REPO        默认 sft-base
REM    --tag          TAG         默认 v1
REM    --dockerfile   DIR         默认 %USERPROFILE%\docker-sft-base
REM ═══════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

REM ── 默认配置 ──────────────────────────────────────────────────
set "REGISTRY=<your-registry>.cn-beijing.personal.cr.aliyuncs.com"
set "NAMESPACE=<your-namespace>"
set "REPO=sft-base"
set "TAG=v1"
set "DOCKERFILE_DIR=%USERPROFILE%\docker-sft-base"
set "USERNAME="
set "PASSWORD="

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
