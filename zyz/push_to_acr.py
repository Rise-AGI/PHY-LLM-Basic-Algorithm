"""
构建并推送 SFT Docker 镜像到阿里云 ACR (Alibaba Container Registry)。

用法:
    python push_to_acr.py                              # 交互式输入密码
    python push_to_acr.py -p <password>                # 命令行指定密码
    python push_to_acr.py -u <user> -p <password>      # 完整参数

可选参数:
    --registry  REGISTRY       默认 registry.cn-hangzhou.aliyuncs.com
    --namespace NS             默认 zhangyuanzheng
    --repo      REPO           默认 sft-base
    --tag       TAG            默认 v1
    --dockerfile DIR           默认 ~/docker-sft-base
    --no-push                  只构建不推送 (用于调试)
"""

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建并推送 SFT Docker 镜像到阿里云 ACR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── ACR 登录信息 ────────────────────────────────────────────
    parser.add_argument("-u", "--username", default=os.getenv("ACR_USERNAME"),
                        help="ACR 登录用户名 (或设置 ACR_USERNAME 环境变量)")
    parser.add_argument("-p", "--password", default=os.getenv("ACR_PASSWORD"),
                        help="ACR Registry 密码 (或设置 ACR_PASSWORD 环境变量)")

    # ── 镜像参数 ────────────────────────────────────────────────
    parser.add_argument("--registry", default="<your-registry>.cn-beijing.personal.cr.aliyuncs.com",
                        help="ACR 域名")
    parser.add_argument("--namespace", default="<your-namespace>",
                        help="ACR 命名空间")
    parser.add_argument("--repo", default="sft-base",
                        help="仓库名称 (默认: sft-base)")
    parser.add_argument("--tag", default="v1",
                        help="镜像标签 (默认: v1)")

    # ── 构建参数 ────────────────────────────────────────────────
    parser.add_argument("--dockerfile", default=None,
                        help="Dockerfile 所在目录或文件路径")
    parser.add_argument("--no-push", action="store_true",
                        help="只构建不推送")

    return parser.parse_args(argv)


def check_docker() -> None:
    """检查 Docker 是否可用。"""
    try:
        subprocess.run(
            ["docker", "version"],
            capture_output=True, check=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[错误] Docker 不可用。请先启动 Docker Desktop。")
        sys.exit(1)
    print("[OK] Docker 运行正常\n")


def docker_login(registry: str, username: str, password: str) -> None:
    """登录阿里云 ACR。"""
    print("=" * 65)
    print("  步骤 1/3 : 登录阿里云 ACR")
    print("=" * 65)

    if not username:
        username = input("ACR 登录用户名: ").strip()
    if not password:
        password = getpass.getpass("ACR Registry 密码: ")

    print(f"  docker login {registry} --username={username} [密码已隐藏]")
    result = subprocess.run(
        ["docker", "login", registry,
         "--username", username,
         "--password-stdin"],
        input=password.encode("utf-8"),
        capture_output=True,
    )
    if result.returncode != 0:
        err = result.stderr.decode().strip()
        print(f"[错误] 登录失败: {err}")
        sys.exit(1)
    print("[OK] ACR 登录成功\n")


def docker_build(image: str, dockerfile_path: str | Path) -> None:
    """构建 Docker 镜像。"""
    print("=" * 65)
    print("  步骤 2/3 : 构建 Docker 镜像")
    print("=" * 65)
    print(f"  源:  {dockerfile_path}")
    print(f"  镜像: {image}\n")

    # ── 判断是目录还是文件 ──────────────────────────────────────
    dp = Path(dockerfile_path)
    if dp.is_file():
        # 用户传的是 Dockerfile 路径
        build_dir = dp.parent
        cmd = ["docker", "build", "-t", image,
               "-f", str(dp.resolve()), str(build_dir.resolve())]
    else:
        cmd = ["docker", "build", "-t", image, str(dp.resolve())]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[错误] 构建失败，请检查 Dockerfile 和网络连接。")
        sys.exit(1)
    print("[OK] 镜像构建成功\n")


def docker_push(image: str) -> None:
    """推送镜像到 ACR。"""
    print("=" * 65)
    print("  步骤 3/3 : 推送镜像到 ACR")
    print("=" * 65)

    result = subprocess.run(["docker", "push", image])
    if result.returncode != 0:
        print("[错误] 推送失败，请检查网络连接和 ACR 配额。")
        sys.exit(1)


def print_summary(image: str) -> None:
    """打印最终结果。"""
    print()
    print("╔" + "═" * 65 + "╗")
    print("║  全部完成！                                            ║")
    print("║" + " " * 65 + "║")
    print(f"║  镜像地址:                                             ║")
    print(f"║  docker://{image}")
    print("║" + " " * 65 + "║")
    print("║  Magnus 蓝图中使用:                                     ║")
    print(f"║  container_image = \"docker://{image}\"")
    print("╚" + "═" * 65 + "╝")
    print()


def main() -> None:
    args = parse_args()

    image = f"{args.registry}/{args.namespace}/{args.repo}:{args.tag}"

    print()
    print("╔" + "═" * 65 + "╗")
    print("║  SFT Docker 镜像 — 构建 & 推送到 ACR                  ║")
    print(f"║  {image}")
    print("╚" + "═" * 65 + "╝")
    print()

    # ── Dockerfile 默认路径 ─────────────────────────────────────
    if args.dockerfile is None:
        default_dir = Path.home() / "docker-sft-base"
        alt_file = Path(__file__).resolve().parent / "Dockerfile.sft"
        if default_dir.exists():
            args.dockerfile = str(default_dir)
        elif alt_file.exists():
            args.dockerfile = str(alt_file)
        elif (Path.home() / "docker-sft-base" / "Dockerfile").exists():
            args.dockerfile = str(Path.home() / "docker-sft-base")
        else:
            print("[错误] 找不到 Dockerfile。请用 --dockerfile 指定路径。")
            print("       期望路径 (任选其一):")
            print(f"         - {Path.home() / 'docker-sft-base'}")
            print(f"         - {alt_file}")
            sys.exit(1)

    # ── 执行流程 ────────────────────────────────────────────────
    check_docker()
    docker_login(args.registry, args.username, args.password)
    docker_build(image, args.dockerfile)
    if not args.no_push:
        docker_push(image)
    else:
        print("[跳过] --no-push 指定，不推送\n")
    print_summary(image)


if __name__ == "__main__":
    main()
