"""Convenient local startup without installing models or changing configuration."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import socket


def port_in_use(port: int, host: str = '127.0.0.1', timeout: float = 1.0) -> bool:
    """A listening port usually means a studio window is already open somewhere."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((host, port)) == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='启动 TimbreForge 角色声线工作室')
    parser.add_argument('--config', help='指定 TOML；默认查找当前目录的 timbreforge.local.toml')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--check', action='store_true', help='只检查配置和本地模型文件')
    args = parser.parse_args(argv)
    from timbreforge.config import set_config_path
    from timbreforge.readiness import design_readiness, readiness_markdown
    selected = args.config or os.environ.get('TIMBREFORGE_CONFIG')
    if not selected and Path('timbreforge.local.toml').is_file():
        selected = 'timbreforge.local.toml'
    try:
        set_config_path(selected)
        report = design_readiness()
        print(readiness_markdown(report))
        if args.check:
            return 0 if report['design_files_ready'] else 1
        if importlib.util.find_spec('gradio') is None:
            print('缺少界面依赖。请运行：python -m pip install ".[ui]"')
            return 1
        if port_in_use(args.port):
            print(f'端口 {args.port} 已被占用：很可能已经有一个工作室窗口在运行'
                  f'（旧窗口不会自己退出，也不会自动加载新代码）。\n'
                  f'请先关闭那个窗口，或改用其它端口，例如：'
                  f'python -m timbreforge.launcher --port {args.port + 1}')
            return 1
        from timbreforge.webui import main as launch
        options = ['--port', str(args.port)]
        if selected:
            options += ['--config', str(selected)]
        if not args.no_browser:
            options += ['--open-browser']
        return launch(options)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'启动失败：{exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
