# 发布准备

仓库只分发代码、文档、测试与后端版本清单。模型、参考录音、角色库、生成结果和本机配置保留在本机忽略目录。

## 检查源码

```powershell
python tools/release_check.py
python tools/backend_audit.py
python -m unittest discover -s tests -p "test_*.py" -v
```

`release_check.py` 检查 Git 跟踪文件和未忽略的新文件：禁止资产、常见凭据、个人 Windows 路径、UTF-8 文本和 Markdown 相对链接。它不会打印匹配的凭据内容，也不能替代人工代码审查或完整密钥扫描。模型日志与角色文件不应成为发布候选。

## 构建安装包

在单独的构建环境安装构建工具后执行：

```powershell
python -m pip install build
python -m build
python tools/release_check.py --archive dist/timbreforge-0.5.0a1-py3-none-any.whl --archive dist/timbreforge-0.5.0a1.tar.gz
```

核对所有包内 worker、三个后端清单及示例配置。wheel 只包含 Python 包和元数据；源码包另外包含文档、测试、启动脚本与维护工具。

在新环境安装 wheel，离开源码目录后执行 `timbreforge-studio --help`、`timbreforge-launch --help` 和 `tools/check_installed.py`，防止源码遮蔽安装包导致误验收。另在仅安装 NumPy/SoundFile 的环境跑核心测试，确认 UI 与模型依赖仍为可选。

## GitHub

1. 核对 staged 文件清单和差异，确认没有忽略文件被强制加入。
2. 提交源码并推送到仓库 `main` 分支。默认私有仓库；调整可见性应另行决定。
3. 查看 GitHub Actions 的实际日志，不在运行前宣称 CI 通过。
4. 如需发行版本，再单独创建标签与 Release，附源码/安装包及 SHA-256；创建仓库不自动发布 PyPI 或公开音频。

版本更新时同步 `pyproject.toml`、`timbreforge/__init__.py` 和 `CHANGELOG.md`，重新执行检查。第三方补丁变化还需更新 `backend-lock.json` 中的指纹。

## 首次上传前的本机结果

Windows / Python 3.12：196 项测试，195 项通过、1 项可选外部检查跳过；源码扫描与后端补丁指纹检查通过。使用已安装的 setuptools 84 构建 wheel 和源码包，检查全部归档内容及五个 worker。wheel 在项目隔离验证环境中离线安装，再于源码目录之外验证版本、入口和模型栈延迟导入；该环境借用本机已有 CPU 依赖，不作为全新机器安装或远端 CI 的替代。
