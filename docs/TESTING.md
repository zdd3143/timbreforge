# 测试与验收边界

## 最新基线与发布检查

2026-10-16 上传前回归为 **196 项：195 项通过、1 项可选检查跳过**，包括发布资产、凭据、文档链接和归档内容检查。历史章节保留当时的结果，不作为当前总数。

发布前执行 `python tools/release_check.py`，核对源码候选中的私有资产、常见凭据和相对文档链接；构建后用同一工具的 `--archive` 检查 wheel 和源码包。操作见[发布指南](RELEASING.md)。

模型调用已接入跨进程资源锁和本机闲置缓存释放；后文关于操作者手动释放的历史描述不代表当前行为。完整策略及真实克隆显存验证见[内存管理](MEMORY_MANAGEMENT.md)。远端 CI 仅在取得实际运行结果后报告通过。

## 角色设定 → 声线推演（本轮新增）

当前本地全量回归 **180 项，1 项可选外部检查跳过，其余通过**；Windows Gradio 仍输出非致命 event-loop ResourceWarning。新增覆盖：声线库（一个角色多条声线、悬空卡片不展示、原子写入失败保留旧索引、损坏索引报错）、生成记录保留上限 3 条并删除最早、按记录切回、保存声线的确认前置、界面控件与失败清空。

真实浏览器（Edge，本机 7871 独立实例）走通：导入设定 → 推演 → 生成记录一 → ↻ 换种子生成记录二 → 点选切回且输入框文字未被覆盖 → 保存弹窗 → 未勾选确认时被服务拒绝 → 勾选后新建角色成功 → 换一条记录选择已有角色追加 → 该角色显示 2 条声线且各自可播放。全程无 pageerror。测试期间生成的音频与角色仅存于本地 `.local-data`，未公开。

同一轮实测修掉两个真实缺陷：模型输出的依据未标注来源时，旧的硬校验会让整次推演连续失败；以及界面里局部组件名遮蔽了同名服务函数，导致声线列表恒为空（现已改名并在注释中标注原因）。

## Qwen VoiceDesign 增量验收

当前本地全量回归 **141 项，1 项可选外部检查跳过，其余通过**；Windows Gradio 仍输出非致命 event-loop ResourceWarning。新主入口、Qwen 参数隔离、worker 文件协议、全局种子、设计会话与人工选择、固定参考角色复用等均有无模型测试。已在真实 Edge DOM 验证设计页为首个 Tab、未确认时生成被拒绝，且无 pageerror；不是视觉或听感验收。新增控件包括可编辑文字预设、随机种子按钮、真实采样参数和独立参考转录确认。

本机已完成 **真实 VoiceDesign 1.7B CUDA FP16 / SDPA** 单候选：24kHz FLOAT WAV，5.84 秒，seed42、主token上限512；模型加载3.70秒，推理13.85秒，含独立进程启动总计28.62秒，PyTorch峰值显存4,364,776,448字节。用户已试听确认参考台词准确，并选择作为后续Base换台词验收参考；这不是音质或提示词匹配度评级。

Windows初次加载出现safetensors直接CUDA切片原生访问异常，以及系统页面文件/提交内存不足。经用户明确允许，暂停本会话ComfyUI服务释放约10GB提交内存；worker在Windows CUDA下使用进程内CPU暂存安全张量读取、随后逐参数转入指定GPU。不是CPU推理或自动设备回退，不修改安装包/权重，元数据记录该加载方式。隔离环境固定safetensors0.5.3；独立仅导入通过不等于真实加载通过。

同一已确认设计参考已由真实Base0.6B换台词生成6.00秒音频，BF16 / SDPA，加载3.85秒、推理25.15秒、总计38.95秒，PyTorch峰值显存2,578,978,304字节。Base首次FP16触发CUDA采样断言，显式BF16重试成功，角色复用默认明确采用BF16（CLI可覆盖），CPU明确使用FP32；不在失败后隐式更换精度。随后通过真实保存角色的Studio完整链路再次运行成功：同样6.00秒，加载2.54秒、推理15.74秒、总计28.69秒，固定参考SHA未变。换台词身份保持与听感仍待用户评价。CPU侧线程数上限4避免Windows笔记本线程过度竞争，日志标识加载/生成阶段。下载成功、模型文件存在、mock 单元测试与浏览器按钮可点击均不能替代实际 Qwen 音频。初始模型取完整固定 revision，跨镜像下载文件仍须匹配官方 HF LFS SHA256。

## 创作功能增量验收

新增多角色项目、候选 A/B、参考助手后，本地回归共 112 项测试（1 个可选外部 AST 类跳过）。覆盖角色绑定、指令/逐句路由、追加候选、失败保留先前结果、人工选择历史、独立评分、分轨对齐、原音哈希、确认裁切与转录失效。

真实 Edge 浏览器验证了参考上传检查、未核验项提示、确认裁切后转录失效、A/B RMS 副本、快捷试听互斥播放、保存审听选择与 ZIP 下载（HTTP 200），无 pageerror。为隔离 GPU 限制，此次浏览器项目使用明确标识的合成测试音，**不是 TTS 或听感证据**。

另以本地授权合成角色执行真实创作推理：旁白生成两个候选、角色 A 指令句生成一个候选成功；角色 B 加载时 CUDA OOM，重试仍失败。先前候选和人工选择保留。未宣称本轮完整三角色真实导出通过；全流程导出与和轨校验由 CPU 测试及上述测试音浏览器验证。三标签复用同一私有角色只验证路由，不代表三个不同声音身份。CosyVoice 外部 frontend 日志出现 ModelScope 元数据联网请求，因此 HF offline 环境变量不能等同于完全离线沙箱。

## 核心回归测试：无需 GPU、网络或私有资产

安装 CPU 核心依赖后，在仓库根目录运行：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

使用 Python 标准库 unittest，不需要 pytest。测试临时生成合成采样文件，mock 外部推理/API/子进程，验证参考传递、文件编码、角色存储、任务路由、参数图、错误传播等逻辑。不会证明模型真正加载、CUDA 可用、音质达标或 UI 实际浏览器交互正常。

既有开发标签 **latest54** 指之前 54 项以 mock 为主的回归记录，而不是端到端真实生成验收。新增配置/发布测试后计数会变化，请报告实际命令、环境、运行数、跳过数和失败数，不将历史计数冒作本次结果。

界面构建 smoke 在未安装 Gradio 时显式跳过；UI extra 固定兼容基线 Gradio 6.30.0。有 Gradio 时该测试也不启动服务器，不能替代手动浏览器检查。CI 的核心安装不带 UI extra。

## 本次本地运行记录

在 Windows、Python 3.12.10、NumPy 2.5.3、SoundFile 0.14.0、已安装的 Gradio 6.30.0 环境，完成入口/配置/worker 整合后运行上面的 discover 命令：**Ran 68 tests in 7.461s，OK (skipped=1)**。跳过项为未设置环境变量的外部 AST 测试类；Gradio 构建 smoke 实际运行，但出现 asyncio 未关闭 event loop 的 ResourceWarning。另将 opt-in AST 指向已审计节点，3 项检查通过。此记录不是 Python 3.11/Linux/远端 CI 结果；真实模型与浏览器验收单独列在下方。

另以不访问外部安装的检查确认 AST 门控：环境变量未设置或空白值会 skip；显式指定不存在的节点目录会报告缺少源码而失败。

## 外部 IndexTTS AST 集成：明确 opt-in

[外部源码检查](<../tests/test_indextts_source_controls.py>)不导入 ComfyUI 或模型，而用 AST 检查兼容的 IndexTTS 自定义节点源码是否转发采样/扩散控制。默认不读取任何固定磁盘路径，也不会访问开发者本机安装。

只有显式设置 `TIMBREFORGE_INDEXTTS_NODE_DIR` 才启用。变量应指向包含 `nodes.py` 和 `index-tts/indextts/infer_v2_5.py` 的**自定义节点根目录**，而非 ComfyUI 根目录或模型目录。

PowerShell 示例：

```powershell
$env:TIMBREFORGE_INDEXTTS_NODE_DIR = 'C:\your\ComfyUI\custom_nodes\ComfyUI-Index-TTS-25'
python -m unittest discover -s tests -p test_indextts_source_controls.py -v
Remove-Item Env:TIMBREFORGE_INDEXTTS_NODE_DIR
```

Linux 示例：

```bash
TIMBREFORGE_INDEXTTS_NODE_DIR=/your/ComfyUI/custom_nodes/ComfyUI-Index-TTS-25 python -m unittest discover -s tests -p test_indextts_source_controls.py -v
```

未设置或仅空白值会 `SkipTest`；设置为错误路径、缺少源码、语法错误或不匹配的接口时应失败，而不是悄悄 skip。它不修改外部安装、不代表运行中的节点重载成功，也不代表真实推理通过。兼容版本与补丁以[补丁说明](<../patches/INDEXTTS-25.md>)为准。

## CI 的含义

[CI 工作流](<../.github/workflows/ci.yml>)定义 Ubuntu / Windows 与 Python 3.11 / 3.12 四种组合，安装核心包后运行 unittest。外部 AST 集成变量在 CI 中留空；没有权重、私有音频或 GPU 要求。CI 安装 Python 包时需要公共包索引；“离线测试”是指测试执行本身无需网络，而非首次依赖安装无需网络。

工作流存在不等于已经在 GitHub 执行或通过。未获得相应远端日志前，不在 README 加通过徽章或跨平台通过声明。

## 发布包检查

维护者可在自己的构建环境构建 wheel/sdist，并核验：核心只有 numpy、soundfile；UI 是可选 extra；`timbreforge.workers` 的三个 worker 脚本已打包；安装后不依赖源码根 `tools`。测试安装包时应在源码目录之外执行导入/入口检查，以避免源码遮蔽安装包。

本次文档任务不自动安装依赖、不下载模型、不启动服务；构建/安装结果必须记录为实际做过的动作，不能靠命令模板宣称完成。

## 独立真实推理验收（不纳入核心 CI）

在拥有许可的环境和参考下，记录后端提交、Python/依赖版本、驱动、GPU、模型清单及补丁版本。先单后端短句，再检查 clone / instruct / 角色新台词，核验 raw 音频、WAV 编码与元数据；听测记录音色、可懂度、缺字、重复、伪影和耗时，不只看自动分数。

实际目标基线为 RTX 4060 Laptop 8 GB，不是最低要求。进程锁不能代替跨程序显存管理：每次只驻留一个重模型。

本地合成参考仅授权用于**私有验收**时，不得把参考、生成音频、录屏或路径含敏感信息的日志提交到仓库。公开 demo 必须另行取得明确授权；当前不以私有验收替代公开授权演示。

### 本轮私有真实验收（不属于仓库资产）

在仓库外、Windows RTX 4060 Laptop 上使用本地合成参考完成了：CosyVoice3 clone → 保存角色卡 → 结束并重新启动 Studio → 角色列表加载 → 新台词生成。两次真实推理均返回 FLOAT WAV，无页面错误；浏览器流程也验证了上传控件、生成按钮和角色复用按钮。参考音频、生成 WAV、MP4 录屏、截图、日志和本机路径均留在私有验收目录，未复制进仓库或发布。该结果只证明这套已配置本机环境的基本链路，不代表 IndexTTS live schema 已重载、跨平台可复现、音质评级或公开授权。
