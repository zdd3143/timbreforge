# 离线微调训练（LoRA / Adapter）

训练是**离线高级功能**，放在「高级训练」页，不参与普通生成流程。本仓库不包含训练循环、优化器实现或模型权重，也不自动下载模型；它提供数据集准备、任务编排、外部训练器接口、模型登记、评估与回滚的完整安全骨架。

## 总体流程

```text
导入有授权录音 -> 自动质量筛选与切分 -> 逐字稿校对 -> 训练/验证集
-> 外部 LoRA/Adapter 训练 -> 固定测试集评估 -> 登记模型版本 -> 绑定角色卡
```

## 1. 数据集准备

```python
from timbreforge.training import prepare_dataset

summary = prepare_dataset(
    [{'audio': r'D:\recordings\a.wav', 'transcript': '准确逐字稿。'}, ...],
    r'D:\datasets\role-a',
    license_confirmed=True,
)
```

- `license_confirmed` 必须为 `True`；否则直接拒绝。
- 每条录音检查：可解码、非空、有限、非静音、时长在 1.5–15 秒、采样率 8–192 kHz、削波比例 < 0.1%、有声比例 ≥ 25%、信噪比估计 ≥ 12 dB。不满足会进入 `rejected` 并给出中文原因，而不是静默丢弃。
- 接受的录音**逐字节复制**到 `clips/<sha256[:16]>.<后缀>`，从不重编码或覆盖源文件。
- 按音频哈希排序后使用固定种子确定性地划分训练集/验证集（默认 90/10）；至少 2 条时保证验证集非空。
- 输出 `dataset.json`、`train.jsonl`、`valid.jsonl`；目录以临时目录构建后原子改名，已存在的输出目录会被拒绝。
- `load_dataset` 会重新验证清单字段、每段哈希与文件存在性；被篡改的数据集会报错。

## 2. 创建训练任务

```python
from timbreforge.training import create_training_job

job = create_training_job(
    r'D:\datasets\role-a',
    base_model=r'D:\models\Qwen3-TTS-12Hz-0.6B-Base',
    output_dir=r'D:\training',
    name='role-a-lora',
    params={'rank': 16, 'alpha': 32, 'epochs': 5, 'learning_rate': 1e-4,
            'batch_size': 1, 'gradient_accumulation': 8, 'save_every': 1, 'seed': 42},
    role_id=role_id,
)
```

- 基础模型必须已存在于本机，且包含 `config.json`；不下载。
- 参数只允许固定键与范围；未知键会被拒绝。
- 根据基础模型 safetensors 体积估算参数量，并给出显存启发式：`2.2 × 参数量(B) + 1.5 GB`。0.6B 约 2.8 GB。**这是估算，不是保证**；1.7B 需要更多显存与主机内存。
- `job.json` 记录基础模型、数据集摘要、参数、显存估算、训练器配置、日志与检查点路径、警告（离线、授权、显存、不覆盖 Base）。
- 训练命令来自环境变量 `TIMBREFORGE_TRAINING_COMMAND`（JSON 数组，可含 `{request}` 与 `{result}` 占位符）。未配置时可创建任务，但运行会明确提示无法训练。

## 3. 运行训练

```powershell
$env:TIMBREFORGE_TRAINING_COMMAND = '["D:\envs\train\python.exe","D:\train\lora_train.py","--request","{request}","--result","{result}"]'
```

```python
from timbreforge.training import run_training
run_training(job_dir)
```

- 请求 JSON 写入任务目录：job id、训练/验证 JSONL、基础模型、检查点暂存目录、参数与 seed。
- 训练器必须在结果 JSON 中声明 `status: "ok"` 与 `checkpoint_dir`，并写入暂存目录。
- 成功时先验证暂存目录，再原子发布为 `checkpoints/`；已有检查点会先改名保存为 `checkpoints.previous-<时间>`，不会直接删除。
- 失败或超时会记录中文错误与日志路径，**旧检查点保持不变**；不会破坏已有角色或 Base 模型。
- 训练器命令由用户提供，本工具不对其内容做沙箱隔离。

## 4. 评估

```python
from timbreforge.training import record_evaluation
record_evaluation(job_dir, {'cer': 0.05, 'wer': 0.08, 'speaker_similarity': 0.82,
                            'mos': 3.8, 'notes': '固定测试集 8 类，最差为长句。'})
```

- 允许的字段：`cer`、`wer`（≥0）、`speaker_similarity`（-1–1）、`mos`（0–5）、`notes`（≤1000 字）。
- 评估写入 `evaluation.json` 并更新 `job.json`；指标是辅助证据，不能替代试听。
- 建议同时使用 [固定测试集](TUNING_AND_EVALUATION.md) 检查短句、长句、语速、音色、性别感与过拟合。

## 5. 登记、绑定与回滚

```python
from timbreforge.training import register_trained_model, rollback_model

register_trained_model(job_dir, bind_role_id=role_id, confirmed=True)
rollback_model(role_id)
```

- 只有 `completed` 且有评估的任务才能登记；必须 `confirmed=True`。
- 检查点目录必须能被模型注册表识别（至少包含 `config.json`）；登记类型为 `lora`，名称沿用任务名，版本为 `job-<id 前 12 位>`。
- 绑定角色会把角色卡模型名称/版本切换到该 LoRA，使旧 Clone Prompt 失效；不绑定则不修改任何角色。
- `rollback_model` 复用模型注册表的回滚记录，恢复切换前的模型版本。
- Base 模型只被引用，从不复制或覆盖。

## 安全与限制

- 只处理有合法使用权的录音；界面要求显式确认授权。
- 逐字稿必须人工校对；噪声、混响和多人说话应在准备阶段被拒绝或人工剔除。
- 本仓库不提供实际训练算法；需要自备 LoRA/Adapter 训练器和足够显存。低显存优先验证 0.6B Base。
- 训练页面显示基础模型、训练日期、数据摘要、参数、评估结果与显存需求，但不会自动开始训练。
- 训练失败不会修改已有角色；检查点发布与回滚都保留旧版本。

## 测试

```powershell
python -m unittest tests.test_training
```

覆盖授权确认、质量筛选与拒绝原因、字节级复制、确定性切分、数据集篡改检测、参数校验、显存估算、无训练器提示、注入 runner 的成功/失败/旧检查点保留、评估校验、登记要求和角色回滚。

