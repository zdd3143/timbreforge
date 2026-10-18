"""Model management, settings/diagnostics and the offline advanced-training page.

Training is deliberately last and off the normal generation path. Nothing on
these pages downloads a model or starts training on page load.
"""
from __future__ import annotations

import json
from pathlib import Path

from timbreforge import ui_tasks
from timbreforge.adapters import AdapterError, AdapterRegistry
from timbreforge.characters import CharacterStore
from timbreforge.config import get_config
from timbreforge.model_registry import KINDS, ModelRegistry, ModelRegistryError

KIND_LABELS = [('TTS 基础模型', 'tts_base'), ('TTS 设计模型', 'tts_design'),
               ('声线转换模型', 'vc'), ('LoRA/Adapter', 'lora'),
               ('其它适配器', 'adapter'), ('角色推演模型', 'planner')]


def registry():
    return ModelRegistry()


def role_choices(store=None):
    cards = (store or CharacterStore()).list()
    return [(f"{card['name']} · v{card.get('version', 1)}", card['id']) for card in cards]


def entry_choices(reg=None):
    return [(f"{entry['name']} @ {entry['revision']} · {entry['kind']}", entry['id'])
            for entry in (reg or registry()).load_all()]


def model_rows(reg=None):
    rows = []
    for entry in (reg or registry()).load_all():
        rows.append([entry['name'], entry['revision'], entry['kind'],
                     f"{entry['bytes'] / 1024 ** 3:.1f} GiB" if entry['bytes'] else '未知',
                     entry['path'], entry['config_sha256'][:12], entry['registered_at'][:19]])
    return rows


def register_model(name, path, revision, kind, notes='', base_model='', replace=False):
    if kind not in KINDS:
        raise ValueError('请选择模型类型。')
    entry = registry().register(name, path, revision, kind=kind, notes=notes or '',
                                base_model=base_model or '', replace=bool(replace))
    verify = registry().verify(entry['id'])
    message = f"已登记「{entry['name']}」@ {entry['revision']}（{entry['kind']}）。"
    if not verify['ok']:
        message += '\n\n检查提示：' + '；'.join(verify['problems'])
    return message


def verify_model(entry_id):
    if not entry_id:
        raise ValueError('请选择要检查的模型记录。')
    report = registry().verify(entry_id)
    if report['ok']:
        return f"模型文件完整：{report['entry']['path']}"
    return '检查发现问题：' + '；'.join(report['problems'])


def activate_model(role_id, entry_id):
    if not role_id:
        raise ValueError('请先选择角色。')
    if not entry_id:
        raise ValueError('请选择模型版本。')
    result = registry().activate(role_id, entry_id)
    if not result['changed']:
        return '角色卡已使用该模型版本，无需切换。'
    card = result['card']
    return (f"已切换为「{result['entry']['name']}」@ {result['entry']['revision']}"
            f"（角色版本 v{card['version']}）。旧 Clone Prompt 已失效，请重新创建。")


def rollback_role(role_id):
    if not role_id:
        raise ValueError('请先选择角色。')
    result = registry().rollback(role_id)
    previous = result['restored']
    return f"已回滚到「{previous.get('name') or '未登记'}」@ {previous.get('revision') or '本地'}。"


def remove_model(entry_id):
    if not entry_id:
        raise ValueError('请选择要移除的模型记录。')
    entry = registry().remove(entry_id)
    return f"已从注册表移除「{entry['name']}」（模型文件仍保留在磁盘上）。"


def adapter_registry():
    return AdapterRegistry()


def adapter_choices(reg=None):
    return [(f"{item['name']} · scale {item['default_scale']:g}", item['id'])
            for item in (reg or adapter_registry()).list()]


def adapter_rows(reg=None):
    rows = []
    for item in (reg or adapter_registry()).list():
        rows.append([item['name'], item['base_model'], item['rank'], item['alpha'],
                     f"{item['default_scale']:g}",
                     f"{item['bytes'] / 1024 ** 2:.1f} MiB",
                     item['author'], item['license'], item['artifact_sha256'][:12]])
    return rows


def import_adapter(path, name='', author='', source_url='', license_name='未知',
                   notes='', base_revision='', scale=0.3):
    if not path:
        raise ValueError('请选择本地 LoRA 文件夹或 ZIP。')
    registry_obj = adapter_registry()
    source = Path(path).expanduser().resolve()
    metadata = dict(name=name, author=author, source_url=source_url,
                    license=license_name, notes=notes,
                    base_revision=base_revision, default_scale=scale)
    if source.is_file() and source.suffix.lower() == '.zip':
        entry = registry_obj.import_zip(source, **metadata)
    else:
        entry = registry_obj.register(source, **metadata)
    return (f"已导入「{entry['name']}」；Base：{entry['base_model'] or '未声明'}；"
            f"rank={entry['rank']}，scale={entry['default_scale']:g}。")


def verify_adapter(adapter_id):
    if not adapter_id:
        raise ValueError('请选择适配器。')
    report = adapter_registry().verify(adapter_id)
    if report['ok']:
        return f"适配器完整：{report['entry']['path']}，哈希 {report['entry']['artifact_sha256']}"
    return '检查发现问题：' + '；'.join(report['problems'])


def remove_adapter(adapter_id):
    if not adapter_id:
        raise ValueError('请选择适配器。')
    entry = adapter_registry().remove(adapter_id)
    return f"已从社区适配器注册表移除「{entry['name']}」（原文件仍保留）。"


def bind_adapter(role_id, adapter_id, scale):
    if not role_id:
        raise ValueError('请选择角色。')
    if not adapter_id:
        raise ValueError('请选择适配器。')
    card = adapter_registry().bind_role(role_id, adapter_id, scale=scale)
    bound = (card.get('provenance') or {}).get('adapter') or {}
    return f"已将「{bound.get('name', '社区适配器')}」绑定到角色「{card['name']}」，scale={bound.get('scale', scale):g}。"


def unbind_adapter(role_id):
    if not role_id:
        raise ValueError('请选择角色。')
    card = adapter_registry().unbind_role(role_id)
    return f"已解除角色「{card['name']}」的社区适配器绑定。"


def vc_markdown():
    try:
        from timbreforge.vc_backend import vc_status_markdown
        return vc_status_markdown()
    except Exception as exc:  # noqa: BLE001 - optional experimental backend
        return f'Voice Conversion 实验后端不可用：{exc}'


def settings_markdown():
    cfg = get_config()
    return (f"### 本机路径\n\n"
            f"- 角色库：`{cfg.characters_dir}`\n"
            f"- 输出：`{cfg.outputs_dir}`\n"
            f"- Qwen Python：`{cfg.qwen_python}`\n"
            f"- Qwen Base：`{cfg.qwen_base_model}`\n"
            f"- Qwen VoiceDesign：`{cfg.qwen_design_model}`\n"
            f"- 语言模型：`{cfg.planner_model}`\n\n"
            f"- {ui_tasks.memory_summary()}\n"
            f"- 模型不会自动下载；所有重型任务串行执行并在失败时清理临时资源。")


def cleanup_previews():
    removed = ui_tasks.clear_old_previews(keep_per_role=1)
    return f'已删除 {removed} 个历史预览文件（每个角色保留最新一条）。'


def cleanup_staging():
    removed = ui_tasks.clear_staging()
    return f'已删除 {removed} 个失败的 Qwen 临时请求/结果文件。'


# ------------------------------------------------------------------ training
def prepare_dataset_from_manifest(manifest_path, output_dir, license_confirmed):
    from timbreforge import training
    if not manifest_path:
        raise ValueError('请上传数据集清单 JSON（[{"audio": ..., "transcript": ...}]）。')
    path = Path(manifest_path)
    try:
        items = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'数据集清单无法读取：{exc}') from exc
    if not isinstance(items, list):
        raise ValueError('数据集清单必须是 JSON 数组。')
    resolved = [{'audio': str((path.parent / item['audio']).resolve()
                              if not Path(item['audio']).is_absolute() else Path(item['audio'])),
                 'transcript': item.get('transcript', '')}
                for item in items if isinstance(item, dict) and item.get('audio')]
    dataset = training.prepare_dataset(resolved, output_dir,
                                       license_confirmed=license_confirmed is True)
    return dataset


def dataset_markdown(dataset):
    if not dataset:
        return '尚未准备数据集。'
    manifest = dataset.get('manifest', dataset)
    counts = dataset.get('counts') or manifest.get('counts') or {}
    items = dataset.get('train', []) or manifest.get('items', [])
    return (f"### 数据集已准备\n\n"
            f"- 接受：{len(items)} 条 · 训练 {counts.get('train', 0)} / "
            f"验证 {counts.get('valid', 0)}\n"
            f"- 被拒绝：{dataset.get('rejected_count', len(dataset.get('rejected', [])))} 条\n"
            f"- 总时长：{dataset.get('total_duration', manifest.get('total_duration', 0)):.1f} 秒\n"
            f"- 目录：`{dataset.get('path', '')}`\n\n"
            '训练集与验证集按音频哈希确定性划分；原始录音只复制不重编码。')


def create_training_job(dataset_dir, base_model, output_dir, name, params_text, role_id):
    from timbreforge import training
    if not dataset_dir:
        raise ValueError('请先准备数据集。')
    params = {}
    if params_text and params_text.strip():
        try:
            params = json.loads(params_text)
        except ValueError as exc:
            raise ValueError(f'训练参数必须是 JSON：{exc}') from exc
    job = training.create_training_job(dataset_dir, base_model=base_model,
                                       output_dir=output_dir, name=name, params=params,
                                       role_id=role_id or None)
    return job


def run_training_job(job_dir):
    from timbreforge import training

    def task(context):
        result = training.run_training(job_dir)
        return {'job': result.get('job', {}), 'status': result.get('status', 'completed')}

    return ui_tasks.run_heavy('training', task, payload={'job_dir': job_dir}, timeout=24 * 3600)


def training_status_markdown(job):
    if not job:
        return '尚未创建训练任务。'
    return (f"### 训练任务 {job.get('name', '')}\n\n"
            f"- 状态：`{job.get('status')}`\n"
            f"- 基础模型：`{job.get('base_model', {}).get('path', '')}`\n"
            f"- 估计显存：{job.get('vram_estimate_gb', 0)} GiB（启发式估算，不是保证）\n"
            f"- 数据集：{job.get('dataset', {}).get('total_duration', 0):.1f} 秒 / "
            f"{job.get('dataset', {}).get('clips', 0)} 段\n"
            f"- 参数：`{json.dumps(job.get('params', {}), ensure_ascii=False)}`\n"
            f"- 检查点：`{job.get('checkpoint_dir') or '尚未生成'}`\n"
            f"- 评估：`{job.get('evaluation') or '尚未评估'}`\n"
            '- 训练是离线高级功能：不覆盖 Base 模型，失败不会破坏已有角色。')


def record_evaluation(job_dir, metrics_text):
    from timbreforge import training
    if not metrics_text or not metrics_text.strip():
        raise ValueError('请填写评估指标 JSON。')
    try:
        metrics = json.loads(metrics_text)
    except ValueError as exc:
        raise ValueError(f'评估指标必须是 JSON：{exc}') from exc
    result = training.record_evaluation(job_dir, metrics)
    return training_status_markdown(result['job'])


def register_trained(job_dir, role_id, confirmed):
    from timbreforge import training
    if confirmed is not True:
        raise ValueError('注册训练模型前，请确认已检查固定测试集与授权。')
    result = training.register_trained_model(
        job_dir, bind_role_id=role_id or None, confirmed=True)
    card = result.get('card')
    message = (f"已注册训练模型「{result['entry']['name']}」@ {result['entry']['revision']}。")
    if card:
        message += f" 已绑定到角色「{card['name']}」（版本 v{card['version']}）。"
    return message


def rollback_trained(role_id):
    from timbreforge import training
    if not role_id:
        raise ValueError('请先选择角色。')
    result = training.rollback_model(role_id)
    return '已回滚模型版本，旧 Clone Prompt 已失效。'


def build_admin_tabs(gr):
    """Settings, model management and the advanced training page."""
    initial_roles = role_choices()
    initial_value = initial_roles[0][1] if initial_roles else None
    initial_entries = entry_choices()

    with gr.Tab('模型管理', id='model_management'):
        gr.Markdown('## 模型管理\n登记本机已有模型目录与版本，随时切换或回滚；'
                    '不下载模型，也不删除磁盘文件。')
        table = gr.Dataframe(headers=['名称', '版本', '类型', '体积', '路径', '配置哈希', '登记时间'],
                             value=model_rows(), interactive=False,
                             elem_id='models_table')
        with gr.Row():
            entry = gr.Dropdown(label='模型记录', choices=initial_entries, scale=4,
                                elem_id='models_entry')
            refresh = gr.Button('刷新', elem_id='models_refresh')
            verify = gr.Button('检查文件', elem_id='models_verify')
            remove = gr.Button('移出注册表', variant='stop', elem_id='models_remove')
        with gr.Accordion('登记本机模型目录', open=False):
            with gr.Row():
                new_name = gr.Textbox(label='名称', elem_id='models_name')
                new_revision = gr.Textbox(label='版本标记', value='local', elem_id='models_revision')
                new_kind = gr.Dropdown(label='类型', choices=KIND_LABELS, value='tts_base',
                                       elem_id='models_kind')
            new_path = gr.Textbox(label='模型目录绝对路径（需已存在 config.json 与权重）',
                                  elem_id='models_path')
            new_notes = gr.Textbox(label='备注', elem_id='models_notes')
            register = gr.Button('登记模型', elem_id='models_register')
        with gr.Accordion('角色模型切换与回滚', open=True):
            with gr.Row():
                role = gr.Dropdown(label='选择角色', choices=initial_roles, value=initial_value,
                                   scale=4, elem_id='models_role')
                activate = gr.Button('切换到所选模型', elem_id='models_activate')
                rollback = gr.Button('回滚上一版本', elem_id='models_rollback')
        status = gr.Markdown('切换模型会使该角色的 Clone Prompt 失效；下次生成会回退到普通克隆。',
                             elem_id='models_status')

        def refresh_models(current=None):
            rows = model_rows()
            choices = entry_choices()
            ids = [item[1] for item in choices]
            value = current if current in ids else (ids[0] if ids else None)
            return rows, gr.update(choices=choices, value=value)

        refresh.click(refresh_models, entry, [table, entry], api_name='models_refresh')

        def do_register(name_value, path_value, revision_value, kind_value, notes_value):
            try:
                message = register_model(name_value, path_value, revision_value, kind_value,
                                         notes_value)
            except (OSError, ValueError, ModelRegistryError) as exc:
                return f'登记失败：{exc}', gr.update(), gr.update()
            rows, update = refresh_models()
            return message, rows, update

        register.click(do_register,
                       [new_name, new_path, new_revision, new_kind, new_notes],
                       [status, table, entry], api_name='models_register')

        def do_verify(entry_id):
            try:
                return verify_model(entry_id)
            except (OSError, ValueError, ModelRegistryError) as exc:
                return f'检查失败：{exc}'

        verify.click(do_verify, entry, status, api_name='models_verify')

        def do_remove(entry_id):
            try:
                return remove_model(entry_id)
            except (OSError, ValueError, ModelRegistryError) as exc:
                return f'移除失败：{exc}'

        remove.click(do_remove, entry, status, api_name='models_remove')

        def do_activate(role_id, entry_id):
            try:
                return activate_model(role_id, entry_id)
            except (OSError, ValueError, ModelRegistryError) as exc:
                return f'切换失败：{exc}'

        activate.click(do_activate, [role, entry], status, api_name='models_activate')

        def do_rollback(role_id):
            try:
                return rollback_role(role_id)
            except (OSError, ValueError, ModelRegistryError) as exc:
                return f'回滚失败：{exc}'

        rollback.click(do_rollback, role, status, api_name='models_rollback')

    with gr.Tab('社区适配器', id='community_adapters'):
        gr.Markdown(
            '## 社区适配器\n导入本地 PEFT LoRA 文件夹或 ZIP，校验后绑定到角色。'
            '不会执行适配器目录中的脚本，也不会自动下载文件；当前支持 Qwen3-TTS LoRA。')
        adapter_table = gr.Dataframe(
            headers=['名称', 'Base 模型', 'rank', 'alpha', '默认 scale', '体积', '作者', '许可证', '哈希'],
            value=adapter_rows(), interactive=False, elem_id='adapters_table')
        with gr.Row():
            adapter = gr.Dropdown(label='适配器', choices=adapter_choices(), scale=4,
                                  elem_id='adapters_entry')
            adapter_refresh = gr.Button('刷新', elem_id='adapters_refresh')
            adapter_verify = gr.Button('校验文件', elem_id='adapters_verify')
            adapter_remove = gr.Button('移除记录', variant='stop', elem_id='adapters_remove')
        with gr.Accordion('导入社区 LoRA', open=True):
            adapter_file = gr.File(label='本地 LoRA 文件夹或 ZIP（adapter_config.json + adapter_model.safetensors）',
                                   type='filepath', file_types=['.zip'], elem_id='adapters_file')
            adapter_folder = gr.Textbox(label='或填写已解压的适配器文件夹绝对路径',
                                        elem_id='adapters_folder')
            with gr.Row():
                adapter_name = gr.Textbox(label='名称', elem_id='adapters_name')
                adapter_author = gr.Textbox(label='作者', elem_id='adapters_author')
                adapter_license = gr.Textbox(label='许可证', value='未知', elem_id='adapters_license')
            with gr.Row():
                adapter_source = gr.Textbox(label='来源 URL', elem_id='adapters_source')
                adapter_revision = gr.Textbox(label='Base 版本/commit', elem_id='adapters_revision')
                adapter_scale = gr.Slider(0, 2, value=0.3, step=0.05, label='默认 LoRA scale', elem_id='adapters_scale')
            adapter_notes = gr.Textbox(label='备注', elem_id='adapters_notes')
            adapter_import = gr.Button('导入并校验', variant='primary', elem_id='adapters_import')
        with gr.Accordion('绑定到角色', open=True):
            with gr.Row():
                adapter_role = gr.Dropdown(label='角色', choices=initial_roles, value=initial_value,
                                           scale=4, elem_id='adapters_role')
                bind_scale = gr.Slider(0, 2, value=0.3, step=0.05, label='本角色 scale', elem_id='adapters_bind_scale')
                adapter_bind = gr.Button('绑定适配器', elem_id='adapters_bind')
                adapter_unbind = gr.Button('解除绑定', elem_id='adapters_unbind')
        adapter_status = gr.Markdown(
            '适配器会在独立 Qwen worker 中加载；哈希或 Base 不匹配时会拒绝生成，不会静默回退。',
            elem_id='adapters_status')

        def refresh_adapters(current=None):
            choices = adapter_choices()
            ids = [item[1] for item in choices]
            return adapter_rows(), gr.update(choices=choices, value=current if current in ids else (ids[0] if ids else None))

        adapter_refresh.click(refresh_adapters, adapter, [adapter_table, adapter], api_name='adapters_refresh')

        def do_import(file_path, folder_path, name, author, license_value, source, revision, scale, notes):
            try:
                message = import_adapter(folder_path or file_path, name, author, source, license_value,
                                         notes, revision, scale)
            except (OSError, ValueError, AdapterError, TypeError) as exc:
                return f'导入失败：{exc}', gr.update(), gr.update()
            rows, update = refresh_adapters()
            return message, rows, update

        adapter_import.click(do_import,
                             [adapter_file, adapter_folder, adapter_name, adapter_author, adapter_license,
                              adapter_source, adapter_revision, adapter_scale, adapter_notes],
                             [adapter_status, adapter_table, adapter], api_name='adapters_import')

        def do_adapter_verify(adapter_id):
            try:
                return verify_adapter(adapter_id)
            except (OSError, ValueError, AdapterError) as exc:
                return f'校验失败：{exc}'

        adapter_verify.click(do_adapter_verify, adapter, adapter_status, api_name='adapters_verify')

        def do_adapter_remove(adapter_id):
            try:
                message = remove_adapter(adapter_id)
            except (OSError, ValueError, AdapterError) as exc:
                return f'移除失败：{exc}', gr.update(), gr.update()
            rows, update = refresh_adapters()
            return message, rows, update

        adapter_remove.click(do_adapter_remove, adapter,
                             [adapter_status, adapter_table, adapter], api_name='adapters_remove')

        def do_adapter_bind(role_id, adapter_id, scale):
            try:
                return bind_adapter(role_id, adapter_id, scale)
            except (OSError, ValueError, AdapterError) as exc:
                return f'绑定失败：{exc}'

        adapter_bind.click(do_adapter_bind, [adapter_role, adapter, bind_scale], adapter_status,
                           api_name='adapters_bind')

        def do_adapter_unbind(role_id):
            try:
                return unbind_adapter(role_id)
            except (OSError, ValueError, AdapterError) as exc:
                return f'解除失败：{exc}'

        adapter_unbind.click(do_adapter_unbind, adapter_role, adapter_status,
                             api_name='adapters_unbind')

    with gr.Tab('设置与任务', id='settings_jobs'):
        gr.Markdown('## 设置与任务\n检查本机路径、内存与任务记录；清理只删除可重建的预览和临时文件。')
        settings = gr.Markdown(settings_markdown(), elem_id='settings_paths')
        with gr.Row():
            refresh_settings = gr.Button('刷新内存与任务', elem_id='settings_refresh')
            clean_preview = gr.Button('清理历史预览', elem_id='settings_clean_preview')
            clean_temp = gr.Button('清理失败临时文件', elem_id='settings_clean_temp')
        jobs = gr.Dataframe(headers=['类型', '状态', '角色', '说明', '错误', '创建时间'],
                            value=ui_tasks.job_rows(), interactive=False,
                            elem_id='settings_jobs')
        settings_status = gr.Markdown('', elem_id='settings_status')
        vc = gr.Markdown(vc_markdown(), elem_id='settings_vc')

        def do_refresh():
            return settings_markdown(), ui_tasks.job_rows(), vc_markdown()

        refresh_settings.click(do_refresh, None, [settings, jobs, vc],
                               api_name='settings_refresh')

        def do_clean_preview():
            try:
                return cleanup_previews(), ui_tasks.job_rows()
            except (OSError, ValueError) as exc:
                return f'清理失败：{exc}', ui_tasks.job_rows()

        clean_preview.click(do_clean_preview, None, [settings_status, jobs],
                            api_name='settings_clean_preview')

        def do_clean_temp():
            try:
                return cleanup_staging(), ui_tasks.job_rows()
            except (OSError, ValueError) as exc:
                return f'清理失败：{exc}', ui_tasks.job_rows()

        clean_temp.click(do_clean_temp, None, [settings_status, jobs],
                         api_name='settings_clean_temp')

    with gr.Tab('高级训练', id='advanced_training'):
        gr.Markdown(
            '## 高级训练（离线）\n只使用有合法使用权的录音；逐字稿准确、去除噪声与多人说话，'
            '训练/验证集分离。训练不覆盖 Base 模型，失败不会破坏已有角色。'
            '本页不影响普通生成流程，需要已配置外部训练命令。')
        with gr.Accordion('1. 数据集准备', open=True):
            manifest = gr.File(label='数据集清单 JSON：每项 {"audio": 路径, "transcript": 逐字稿}',
                               type='filepath', file_types=['.json'],
                               elem_id='training_manifest')
            dataset_dir = gr.Textbox(label='数据集输出目录（必须不存在或为空）',
                                     elem_id='training_dataset_dir')
            license_confirm = gr.Checkbox(label='我确认所有录音都有合法使用权与说话者授权',
                                          value=False, elem_id='training_license')
            prepare = gr.Button('准备数据集（切分 + 质量筛选）', elem_id='training_prepare')
            dataset_status = gr.Markdown('尚未准备数据集。', elem_id='training_dataset_status')
            dataset_state = gr.State(None)
        with gr.Accordion('2. 创建训练任务', open=True):
            base_model = gr.Textbox(label='本地基础模型目录（0.6B 优先，需已有权重）',
                                    elem_id='training_base_model')
            job_output = gr.Textbox(label='训练任务输出目录', elem_id='training_job_dir')
            job_name = gr.Textbox(label='训练任务名称', elem_id='training_name')
            job_params = gr.Textbox(
                label='训练参数 JSON（rank/alpha/epochs/learning_rate/batch_size/...）',
                value='{"rank": 16, "alpha": 32, "epochs": 5, "learning_rate": 0.0001, '
                      '"batch_size": 1}', elem_id='training_params')
            train_role = gr.Dropdown(label='训练后要绑定的角色（可选）', choices=initial_roles,
                                     value=initial_value, elem_id='training_role')
            create_job = gr.Button('创建训练任务', elem_id='training_create')
            run_job = gr.Button('运行训练（离线，需要外部训练器）', elem_id='training_run')
            job_status = gr.Markdown('尚未创建训练任务。', elem_id='training_job_status')
            job_state = gr.State(None)

        def do_prepare(path, output, confirmed):
            try:
                dataset = prepare_dataset_from_manifest(path, output, confirmed)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return f'数据集准备失败：{exc}', None
            return dataset_markdown(dataset), dataset

        prepare.click(do_prepare, [manifest, dataset_dir, license_confirm],
                      [dataset_status, dataset_state], api_name='training_prepare')

        def do_create(dataset, model_path, output, name, params, role_id):
            if not dataset:
                return '请先准备数据集。', None
            try:
                job = create_training_job(dataset['path'], model_path, output, name, params, role_id)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return f'创建训练任务失败：{exc}', None
            return training_status_markdown(job), job

        create_job.click(do_create, [dataset_state, base_model, job_output, job_name, job_params,
                                     train_role],
                         [job_status, job_state], api_name='training_create')

        def do_run(job):
            if not job:
                return '请先创建训练任务。'
            try:
                result = run_training_job(job.get('path', ''))
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return f'训练失败：{exc}'
            payload = result or {}
            return training_status_markdown(payload.get('job') or job)

        run_job.click(do_run, job_state, job_status, api_name='training_run')

        with gr.Accordion('3. 评估、注册与回滚', open=True):
            metrics = gr.Textbox(
                label='评估指标 JSON（cer/wer/speaker_similarity/mos/notes）',
                value='{"cer": 0.05, "speaker_similarity": 0.8}', elem_id='training_metrics')
            record = gr.Button('记录评估', elem_id='training_record')
            register = gr.Button('注册训练模型', elem_id='training_register')
            rollback = gr.Button('回滚角色模型', elem_id='training_rollback')
            bind_confirm = gr.Checkbox(label='我确认已检查固定测试集与授权，可以注册并绑定',
                                       value=False, elem_id='training_confirm')
            trained_status = gr.Markdown('', elem_id='training_trained_status')

        def do_record(job, text):
            if not job:
                return '请先创建训练任务。'
            try:
                return record_evaluation(job.get('path', ''), text)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return f'记录评估失败：{exc}'

        record.click(do_record, [job_state, metrics], trained_status,
                     api_name='training_record')

        def do_register(job, role_id, confirmed):
            if not job:
                return '请先创建训练任务。'
            try:
                return register_trained(job.get('path', ''), role_id, confirmed)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return f'注册失败：{exc}'

        register.click(do_register, [job_state, train_role, bind_confirm], trained_status,
                       api_name='training_register')

        def do_rollback(role_id):
            try:
                return rollback_trained(role_id)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return f'回滚失败：{exc}'

        rollback.click(do_rollback, train_role, trained_status, api_name='training_rollback')

