"""Task-oriented CLI; legacy voicegen remains available for bank retrieval."""
from __future__ import annotations
import argparse
import json
from timbreforge.config import set_config_path, get_config
from timbreforge.characters import CharacterStore
from timbreforge.studio import GenerationRequest, capabilities, generate


def main(argv=None) -> int:
    # Keep the primary entry point while allowing a complete nested creator CLI.
    import sys
    values = list(sys.argv[1:] if argv is None else argv)
    prefix = []
    if len(values) >= 2 and values[0] == '--config':
        prefix, values = values[:2], values[2:]
    if values and values[0] == 'design':
        from timbreforge.design_cli import main as design_main
        return design_main(prefix + values[1:])
    if values and values[0] == 'create':
        from timbreforge.creation_cli import main as creation_main
        return creation_main(prefix + values[1:])
    parser = argparse.ArgumentParser(description='TimbreForge 克隆 / 指令 / 角色工作室；创作工具：create --help')
    parser.add_argument('--config', default=None, help='TOML configuration file')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('capabilities', help='本地能力状态（不加载模型、不访问网络）')
    for name in ('characters', 'save-character', 'generate'):
        p = sub.add_parser(name)
        p.add_argument('--store', default=None, help='角色库目录，默认使用配置 characters_dir')
        if name == 'characters':
            continue
        if name == 'save-character':
            p.add_argument('--name', required=True)
            p.add_argument('--reference', required=True)
            p.add_argument('--ref-text', default='')
            p.add_argument('--description', default='')
            p.add_argument('--instruction', default='')
            p.add_argument('--backend', choices=['indextts', 'cosyvoice3', 'qwen3'], default='indextts')
        else:
            p.add_argument('--task', choices=['clone', 'instruct', 'character', 'design'], default='clone')
            p.add_argument('--backend', choices=['indextts', 'cosyvoice3', 'qwen3'], default='indextts')
            p.add_argument('--text', required=True)
            p.add_argument('--reference')
            p.add_argument('--ref-text', default='')
            p.add_argument('--instruction', default='')
            p.add_argument('--character')
            p.add_argument('--save-name', default='')
            p.add_argument('--description', default='')
            p.add_argument('--seed', type=int, default=None, help='不填时角色复用保留存档配置')
            p.add_argument('--base-url', default='')
            p.add_argument('--output-dir', default=None)
            p.add_argument('--device', choices=['cuda:0','cpu'], default='cuda:0', help='Qwen设备，CPU为显式慢速模式')
            p.add_argument('--qwen-dtype', choices=['bfloat16','float16','float32'], default='bfloat16', help='Qwen Base GPU精度，默认BF16避免FP16不稳定；CPU固定FP32')
        p.add_argument('--confirm-rights', action='store_true', help='确认拥有参考声音使用许可')
    args = parser.parse_args(argv)
    try:
        if args.config:
            set_config_path(args.config)
            get_config()  # Validate before any command-side effects.
        if args.command == 'capabilities':
            result = capabilities()
        else:
            store = CharacterStore(args.store) if args.store else CharacterStore(get_config().characters_dir)
            if args.command == 'characters':
                result = store.list()
            else:
                if not args.confirm_rights:
                    parser.error('请确认参考音频的使用许可，并添加 --confirm-rights')
                if args.command == 'save-character':
                    result = store.create(args.name, args.reference, reference_text=args.ref_text,
                                          description=args.description, backend=args.backend,
                                          instruction=args.instruction)
                else:
                    request = GenerationRequest(task=args.task, backend=args.backend, text=args.text,
                                                reference_audio=args.reference, reference_text=args.ref_text,
                                                instruction=args.instruction, character_id=args.character,
                                                seed_override=args.seed, base_url=args.base_url, device=args.device, qwen_dtype=args.qwen_dtype,
                                                save_character_name=args.save_name, description=args.description)
                    kwargs = {'character_store': store}
                    if args.output_dir:
                        kwargs['output_dir'] = args.output_dir
                    result = generate(request, **kwargs)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
