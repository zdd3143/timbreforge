"""Offline-safe command surface for projects and reference preparation."""
from __future__ import annotations

import argparse
import json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='TimbreForge 多角色创作与参考音频助手')
    parser.add_argument('--config', default=None, help='TOML configuration file (before subcommand)')
    parser.add_argument('--root', default=None, help='Project store directory')
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    create.add_argument('--name', required=True)
    create.add_argument('--script', required=True, help='Dialogue script text, not a filename')
    sub.add_parser('list')
    for command in ('show', 'bind', 'generate', 'review', 'assess', 'export'):
        p = sub.add_parser(command)
        p.add_argument('--project', required=True, help='Stable project ID')
        if command == 'bind':
            p.add_argument('--speaker', required=True)
            p.add_argument('--character', required=True, help='Stable character ID')
        if command in ('generate', 'review', 'assess'):
            p.add_argument('--line', required=True, help='Stable line ID, not row number')
        if command == 'generate':
            p.add_argument('--count', type=int, choices=range(1, 5), default=1)
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--device', choices=['cuda:0', 'cpu'], default='cuda:0',
                           help='Qwen only; ignored by other backends; no automatic fallback')
            p.add_argument('--confirm-rights', action='store_true')
        if command == 'review':
            p.add_argument('--candidate', required=True)
            p.add_argument('--selected', action=argparse.BooleanOptionalAction, default=None)
            p.add_argument('--favorite', action=argparse.BooleanOptionalAction, default=None)
            p.add_argument('--issue', action='append', default=None, help='Repeat for multiple issues')
            p.add_argument('--clear-issues', action='store_true')
            p.add_argument('--note', default=None)
        if command == 'assess':
            p.add_argument('--candidate', required=True)
            p.add_argument('--observed-text', default=None, help='Human transcription; no automatic ASR')
            p.add_argument('--run-speaker', action='store_true', help='Opt in to local CAM++')
            p.add_argument('--run-quality', action='store_true', help='Opt in to local MOS')
        if command == 'export':
            p.add_argument('--gap-seconds', type=float, default=.25)
    for command in ('inspect-reference', 'prepare-reference'):
        p = sub.add_parser(command)
        p.add_argument('--path', required=True)
        p.add_argument('--transcript', default='')
        if command == 'inspect-reference':
            p.add_argument('--backend', choices=['indextts', 'cosyvoice3'], default='indextts')
            p.add_argument('--instruction', default='')
        else:
            p.add_argument('--confirm-processing', action='store_true', help='Explicitly approve the requested trim/conversion')
            p.add_argument('--start-seconds', type=float, default=0)
            p.add_argument('--end-seconds', type=float, default=None)
            p.add_argument('--mono', action='store_true')
            p.add_argument('--output-dir', default=None)
            p.add_argument('--confirm-transcript', action='store_true', help='Confirm transcript matches the prepared clip')
    # Accept --root either globally or after a subcommand without clobbering it.
    for command_parser in sub.choices.values():
        command_parser.add_argument('--root', default=argparse.SUPPRESS, help='Project store directory')
    args = parser.parse_args(argv)
    try:
        if args.config:
            from timbreforge.config import set_config_path, get_config
            set_config_path(args.config)
            get_config()
        if args.command == 'generate' and not args.confirm_rights:
            raise ValueError('Generation requires --confirm-rights (参考声音使用授权)')
        if args.command == 'prepare-reference' and not args.confirm_processing:
            raise ValueError('Reference processing requires --confirm-processing; original audio is never overwritten')
        if args.command in ('inspect-reference', 'prepare-reference'):
            from timbreforge.reference_assistant import inspect_reference, prepare_confirmed_reference
            if args.command == 'inspect-reference':
                result = inspect_reference(args.path, transcript=args.transcript,
                                           backend=args.backend, instruction=args.instruction)
            else:
                result = prepare_confirmed_reference(
                    args.path, confirmed=args.confirm_processing, start_seconds=args.start_seconds,
                    end_seconds=args.end_seconds, mono=args.mono, output_dir=args.output_dir,
                    transcript=args.transcript, transcript_confirmed=args.confirm_transcript)
        else:
            from timbreforge.projects import ProjectStore
            store = ProjectStore(args.root)
            if args.command == 'create':
                result = store.create(args.name, args.script)
            elif args.command == 'list':
                result = store.list()
            elif args.command == 'show':
                result = store.load(args.project)
            elif args.command == 'bind':
                result = store.bind(args.project, args.speaker, args.character)
            elif args.command == 'generate':
                result = store.generate_line(args.project, args.line, count=args.count,
                                             seed=args.seed, confirmed=args.confirm_rights, device=args.device)
            elif args.command == 'assess':
                result = store.assess(args.project, args.line, args.candidate,
                                      observed_text=args.observed_text or None,
                                      run_speaker=args.run_speaker, run_quality=args.run_quality)
            elif args.command == 'review':
                result = store.review(args.project, args.line, args.candidate,
                                      selected=args.selected, favorite=args.favorite,
                                      issues=[] if args.clear_issues else args.issue, note=args.note)
            else:
                result = store.export(args.project, gap_seconds=args.gap_seconds)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
