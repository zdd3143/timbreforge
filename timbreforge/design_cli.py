"""Prompt/seed VoiceDesign command interface; no model imports before actions."""
import argparse
import json


def main(argv=None):
    parser=argparse.ArgumentParser(description='Qwen VoiceDesign 声线设计候选与固定角色')
    parser.add_argument('--config')
    parser.add_argument('--root')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    for command in ('create','append','show','select','save-character'):
        p=sub.add_parser(command)
        if command!='create':p.add_argument('--session',required=True)
        if command=='create':
            p.add_argument('--prompt',required=True)
            p.add_argument('--text',required=True)
        if command in ('create','append'):
            p.add_argument('--seed',type=int,default=42)
            p.add_argument('--count',type=int,choices=range(1,5),default=1)
            p.add_argument('--language',default='Chinese')
            p.add_argument('--temperature',type=float,default=.9)
            p.add_argument('--top-p',type=float,default=1.)
            p.add_argument('--top-k',type=int,default=50)
            p.add_argument('--repetition-penalty',type=float,default=1.05)
            p.add_argument('--max-new-tokens',type=int,default=1024)
            p.add_argument('--subtalker-temperature',type=float,default=.9)
            p.add_argument('--subtalker-top-p',type=float,default=1.)
            p.add_argument('--subtalker-top-k',type=int,default=50)
            p.add_argument('--greedy',action='store_true',help='关闭主talker与subtalker采样；种子不再制造随机候选')
            p.add_argument('--device',choices=['cuda:0','cpu'],default='cuda:0')
            p.add_argument('--confirm-design',action='store_true')
        if command in ('select','save-character'):
            p.add_argument('--candidate',required=True)
        if command=='select':
            p.add_argument('--favorite',action=argparse.BooleanOptionalAction,default=None)
            p.add_argument('--note',default=None)
        if command=='save-character':
            p.add_argument('--name',required=True)
            p.add_argument('--confirm-transcript',action='store_true',help='确认已选音频逐字对应试听台词且可用作角色参考')
    args=parser.parse_args(argv)
    try:
        if args.config:
            from timbreforge.config import set_config_path
            set_config_path(args.config)
        from timbreforge.designs import DesignStore
        store=DesignStore(args.root)
        if args.command in ('create','append'):
            from timbreforge.backend_qwen import QwenParams
            params=QwenParams(seed=args.seed,language=args.language,temperature=args.temperature,
                top_p=args.top_p,top_k=args.top_k,repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,subtalker_temperature=args.subtalker_temperature,
                subtalker_top_p=args.subtalker_top_p,subtalker_top_k=args.subtalker_top_k,
                do_sample=not args.greedy,subtalker_dosample=not args.greedy)
            kwargs=dict(params=params,count=args.count,device=args.device,confirmed=args.confirm_design)
            result=store.create(args.prompt,args.text,**kwargs) if args.command=='create' else store.append(args.session,**kwargs)
        elif args.command=='list':result=store.list()
        elif args.command=='show':result=store.load(args.session)
        elif args.command=='select':result=store.select(args.session,args.candidate,favorite=args.favorite,note=args.note)
        else:result=store.save_character(args.session,args.candidate,args.name,confirmed=args.confirm_transcript)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except (OSError,ValueError,RuntimeError,TypeError,KeyError) as exc:
        parser.exit(1,f'{type(exc).__name__}: {exc}\n')


if __name__=='__main__':raise SystemExit(main())
