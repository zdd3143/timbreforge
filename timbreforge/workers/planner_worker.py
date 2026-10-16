"""One local Qwen language-model process per voice plan; no remote model code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from timbreforge.role_planner import PLAN_SYSTEM_PROMPT, parse_plan, validate_profile


def run(request):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    profile = validate_profile(request['profile'])
    profile_sha = hashlib.sha256(profile.encode('utf-8')).hexdigest()
    if profile_sha != request['profile_sha256']:
        raise ValueError('角色设定哈希不一致')
    model_dir = str(Path(request['model_dir']).resolve())
    cuda = torch.cuda.is_available()
    device = 'cuda:0' if cuda else 'cpu'
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
    start = time.perf_counter()
    print(f'Loading planner on {device}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    # CPU staging avoids Windows CUDA safetensors mapping failures. Generation uses the selected device.
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='cpu', dtype=dtype,
        attn_implementation='sdpa', local_files_only=True, trust_remote_code=False).to(device).eval()
    loaded = time.perf_counter()
    torch.manual_seed(42)
    messages = [{'role':'system', 'content':PLAN_SYSTEM_PROMPT},
                {'role':'user', 'content':'请为以下角色设定设计声线：\n' + profile}]
    last_error = None
    for attempt in range(2):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(rendered, return_tensors='pt').to(device)
        if inputs.input_ids.shape[-1] > 10000:
            raise ValueError('角色设定超出模型输入上限，请精简后重试。')
        print(f'Generating voice plan, attempt {attempt + 1}', flush=True)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=1600, do_sample=True,
                temperature=.7, top_p=.8, top_k=20, repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        try:
            plan = parse_plan(answer)
            return dict(plan=plan, model_dir=model_dir, profile_sha256=profile_sha,
                device=device, dtype=str(dtype), load_seconds=loaded-start,
                inference_seconds=time.perf_counter()-loaded, attempts=attempt+1,
                model_config_sha256=hashlib.sha256((Path(model_dir)/'config.json').read_bytes()).hexdigest())
        except (ValueError, TypeError) as exc:
            last_error = exc
            messages += [{'role':'assistant','content':answer},
                         {'role':'user','content':'请修正为要求的 JSON 对象，补全字段，不要代码块或额外文字。错误：' + str(exc)}]
    raise ValueError(f'模型未返回有效声线方案，请重试：{last_error}')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args(argv)
    try:
        result = run(json.loads(Path(args.request).read_text(encoding='utf-8')))
        code = 0
    except Exception as exc:
        traceback.print_exc()
        result, code = {'error':f'{type(exc).__name__}: {exc}'}, 1
    Path(args.result).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
