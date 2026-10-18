"""Portable Qwen adapter/worker tests: fake runtime, real temporary WAVs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from timbreforge import backend_qwen as q
from timbreforge.workers import qwen_worker as worker


class AdapterTests(unittest.TestCase):
    def setUp(self):
        for target in ('timbreforge.resources.release_idle_models', 'timbreforge.backend_qwen.release_idle_models'):
            cleanup=patch(target);cleanup.start();self.addCleanup(cleanup.stop)
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.model=self.root/'model'; self.model.mkdir()
        (self.model/'config.json').write_text('{}',encoding='utf-8')
        (self.model/'model.safetensors').write_bytes(b'mocked')
        self.python=self.root/'python.exe';self.python.write_bytes(b'mocked')
        self.ref=self.root/'ref.wav';sf.write(self.ref,np.full(2400,.1),24000,subtype='FLOAT')
        self.cfg=types.SimpleNamespace(qwen_python=self.python,qwen_design_model=self.model,qwen_base_model=self.model,outputs_dir=self.root/'out')
        self.config_patch=patch.object(q,'get_config',return_value=self.cfg);self.config_patch.start();self.addCleanup(self.config_patch.stop)

    def result(self,command,**kwargs):
        self.request=json.loads(Path(command[command.index('--request')+1]).read_text(encoding='utf-8'))
        r=self.request
        sf.write(r['output'],np.full(2400,.2),24000,subtype='FLOAT')
        result={k:r[k] for k in ('mode','params','model_dir','device','dtype','attention','model_config_sha256')}
        result.update(file=r['output'],seed=r['params']['seed'],sample_rate=24000,duration=.1,
                      clone_mode=r.get('clone_mode','none'),clone_prompt_sha256=None,
                      qwen_package_version='0.1.1',reference_sha256=r['reference_sha256'],load_seconds=.1,inference_seconds=.2)
        Path(command[command.index('--result')+1]).write_text(json.dumps(result),encoding='utf-8')
        return types.SimpleNamespace(returncode=0)

    def test_design_file_protocol_and_effective_params(self):
        with patch.object(q.subprocess,'run',side_effect=self.result) as run:
            result=q.design('你好','温暖男声',q.QwenParams(seed=123,max_new_tokens=512))
        self.assertEqual(result['seed'],123);self.assertEqual(sf.info(result['file']).subtype,'FLOAT')
        self.assertEqual(self.request['instruction'],'温暖男声')
        self.assertEqual(self.request['params']['max_new_tokens'],512)
        self.assertEqual(run.call_args.kwargs['env']['HF_HUB_OFFLINE'],'1')
        self.assertNotIn('capture_output',run.call_args.kwargs)
        self.assertNotIn('guidance_scale',self.request['params'])

    def test_clone_exact_transcript_and_original_hash(self):
        before=self.ref.read_bytes()
        with patch.object(q.subprocess,'run',side_effect=self.result):
            result=q.clone(self.ref,'参考台词','新台词',q.QwenParams())
        self.assertEqual(self.request['ref_text'],'参考台词')
        self.assertEqual(result['reference_sha256'],hashlib.sha256(before).hexdigest())
        self.assertEqual(self.ref.read_bytes(),before)

    def test_bad_requests_fail_before_subprocess(self):
        with patch.object(q.subprocess,'run') as run:
            for kwargs in ({'device':'cpu'},{'device':'mps'},{'timeout':float('nan')},{'attention':'CFG'}):
                with self.assertRaises(ValueError):q.design('text','prompt',q.QwenParams(),**kwargs)
            with self.assertRaises(ValueError):q.clone(self.ref,'','text',q.QwenParams())
            (self.model/'model.safetensors').unlink()
            with self.assertRaises(FileNotFoundError):q.design('text','prompt',q.QwenParams())
        run.assert_not_called()

    def test_result_provenance_and_audio_rejected(self):
        for tamper in ('file','seed','params','sample_rate','qwen_package_version','nonfinite','pcm','nan_duration'):
            def bad(command,**kwargs):
                completed=self.result(command,**kwargs)
                path=Path(command[command.index('--result')+1]);r=json.loads(path.read_text())
                if tamper=='nonfinite':sf.write(r['file'],[float('nan')],24000,subtype='FLOAT')
                elif tamper=='pcm':sf.write(r['file'],[.1],24000,subtype='PCM_16')
                elif tamper=='params':r[tamper]={}
                elif tamper=='nan_duration':r['duration']=float('nan')
                else:r[tamper]='tampered'
                path.write_text(json.dumps(r));return completed
            with self.subTest(tamper=tamper),patch.object(q.subprocess,'run',side_effect=bad):
                with self.assertRaises((RuntimeError,ValueError)):q.design('text','prompt',q.QwenParams())

    def test_worker_error_timeout_and_reference_mutation(self):
        def error(command,**kwargs):
            Path(command[command.index('--result')+1]).write_text(json.dumps({'error':'CUDA out of memory'}))
            return types.SimpleNamespace(returncode=1)
        with patch.object(q.subprocess,'run',side_effect=error):
            with self.assertRaisesRegex(RuntimeError,'no automatic CPU fallback'):q.design('text','prompt',q.QwenParams())
        with patch.object(q.subprocess,'run',side_effect=subprocess.TimeoutExpired('qwen',1)):
            with self.assertRaises(TimeoutError):q.design('text','prompt',q.QwenParams())
        def mutate(command,**kwargs):
            result=self.result(command,**kwargs);sf.write(self.ref,np.full(2400,.2),24000,subtype='FLOAT');return result
        with patch.object(q.subprocess,'run',side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError,'Reference changed'):q.clone(self.ref,'ref','text',q.QwenParams())

    def test_oom_retries_once_with_same_recipe_and_nonmemory_errors_do_not_retry(self):
        attempts=[]
        def once(command, **kwargs):
            attempts.append(command)
            if len(attempts) == 1:
                Path(command[command.index('--result')+1]).write_text(json.dumps({'error':'CUDA out of memory'}))
                return types.SimpleNamespace(returncode=1)
            return self.result(command, **kwargs)
        with patch.object(q.subprocess,'run',side_effect=once), patch.object(q,'release_idle_models'):
            result=q.clone(self.ref,'ref','line',q.QwenParams(seed=123))
        self.assertEqual(result['attempts'],2)
        self.assertEqual(result['seed'],123)
        self.assertEqual(attempts[0],attempts[1])
        def invalid(command, **kwargs):
            Path(command[command.index('--result')+1]).write_text(json.dumps({'error':'invalid audio'}))
            return types.SimpleNamespace(returncode=1)
        with patch.object(q.subprocess,'run',side_effect=invalid) as run:
            with self.assertRaisesRegex(RuntimeError,'invalid audio'):q.design('text','voice',q.QwenParams())
        self.assertEqual(run.call_count,1)

    def test_clone_prompt_creation_and_cached_clone_request(self):
        def prompt_worker(command, **kwargs):
            request = json.loads(Path(command[command.index('--request')+1]).read_text(encoding='utf-8'))
            Path(request['output']).write_bytes(b'prompt-tensors')
            digest = hashlib.sha256(b'prompt-tensors').hexdigest()
            result = dict(file=request['output'], mode='clone_prompt',
                          model_dir=request['model_dir'], device=request['device'],
                          dtype=request['dtype'], attention=request['attention'],
                          model_config_sha256=request['model_config_sha256'],
                          qwen_package_version='0.1.1',
                          reference_sha256=request['reference_sha256'],
                          clone_mode=request['clone_mode'], prompt_sha256=digest,
                          tensor_info={'ref_code': {'shape': [3, 2]},
                                       'ref_spk_embedding': {'shape': [8]}},
                          load_seconds=.1, inference_seconds=.2)
            Path(command[command.index('--result')+1]).write_text(json.dumps(result))
            return types.SimpleNamespace(returncode=0)
        with patch.object(q.subprocess,'run',side_effect=prompt_worker):
            created = q.create_clone_prompt(self.ref, '参考台词', clone_mode='icl')
        self.assertTrue(Path(created['prompt_file']).is_file())
        self.assertEqual(created['clone_mode'], 'icl')
        self.assertEqual(created['prompt_sha256'],
                         hashlib.sha256(b'prompt-tensors').hexdigest())
        self.assertEqual(created['tensor_info']['ref_code']['shape'], [3, 2])

        reference_hash = hashlib.sha256(self.ref.read_bytes()).hexdigest()
        def clone_worker(command, **kwargs):
            request = json.loads(Path(command[command.index('--request')+1]).read_text(encoding='utf-8'))
            self.request = request
            sf.write(request['output'], np.full(2400,.2), 24000, subtype='FLOAT')
            result = {k: request[k] for k in ('mode','params','model_dir','device','dtype',
                                              'attention','model_config_sha256')}
            result.update(file=request['output'], seed=request['params']['seed'],
                          sample_rate=24000, duration=.1, clone_mode='icl',
                          clone_prompt_sha256=request['clone_prompt_sha256'],
                          reference_sha256=request['reference_sha256'],
                          qwen_package_version='0.1.1', load_seconds=.1, inference_seconds=.2)
            Path(command[command.index('--result')+1]).write_text(json.dumps(result))
            return types.SimpleNamespace(returncode=0)
        with patch.object(q.subprocess,'run',side_effect=clone_worker):
            output = q.clone(None, '参考台词', '新台词', q.QwenParams(),
                             clone_prompt=created['prompt_file'], clone_mode='icl',
                             reference_sha256=reference_hash)
        self.assertIsNone(self.request['ref_audio'])
        self.assertEqual(self.request['clone_prompt'], created['prompt_file'])
        self.assertEqual(self.request['clone_prompt_sha256'], created['prompt_sha256'])
        self.assertEqual(output['clone_prompt_sha256'], created['prompt_sha256'])

    def test_clone_prompt_rejects_invalid_requests_before_subprocess(self):
        with patch.object(q.subprocess,'run') as run:
            with self.assertRaises(ValueError):
                q.create_clone_prompt(self.ref, '参考台词', clone_mode='magic')
            with self.assertRaises(FileNotFoundError):
                q.clone(None, '参考台词', '台词', q.QwenParams(),
                        clone_prompt=self.root/'missing.pt', clone_mode='icl',
                        reference_sha256='a'*64)
            with self.assertRaises(ValueError):
                q.clone(self.ref, '参考台词', '台词', q.QwenParams(),
                        clone_prompt=self.root/'prompt.pt', clone_mode='icl',
                        reference_sha256='a'*64)
        run.assert_not_called()

    def test_params_reject_bool_nan_and_unknown_language(self):
        for kwargs in ({'seed':True},{'seed':-1},{'temperature':float('nan')},{'top_p':0},{'top_k':1.5},
                       {'do_sample':1},{'subtalker_dosample':'yes'},{'max_new_tokens':8193},{'language':'invented'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):q.QwenParams(**kwargs).validate()


class WorkerTests(unittest.TestCase):
    def test_clone_chunks_preserve_text_and_bound_length(self):
        text='保持冷静，跟紧我。'*25 + '\n' + 'Long sentence ' * 40 + '结束。'
        parts=worker.clone_segments(text)
        self.assertEqual(''.join(parts),text)
        self.assertTrue(all(0 < len(part) <= 100 for part in parts))

    def test_actual_api_whitelist_rng_and_clone(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'config.json').write_text('{}');ref=root/'ref.wav'
            sf.write(ref,np.full(2400,.2),24000,subtype='FLOAT')
            events=[];calls=[]
            class Model:
                @classmethod
                def from_pretrained(cls,*args,**kwargs):
                    events.append('load');calls.append(('load',args,kwargs));return cls()
                def generate_voice_design(self,**kwargs):
                    events.append('generate');calls.append(('design',kwargs));return [np.full(2400,.25)],24000
                def generate_voice_clone(self,**kwargs):
                    events.append('generate');calls.append(('clone',kwargs));return [np.full(2400,.25)],24000
                def create_voice_clone_prompt(self,**kwargs):
                    calls.append(('prompt',kwargs));return ['reference-conditioning']
            cuda=types.SimpleNamespace(is_available=lambda:True,manual_seed_all=lambda v:events.append(('cuda_seed',v)),
                reset_peak_memory_stats=lambda device:None,max_memory_allocated=lambda device:123)
            torch=types.SimpleNamespace(set_num_threads=lambda n:None,cuda=cuda,float16='mock_fp16',float32='mock_fp32',manual_seed=lambda v:events.append(('seed',v)))
            params=q.QwenParams(seed=123,do_sample=False,subtalker_dosample=False).to_dict()
            base=dict(mode='voice_design',text='text',instruction='voice',params=params,model_dir=str(root),
                model_config_sha256=hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),
                output=str(root/'design.wav'),device='cuda:0',dtype='float16',attention='sdpa',reference_sha256=None)
            modeling = types.ModuleType('transformers.modeling_utils')
            modeling.safe_open = lambda *args, **kwargs: None
            transformers = types.ModuleType('transformers')
            transformers.modeling_utils = modeling
            with patch.dict('sys.modules',{'torch':torch,'qwen_tts':types.SimpleNamespace(Qwen3TTSModel=Model),
                 'transformers':transformers,'transformers.modeling_utils':modeling}), \
                 patch.object(worker.importlib.metadata,'version',return_value='0.1.1'):
                result=worker.run(base)
                self.assertLess(events.index('load'),events.index(('seed',123)))
                self.assertLess(events.index(('seed',123)),events.index('generate'))
                generation=calls[-1][1]
                self.assertEqual(set(generation),worker.GENERATION_KEYS|{'text','language','instruct','non_streaming_mode'})
                self.assertNotIn('seed',generation);self.assertTrue(generation['non_streaming_mode'])
                self.assertFalse(generation['subtalker_dosample'])
                self.assertEqual(calls[0][2]['attn_implementation'],'sdpa');self.assertTrue(calls[0][2]['local_files_only'])
                self.assertEqual(result['cuda_peak_memory_bytes'],123)
                if sys.platform == 'win32':
                    self.assertEqual(result['safetensors_load_staging'],'cpu_to_requested_device')
                    self.assertIsNone(modeling.safe_open('probe'))
                cloned=dict(base,mode='voice_clone',ref_audio=str(ref),ref_text='exact text',
                    reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),output=str(root/'clone.wav'))
                result=worker.run(cloned)
                self.assertEqual(calls[-1][1]['voice_clone_prompt'],['reference-conditioning'])
                prompt_call=next(c for c in calls if c[0]=='prompt')
                self.assertFalse(prompt_call[1]['x_vector_only_mode']);self.assertEqual(prompt_call[1]['ref_text'],'exact text')
                self.assertTrue(result['ref_text_used'])
                with self.assertRaises(FileExistsError):worker.run(cloned)


if __name__=='__main__':unittest.main()

