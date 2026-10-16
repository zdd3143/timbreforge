"""VoiceDesign session persistence with real temporary WAVs, mocked inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid
import numpy as np
import soundfile as sf
from timbreforge.backend_qwen import QwenParams
from timbreforge.designs import DesignStore


class DesignStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.store=DesignStore(self.root/'sessions');self.calls=[]
        self.backend=patch('timbreforge.backend_qwen.design',side_effect=self.render);self.backend.start();self.addCleanup(self.backend.stop)

    def render(self,text,prompt,params,**kwargs):
        self.calls.append((text,prompt,params.to_dict(),kwargs))
        source=self.root/(uuid.uuid4().hex+'.wav');sf.write(source,np.full(2400,.2),24000,subtype='FLOAT')
        return dict(file=str(source),mode='voice_design',seed=params.seed,params=params.to_dict(),sample_rate=24000,duration=.1)

    def create(self,**kwargs):
        return self.store.create('温暖男声','参考台词',QwenParams(seed=123),confirmed=True,**kwargs)

    def test_seed_candidates_manual_choice_and_persistence(self):
        p=self.create(count=2);sid=p['id']
        self.assertEqual([c['seed'] for c in p['candidates']],[123,124]);self.assertIsNone(p['selected_candidate_id'])
        selected=p['candidates'][0]['id'];p=self.store.select(sid,selected,favorite=True,note='人工选择')
        before=Path(self.store.audio_path(sid,selected)).read_bytes()
        p=self.store.append(sid,params=QwenParams(seed=900),confirmed=True)
        self.assertEqual(p['selected_candidate_id'],selected);self.assertEqual(p['candidates'][-1]['seed'],900)
        self.assertEqual(Path(self.store.audio_path(sid,selected)).read_bytes(),before)
        self.assertEqual(DesignStore(self.store.root).load(sid),p)
        self.assertTrue(p['candidates'][0]['favorite']);self.assertEqual(p['candidates'][0]['reviews'][0]['note'],'人工选择')

    def test_partial_failure_retains_success_then_retry(self):
        def renderer(*args,**kwargs):
            if len(self.calls)==1:raise RuntimeError('backend failed')
            return self.render(*args,**kwargs)
        with patch('timbreforge.backend_qwen.design',side_effect=renderer):
            with self.assertRaisesRegex(RuntimeError,'backend failed'):self.create(count=2)
        p=self.store.list()[0];self.assertEqual(len(p['candidates']),1);self.assertIsNone(p['selected_candidate_id'])
        p=self.store.append(p['id'],params=QwenParams(seed=500),confirmed=True)
        self.assertEqual(len(p['candidates']),2)

    def test_confirmation_and_argument_validation_no_writes(self):
        for kwargs in ({'confirmed':False},{'confirmed':1},{'count':5,'confirmed':True},{'device':'auto','confirmed':True}):
            with self.assertRaises(ValueError):self.store.create('voice','text',QwenParams(),**kwargs)
        self.assertFalse(self.store.root.exists())
        with self.assertRaises(ValueError):self.store.create('','text',QwenParams(),confirmed=True)
        with self.assertRaises(ValueError):self.store.create('voice','x'*501,QwenParams(),confirmed=True)

    def test_tampered_hash_path_selection_and_id_fail(self):
        p=self.create();sid=p['id'];manifest=self.store.root/sid/'session.json';original=manifest.read_text(encoding='utf-8')
        for mutate in (lambda d:d['candidates'][0].update(file='../outside.wav'),
                       lambda d:d.update(selected_candidate_id='a'*32),
                       lambda d:d.update(id='b'*32),
                       lambda d:d['candidates'][0].update(sha256='0'*64)):
            data=json.loads(original);mutate(data);manifest.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(ValueError):self.store.load(sid)
            self.assertEqual(self.store.list(),[])
        manifest.write_text(original,encoding='utf-8')
        with self.assertRaises(ValueError):self.store.load('../escape')
        source=Path(self.store.audio_path(sid,p['candidates'][0]['id']));source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):self.store.load(sid)

    def test_corrupt_results_seed_and_nonfinite_metadata_rejected(self):
        p=self.create();sid=p['id'];manifest=self.store.root/sid/'session.json';original=manifest.read_text(encoding='utf-8')
        for mutate in (lambda d:d['candidates'][0].update(result=[]),
                       lambda d:d['candidates'][0].update(seed=True),
                       lambda d:d['candidates'][0].update(seed=-1),
                       lambda d:d['candidates'][0].update(favorite='yes'),
                       lambda d:d['candidates'][0]['result'].update(duration=float('nan')),
                       lambda d:d.update(device='auto')):
            data=json.loads(original);mutate(data);manifest.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises((ValueError,TypeError)):self.store.load(sid)
            self.assertEqual(self.store.list(),[])
        manifest.write_text(original,encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'select'):
            self.store.save_character(sid,p['candidates'][0]['id'],'name',confirmed=True)

    def test_save_character_has_fixed_reference_and_exact_transcript(self):
        p=self.create();cid=p['candidates'][0]['id'];self.store.select(p['id'],cid)
        with patch('timbreforge.designs.CharacterStore') as cards:
            cards.return_value.create.return_value={'id':'c'*32}
            result=self.store.save_character(p['id'],cid,'角色',confirmed=True)
        self.assertEqual(result['id'],'c'*32)
        args=cards.return_value.create.call_args
        self.assertEqual(args.kwargs['backend'],'qwen3');self.assertEqual(args.kwargs['reference_text'],'参考台词')
        self.assertEqual(args.kwargs['instruction'],'');self.assertEqual(args.kwargs['description'],'温暖男声')
        self.assertEqual(args.kwargs['params']['qwen']['seed'],123)
        self.assertTrue(args.kwargs['provenance']['reference_transcript_user_confirmed'])
        self.assertEqual(args.args[1],self.store.audio_path(p['id'],cid))
        with self.assertRaises(ValueError):self.store.save_character(p['id'],cid,'角色')

    def test_explicit_cpu_and_invalid_audio(self):
        p=self.create(device='cpu');self.assertEqual(self.calls[0][3]['dtype'],'float32')
        def silent(*args,**kwargs):
            result=self.render(*args,**kwargs);sf.write(result['file'],np.zeros(20),24000,subtype='FLOAT');return result
        with patch('timbreforge.backend_qwen.design',side_effect=silent):
            with self.assertRaises(ValueError):self.store.append(p['id'],confirmed=True)
        self.assertEqual(len(self.store.load(p['id'])['candidates']),1)


if __name__=='__main__':unittest.main()
