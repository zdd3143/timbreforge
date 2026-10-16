import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from timbreforge.characters import CharacterStore
from timbreforge.studio import generate,GenerationRequest


class QwenStudioTests(unittest.TestCase):
    def test_saved_design_identity_uses_base_not_redesign(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); ref=root/'ref.wav'; raw=root/'raw.wav'
            sf.write(ref,np.full(2400,.1),24000,subtype='FLOAT')
            sf.write(raw,np.full(4800,.2),24000,subtype='FLOAT')
            store=CharacterStore(root/'cards')
            card=store.create('设计角色',ref,reference_text='原始试听台词。',backend='qwen3',params={'qwen':{'seed':17,'temperature':.8}})
            with patch('timbreforge.backend_qwen.clone',return_value={'file':str(raw),'elapsed':.2}) as clone, patch('timbreforge.backend_qwen.design',side_effect=AssertionError('never redesign')):
                result=generate(GenerationRequest(task='character',character_id=card['id'],text='新的台词。',seed_override=19),character_store=store,output_dir=root/'renders')
            self.assertEqual(result['backend'],'qwen3')
            self.assertEqual(result['character_id'],card['id'])
            self.assertEqual(result['params']['qwen']['seed'],19)
            self.assertEqual(clone.call_args.kwargs['dtype'],'bfloat16')
            self.assertEqual(clone.call_args.args[1],'原始试听台词。')
            self.assertEqual(clone.call_args.args[2],'新的台词。')
            self.assertTrue(Path(result['file']).is_relative_to(root/'renders'))
            self.assertEqual(result['reference_info']['sha256'],hashlib.sha256(ref.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError,'指令'):
                generate(GenerationRequest(task='character',character_id=card['id'],text='test',instruction='迟疑'),character_store=store)

    def test_reference_mutation_never_publishes_stale_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();ref=root/'ref.wav';raw=root/'raw.wav'
            sf.write(ref,np.full(2400,.1),24000,subtype='FLOAT')
            sf.write(raw,np.full(4800,.2),24000,subtype='FLOAT')
            def mutate(*args,**kwargs):
                sf.write(ref,np.full(2400,.3),24000,subtype='FLOAT')
                return {'file':str(raw),'elapsed':.1}
            with patch('timbreforge.backend_qwen.clone',side_effect=mutate), self.assertRaisesRegex(RuntimeError,'Reference changed'):
                generate(GenerationRequest(task='clone',backend='qwen3',text='new',reference_audio=str(ref),reference_text='ref'),output_dir=root/'out')
            self.assertFalse((root/'out').exists())

    def test_qwen_instruct_never_silently_ignored(self):
        with self.assertRaisesRegex(ValueError,'instruct'):
            generate(GenerationRequest(task='instruct',backend='qwen3',text='test',reference_audio='unused.wav',reference_text='ref'))


if __name__=='__main__':unittest.main()
