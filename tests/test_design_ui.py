"""Lazy VoiceDesign controls; no model, network, GPU or private audio needed."""
import importlib.util
import unittest
from unittest.mock import patch
from timbreforge import design_ui as ui
from timbreforge.backend_qwen import QwenParams


class DesignUITests(unittest.TestCase):
    def test_prompt_assembly_is_transparent_text(self):
        self.assertEqual(ui.assemble_prompt('温暖','女声','成年','','低音','','平静'),
                         '温暖；性别表达：女声；年龄感：成年；音高感：低音；表演：平静')

    def test_generate_requires_explicit_consent_and_known_params(self):
        values=QwenParams().to_dict()
        with patch.object(ui,'design_store') as store:
            for consent in (False,None,1):
                with self.assertRaises(ValueError):ui.generate_design('prompt','text',values,confirmed=consent)
            store.assert_not_called()
            store.return_value.create.return_value={'id':'session'}
            result=ui.generate_design(' prompt ',' text ',values,count=2.,confirmed=True)
            self.assertEqual(result,{'id':'session'})
            args=store.return_value.create.call_args
            self.assertEqual(args.args[:2],('prompt','text'));self.assertEqual(args.kwargs['count'],2)
            self.assertIsInstance(args.args[2],QwenParams)
        with self.assertRaises(ValueError):ui._params(dict(values,guidance_scale=7))
        with self.assertRaises(ValueError):ui._params(dict(values,seed=True))

    def test_append_selection_and_save_selected_only(self):
        values=QwenParams().to_dict()
        with patch.object(ui,'design_store') as store:
            ui.append_design('s',values,count=1,confirmed=True)
            self.assertEqual(store.return_value.append.call_args.args,('s',))
            ui.select_design('s','c',True,'note')
            store.return_value.select.assert_called_once_with('s','c',favorite=True,note='note')
            store.return_value.load.return_value={'selected_candidate_id':None}
            with self.assertRaises(ValueError):ui.save_design_character('s','c','name',confirmed=True)
            store.return_value.save_character.assert_not_called()
            store.return_value.load.return_value={'selected_candidate_id':'c'}
            ui.save_design_character('s','c',' name ',confirmed=True)
            store.return_value.save_character.assert_called_once_with('s','c','name',confirmed=True)
        with self.assertRaises(ValueError):ui.save_design_character('s','c','name')

    def test_ab_uses_validated_audio_paths(self):
        with patch.object(ui,'design_store') as store,patch('timbreforge.candidates.compare_audio',return_value={'a':'a','b':'b'}) as compare:
            store.return_value.audio_path.side_effect=['verified/a.wav','verified/b.wav']
            self.assertEqual(ui.compare_designs('s','a','b',True),{'a':'a','b':'b'})
        compare.assert_called_once_with('verified/a.wav','verified/b.wav',match_volume=True)
        with self.assertRaises(ValueError):ui.compare_designs('s','a','b','yes')

    @unittest.skipUnless(importlib.util.find_spec('gradio'),'UI extra not installed')
    def test_build_lazy_and_no_fictional_controls(self):
        import gradio as gr
        with patch.object(ui,'design_store',side_effect=AssertionError('no IO during UI build')):
            with gr.Blocks(analytics_enabled=False) as demo:
                with gr.Tabs():ui.build_design_tab(gr,{'concurrency_id':'timbreforge-synthesis','concurrency_limit':1})
        config=demo.get_config_file()
        props=[c.get('props',{}) for c in config['components']]
        ids={p.get('elem_id') for p in props}
        self.assertTrue({'design_prompt','design_seed','design_generate','design_append','design_a','design_b',
                         'design_save','design_temperature','design_subtalker_dosample'}<=ids)
        self.assertIn('design_save_confirm', ids)
        for elem_id in ('design_generate', 'design_append'):
            self.assertFalse(next(p for p in props if p.get('elem_id') == elem_id)['interactive'])
        self.assertFalse({'design_cfg','design_steps','design_negative_prompt'}&ids)
        apis={d.get('api_name') for d in config['dependencies']}
        self.assertTrue({'design_generate','design_append','design_select','design_save'}<=apis)
        loader=next(f.fn for f in demo.fns.values() if f.api_name=='design_load')
        original=QwenParams(seed=42).to_dict()
        chosen=QwenParams(seed=99, temperature=.7).to_dict()
        session=dict(id='session', prompt='voice', text='line', params=original, device='cuda:0',
                     selected_candidate_id='second', candidates=[
                         dict(id='first', seed=42, result={'params':original}),
                         dict(id='second', seed=99, result={'params':chosen, 'device':'cpu'})])
        with patch.object(ui, 'load_session', return_value=session):
            restored=loader('session')
        self.assertEqual(restored[5], 99)
        self.assertEqual(restored[-3], 'cpu')
        self.assertEqual(restored[-2]['value'], 'second')
        self.assertEqual(restored[-1]['value'], 'first')
        demo.close()


if __name__=='__main__':unittest.main()
