import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from timbreforge import role_planner as rp


def sample_plan():
    return dict(character_name='林澈', summary='沉着的救援队长',
        voice_prompt='成年女性，中低音，略带沙哑，吐字清晰。', preview_text='别担心，我会带你回家。',
        traits=[dict(dimension='音色', suggestion='略带沙哑', basis='设定：声音略带沙哑'),
                dict(dimension='音高', suggestion='中低音', basis='建议：体现沉稳'),
                dict(dimension='语速', suggestion='适中', basis='建议：体现克制')],
        uncertainties=['口音未指定，建议普通话。'])


def save_plan(root, plan_id='b'*32, created_at='2026-10-16T10:00:00+00:00'):
    record = dict(id=plan_id, profile='source', profile_sha256=hashlib.sha256(b'source').hexdigest(),
                  plan=sample_plan(), backend={'model_dir':'local-model'}, created_at=created_at)
    folder = root/'role-plans'/plan_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'plan.json').write_text(json.dumps(record), encoding='utf-8')
    return record, folder


class RoleImportTests(unittest.TestCase):
    def test_strict_docx_and_malformed_utf16(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'strict.docx'
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('word/document.xml', '<w:document xmlns:w="http://purl.oclc.org/ooxml/wordprocessingml/main">'
                    '<w:body><w:p><w:r><w:t>角色设定</w:t></w:r></w:p></w:body></w:document>')
            self.assertEqual(rp.import_profile(path),'角色设定')
            text=Path(td)/'broken.txt'
            for raw in (b'\xff\xfe\x41',b'\xff\xfe\x00\xd8'):
                text.write_bytes(raw)
                with self.assertRaisesRegex(ValueError,'无法识别文本编码'): rp.import_profile(text)
    def test_text_encodings_and_structured_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for encoding in ('utf-8-sig', 'utf-16', 'gb18030'):
                path = root / '角色.txt'
                path.write_bytes('林澈：沉着温柔。'.encode(encoding))
                self.assertEqual(rp.import_profile(path), '林澈：沉着温柔。')
            path = root / '角色.json'
            path.write_text('{"姓名":"林澈","性格":"沉着"}', encoding='utf-8')
            self.assertEqual(json.loads(rp.import_profile(path))['姓名'], '林澈')
            for invalid in ('42', '{'):
                path.write_text(invalid)
                with self.assertRaises(ValueError): rp.import_profile(path)

    def test_docx_paragraphs_tables_and_unsupported_doc(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / '角色.docx'
            xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                   '<w:body><w:p><w:r><w:t>林澈</w:t><w:br/><w:t>救援队长</w:t></w:r></w:p>'
                   '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>性格沉着</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
                   '</w:body></w:document>')
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('word/document.xml', xml)
            self.assertEqual(rp.import_profile(path), '林澈 救援队长\n性格沉着')
            old = path.with_suffix('.doc')
            old.write_bytes(b'doc')
            with self.assertRaisesRegex(ValueError, '另存为'): rp.import_profile(old)

    def test_corrupt_empty_and_oversized_imports(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / 'corrupt.docx'
            path.write_bytes(b'not zip')
            with self.assertRaisesRegex(ValueError, '无法读取 Word'): rp.import_profile(path)
            path = root / 'large.txt'
            path.write_bytes(b'a' * (rp.MAX_IMPORT_BYTES + 1))
            with self.assertRaisesRegex(ValueError, '4 MB'): rp.import_profile(path)
            for content in ('', 'a'*(rp.MAX_PROFILE_CHARS+1), 'name\x00data'):
                path.write_text(content)
                with self.assertRaises(ValueError): rp.import_profile(path)

    def test_zip_expansion_limit_and_xml_entities(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'bad.docx'
            for content in ('a'*(8*1024*1024+1), '<!DOCTYPE doc [<!ENTITY e "test">]><doc>&e;</doc>'):
                with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
                    z.writestr('word/document.xml', content)
                with self.assertRaises(ValueError): rp.import_profile(path)


class RolePlanTests(unittest.TestCase):
    def setUp(self):
        cleanup=patch('timbreforge.resources.release_idle_models');cleanup.start();self.addCleanup(cleanup.stop)

    def test_fenced_json_and_validation(self):
        plan = sample_plan()
        self.assertEqual(rp.parse_plan('```json\n'+json.dumps(plan)+'\n```'), plan)
        self.assertEqual(rp.parse_plan('```JSON\n'+json.dumps(plan)+'\n```'), plan)
        for field, value in (('voice_prompt',''), ('traits',[]), ('traits','none'), ('uncertainties','none'),
                             ('preview_text','a'*501), ('summary',False)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                rp.validate_plan(dict(plan, **{field:value}))
        with self.assertRaises(ValueError): rp.parse_plan('I think this is a nice voice.')
        with self.assertRaisesRegex(ValueError,'模型返回'): rp.parse_plan('['*5000+']'*5000)
        # An unnamed source must be labelled, never presented as a quote from the setting.
        bad = sample_plan()
        bad['traits'][0]['basis'] = '无依据'
        cleaned = rp.validate_plan(bad)
        self.assertEqual(cleaned['traits'][0]['basis'], '未标注：无依据')
        self.assertTrue(any('未标注' in note for note in cleaned['_notes']))
        # Half-width colons are accepted and normalised.
        half = sample_plan()
        half['traits'][0]['basis'] = '设定:声音偏低沉'
        self.assertEqual(rp.validate_plan(half)['traits'][0]['basis'], '设定：声音偏低沉')
        # Stylistic deviations are reported, not turned into a failed run.
        few = sample_plan()
        few['traits'] = few['traits'][:2]
        few['uncertainties'] = ['a', 'b', 'c', 'd', 'e']
        few['preview_text'] = '短句'
        noted = rp.validate_plan(few)
        self.assertEqual(len(noted['traits']), 2)
        self.assertEqual(len(noted['uncertainties']), 5)
        self.assertEqual(len(noted['_notes']), 3)
        # Reloading a stored plan must not double-label the basis.
        once = rp.validate_plan(bad)
        twice = rp.validate_plan(once)
        self.assertEqual(twice['traits'], once['traits'])
        self.assertEqual(twice['traits'][0]['basis'], '未标注：无依据')

    def test_real_subprocess_contract_and_persistence_with_mock_worker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / 'model'
            model.mkdir()
            (model/'config.json').write_text('{}')
            python = root / 'python.exe'
            python.touch()
            cfg = SimpleNamespace(planner_python=python, planner_model=model, outputs_dir=root/'outputs')
            def worker(command, **kwargs):
                request = json.loads(Path(command[command.index('--request')+1]).read_text(encoding='utf-8'))
                self.assertEqual(request['profile'], '角色设定')
                self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
                result = dict(plan=sample_plan(), model_dir=str(model), profile_sha256=request['profile_sha256'])
                Path(command[command.index('--result')+1]).write_text(json.dumps(result))
                return SimpleNamespace(returncode=0)
            with patch.object(rp, 'get_config', return_value=cfg), patch.object(rp.subprocess,'run',side_effect=worker):
                result = rp.infer_profile(' 角色设定 ')
            record = root/'outputs'/'role-plans'/result['id']/'plan.json'
            self.assertEqual(json.loads(record.read_text(encoding='utf-8'))['plan'], sample_plan())

    def test_changed_profile_and_missing_plan_never_generate(self):
        record = dict(id='a'*32, profile='original')
        with tempfile.TemporaryDirectory() as td, \
             patch.object(rp, 'get_config', return_value=SimpleNamespace(outputs_dir=Path(td))), \
             patch('timbreforge.simple_ui.generate_simple') as generate:
            for source in ('changed','original'):
                with self.assertRaises(ValueError): rp.generate_from_plan(record,source,'prompt','line')
            generate.assert_not_called()

    def test_edited_prompt_and_preview_reach_generator_and_are_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            record, folder = save_plan(root)
            with patch.object(rp,'get_config',return_value=SimpleNamespace(outputs_dir=root)), \
                 patch('timbreforge.simple_ui.generate_simple',return_value=('audio.wav',{'session_id':'s'})) as gen:
                rp.generate_from_plan(record,'source','edited voice','edited line')
            gen.assert_called_once_with('edited voice',None,preview_text='edited line')
            stored=json.loads(next(folder.glob('generation-*.json')).read_text(encoding='utf-8'))
            self.assertEqual(stored['edited_voice_prompt'],'edited voice')
            self.assertEqual(stored['edited_preview_text'],'edited line')

    def test_history_sorted_and_corrupt_records_ignored(self):
        with tempfile.TemporaryDirectory() as td, patch.object(rp, 'get_config', return_value=SimpleNamespace(outputs_dir=Path(td))):
            root = Path(td)
            save_plan(root)
            newer, folder = save_plan(root, 'c'*32, '2026-10-17T10:00:00+00:00')
            _, broken = save_plan(root, 'd'*32)
            (broken/'plan.json').write_text('{broken')
            self.assertEqual([r['id'] for r in rp.list_plans()], [newer['id'], 'b'*32])
            restored = rp.restore_plan(newer['id'])
            self.assertEqual(restored['prompt'], sample_plan()['voice_prompt'])
            self.assertIsNone(restored['audio'])
            for invalid in ('../escape', None, 123):
                with self.assertRaises(ValueError): rp.load_plan(invalid)
            changed = dict(newer, profile='tampered')
            (folder/'plan.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, '不一致'): rp.load_plan(newer['id'])

    def test_saved_source_is_authoritative_before_generation(self):
        with tempfile.TemporaryDirectory() as td, patch.object(rp, 'get_config', return_value=SimpleNamespace(outputs_dir=Path(td))), \
                patch('timbreforge.simple_ui.generate_simple') as generate:
            record, _ = save_plan(Path(td))
            with self.assertRaisesRegex(ValueError, '已修改'):
                rp.generate_from_plan(dict(record, profile='changed'), 'changed', 'voice', 'line')
            generate.assert_not_called()

    def test_restore_real_audio_edited_text_and_missing_audio(self):
        import numpy as np
        import soundfile as sf
        from timbreforge.designs import DesignStore
        from timbreforge.backend_qwen import QwenParams
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = SimpleNamespace(outputs_dir=root)
            record, folder = save_plan(root)
            source = root/'source.wav'
            sf.write(source, np.full(2400, .2), 24000)
            with patch.object(rp, 'get_config', return_value=cfg), \
                 patch('timbreforge.designs.get_config', return_value=cfg), \
                 patch('timbreforge.backend_qwen.design', return_value={'file':str(source)}):
                store = DesignStore()
                session = store.create('edited voice', 'edited line', QwenParams(), confirmed=True)
                cid = session['candidates'][0]['id']
                link = dict(plan_id=record['id'], edited_voice_prompt='edited voice', edited_preview_text='edited line',
                            generation=dict(session_id=session['id'], candidate_id=cid))
                rp._save_record(folder/'generation-test.json', link)
                restored = rp.restore_plan(record['id'])
                self.assertEqual(restored['prompt'], 'edited voice')
                self.assertEqual(restored['preview_text'], 'edited line')
                audio = Path(restored['audio'])
                self.assertTrue(audio.is_file())
                self.assertEqual(restored['warning'], '')
                # Mismatched text must not be paired with otherwise valid audio.
                rp._save_record(folder/'generation-test.json', dict(link, edited_preview_text='different'))
                mismatch = rp.restore_plan(record['id'])
                self.assertIsNone(mismatch['audio'])
                self.assertIn('不一致', mismatch['warning'])
                rp._save_record(folder/'generation-test.json', link)
                audio.unlink()
                missing = rp.restore_plan(record['id'])
                self.assertIsNone(missing['audio'])
                self.assertEqual(missing['prompt'], 'edited voice')
                self.assertIn('未恢复', missing['warning'])

    def test_failed_atomic_save_preserves_existing_record(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'plan.json'
            path.write_text('original')
            with patch.object(rp.os, 'replace', side_effect=OSError('disk error')):
                with self.assertRaises(OSError): rp._save_record(path, {'new':True})
            self.assertEqual(path.read_text(), 'original')
            self.assertEqual(list(Path(td).iterdir()), [path])

    def test_generation_retention_keeps_three_newest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, folder = save_plan(root)
            for index in range(5):
                path = folder / f'generation-{index:032x}.json'
                path.write_text('{}', encoding='utf-8')
                os.utime(path, (1000 + index, 1000 + index))
            self.assertEqual(len(rp._generation_paths(folder)), 5)
            removed = rp._prune_generations(folder)
            self.assertEqual(len(removed), 2)
            remaining = sorted(path.name for path in folder.glob('generation-*.json'))
            self.assertEqual(remaining, [f'generation-{index:032x}.json' for index in (2, 3, 4)])
            # The plan record itself is never touched by retention.
            self.assertTrue((folder / 'plan.json').is_file())

    def test_generation_lookup_rejects_unsafe_ids_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            record, folder = save_plan(root)
            with patch.object(rp, 'get_config', return_value=SimpleNamespace(outputs_dir=root)):
                for bad in ('../../etc/passwd', 'generation-../../x', 123, '', 'plan'):
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        rp._generation_by_id(folder, bad)
                restored = rp.restore_plan(record['id'], 'generation-' + 'a' * 32)
                self.assertIsNone(restored['audio'])
                self.assertEqual(restored['generations'], [])
                self.assertIn('找不到', restored['warning'])

    def test_generate_prunes_and_reports_kept_takes(self):
        with tempfile.TemporaryDirectory() as td, \
                patch.object(rp, 'get_config', return_value=SimpleNamespace(outputs_dir=Path(td))), \
                patch('timbreforge.simple_ui.generate_simple', return_value=('a.wav', {'session_id': 's'})):
            root = Path(td)
            record, folder = save_plan(root)
            for index in range(4):
                path = folder / f'generation-{index:032x}.json'
                path.write_text('{}', encoding='utf-8')
                os.utime(path, (1000 + index, 1000 + index))
            audio, details = rp.generate_from_plan(record, 'source', 'voice', 'line')
            self.assertEqual(audio, 'a.wav')
            self.assertEqual(len(details['pruned']), 2)
            self.assertEqual(details['kept'], 0)  # the fake takes carry no usable audio
            self.assertTrue(details['generation_id'].startswith('generation-'))
            self.assertTrue((folder / f"{details['generation_id']}.json").is_file())
            self.assertEqual(len(list(folder.glob('generation-*.json'))), 3)

    def test_save_voice_confirmation_and_library_registration(self):
        import numpy as np
        import soundfile as sf
        from timbreforge.characters import CharacterStore
        from timbreforge.voice_library import VoiceLibrary
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = SimpleNamespace(outputs_dir=root, characters_dir=root / 'characters')
            record, _ = save_plan(root)
            wav = root / 'take.wav'
            sf.write(wav, np.full(2400, .2), 24000, subtype='FLOAT')
            card = CharacterStore(cfg.characters_dir).create('林澈', wav, reference_text='edited line',
                                                             backend='qwen3')
            take = dict(id='generation-' + 'a' * 32, created_at='2026-10-16T10:00:00+00:00',
                        prompt='edited voice', preview_text='edited line', session_id='s' * 32,
                        candidate_id='c' * 32, audio=str(wav), generation={})
            with patch.object(rp, 'get_config', return_value=cfg), \
                 patch('timbreforge.voice_library.get_config', return_value=cfg), \
                 patch('timbreforge.characters.get_config', return_value=cfg), \
                 patch.object(rp, '_generation_by_id', return_value=take), \
                 patch('timbreforge.designs.DesignStore') as design:
                design.return_value.save_character.return_value = card
                with self.assertRaisesRegex(ValueError, '确认'):
                    rp.save_voice(record['id'], take['id'], '林澈')
                self.assertFalse(design.return_value.select.called)
                result = rp.save_voice(record['id'], take['id'], '林澈', description='救援队长',
                                       confirmed=True)
            design.return_value.select.assert_called_once_with(take['session_id'], take['candidate_id'])
            self.assertTrue(result['created_character'])
            library = VoiceLibrary(cfg.characters_dir)
            group = library.get(result['character']['id'])
            self.assertEqual(group['name'], '林澈')
            self.assertEqual(group['description'], '救援队长')
            self.assertEqual(len(group['voices']), 1)
            self.assertEqual(group['voices'][0]['character_id'], card['id'])
            self.assertEqual(group['voices'][0]['origin']['generation_id'], take['id'])
            self.assertEqual(rp.library_audio(group['id'], card['id'], cfg.characters_dir),
                             card['reference_audio'])
            # A second save can join the same character instead of creating another one.
            second = CharacterStore(cfg.characters_dir).create('林澈B', wav, backend='qwen3')
            with patch.object(rp, 'get_config', return_value=cfg), \
                 patch('timbreforge.voice_library.get_config', return_value=cfg), \
                 patch('timbreforge.characters.get_config', return_value=cfg), \
                 patch.object(rp, '_generation_by_id', return_value=take), \
                 patch('timbreforge.designs.DesignStore') as design:
                design.return_value.save_character.return_value = second
                joined = rp.save_voice(record['id'], take['id'], '忽略', target_character_id=group['id'],
                                       confirmed=True)
            self.assertFalse(joined['created_character'])
            self.assertEqual(len(joined['character']['voices']), 2)
            view = rp.library_characters(cfg.characters_dir)
            self.assertTrue(view['ok'])
            self.assertEqual(len(view['characters']), 1)
            self.assertEqual(len(view['characters'][0]['voices']), 2)

    def test_save_voice_requires_audible_take(self):
        with tempfile.TemporaryDirectory() as td, \
                patch.object(rp, 'get_config', return_value=SimpleNamespace(outputs_dir=Path(td))):
            root = Path(td)
            record, _ = save_plan(root)
            with self.assertRaises(ValueError):
                rp.save_voice(record['id'], 'generation-' + 'b' * 32, '林澈', confirmed=True)
            with self.assertRaisesRegex(ValueError, '1–120'):
                rp.save_voice(record['id'], 'generation-' + 'b' * 32, '   ', confirmed=True)

    @unittest.skipUnless(importlib.util.find_spec('gradio'), 'UI extra not installed')
    def test_ui_build_and_failed_analysis_clear_old_plan(self):
        import gradio as gr
        from timbreforge import role_ui
        with patch.object(role_ui, 'infer_profile', side_effect=AssertionError('lazy UI')), \
             patch.object(role_ui, 'list_plans', side_effect=AssertionError('lazy history')), \
             patch.object(role_ui, 'library_characters', side_effect=AssertionError('lazy library')):
            with gr.Blocks(analytics_enabled=False) as demo:
                with gr.Tabs():
                    role_ui.build_role_tab(gr)
        config = demo.get_config_file()
        self.assertNotIn('json', [item['type'] for item in config['components']])
        ids = {item['props'].get('elem_id') for item in config['components']}
        for expected in ('role_records', 'role_reroll', 'role_save_open', 'role_save_name',
                         'role_save_description', 'role_save_confirm', 'role_save_do',
                         'role_library_open', 'role_library_characters', 'role_library_voices',
                         'role_library_audio'):
            self.assertIn(expected, ids)
        analyze = next(f.fn for f in demo.fns.values() if f.api_name == 'role_analyze')
        with patch.object(role_ui, 'infer_profile', side_effect=RuntimeError('test failure')):
            outputs = list(analyze('text', None))
        self.assertIsNone(outputs[-1][0])
        self.assertEqual(outputs[-1][4]['value'], '')
        self.assertFalse(outputs[-1][6]['interactive'])
        self.assertFalse(outputs[-1][10]['interactive'])  # save stays off without a take
        self.assertIn('test failure', outputs[-1][-1])
        self.assertTrue(all(not item['interactive'] for item in outputs[0][12:-1]))
        self.assertTrue(all(item['interactive'] for item in outputs[-1][12:-1]))

        render = next(f.fn for f in demo.fns.values() if f.api_name == 'role_generate')
        with patch.object(role_ui, 'generate_from_plan', side_effect=RuntimeError('render failure')):
            updates = list(render({}, 'source', 'prompt', 'line', None))
        self.assertIsNone(updates[-1][0])
        self.assertIn('render failure', updates[-1][1])
        self.assertFalse(updates[-1][5]['interactive'])   # nothing on screen to save
        self.assertTrue(updates[-1][6]['interactive'])    # re-rolling stays possible
        self.assertTrue(all(item['interactive'] for item in updates[-1][7:]))
        self.assertTrue(all(not item['interactive'] for item in updates[0][5:]))

        reroll = next(f.fn for f in demo.fns.values() if f.api_name == 'role_reroll')
        take = 'generation-' + 'a' * 32
        with patch.object(role_ui, 'generate_from_plan',
                          return_value=('take.wav', {'generation_id': take, 'pruned': [take], 'kept': 2})), \
             patch.object(role_ui, 'record_choices', return_value=([('记录一 · 10:00', take)], take)):
            updates = list(reroll({}, 'source', 'prompt', 'line', None))
        self.assertEqual(updates[-1][3], take)
        self.assertTrue(updates[-1][5]['interactive'])
        self.assertIn('已删除最早的 1 条旧记录', updates[-1][1])

        select = next(f.fn for f in demo.fns.values() if f.api_name == 'role_select_take')
        with patch.object(role_ui, 'restore_plan', return_value=dict(audio='old.wav', prompt='p',
                                                                    preview_text='old line',
                                                                    generation_id=take, warning='')):
            chosen = select({}, take)
        self.assertEqual(chosen[0], 'old.wav')
        self.assertEqual(chosen[1], take)
        self.assertTrue(chosen[4]['interactive'])
        self.assertIn('没有被这条记录覆盖', chosen[3])

        save_open = next(f.fn for f in demo.fns.values() if f.api_name == 'role_save_open')
        self.assertIn('先生成一条试听', save_open({}, None)[-1])
        with patch.object(role_ui, 'library_characters', return_value=dict(ok=True, error='', characters=[])):
            panel = save_open({'plan': {'character_name': '林澈'}}, take)
        self.assertTrue(panel[0]['visible'])
        self.assertEqual(panel[1]['value'], '林澈')
        self.assertEqual(panel[2]['choices'][0], ('新建角色', ''))

        library = next(f.fn for f in demo.fns.values() if f.api_name == 'role_library')
        with patch.object(role_ui, 'library_characters', side_effect=ValueError('broken index')):
            opened = library()
        self.assertTrue(opened[0]['visible'])
        self.assertIn('无法读取', opened[3])

        restore = next(f.fn for f in demo.fns.values() if f.api_name == 'role_restore')
        with patch.object(role_ui, 'restore_plan', side_effect=ValueError('missing plan')):
            result = restore('missing')
        self.assertTrue(all(item == gr.skip() for item in result[:-1]))
        self.assertIn('当前编辑内容已保留', result[-1])
        demo.close()


if __name__ == '__main__': unittest.main()

