"""UI callback tests with the synthesis and character services mocked."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from timbreforge import webui


class StudioWebUITests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        studio = types.ModuleType("timbreforge.studio")

        def generation_request(**kwargs):
            values = dict(task="clone", backend="indextts", text="", reference_audio=None,
                          reference_text="", instruction="", character_id=None, params=None,
                          seed_override=None, save_character_name="", description="")
            values.update(kwargs)
            request = types.SimpleNamespace(**values)
            self.calls.append(request)
            return request

        def generate(request, **kwargs):
            self.calls.append(("generate", request, kwargs))
            if self.generate_error:
                raise self.generate_error
            return {
                "file": "output.wav", "metadata_file": "output.json",
                "task": request.task, "backend": request.backend,
                "elapsed": 1.25, "reference_audio": request.reference_audio,
                "reference_text": request.reference_text,
                "instruction": request.instruction, "warnings": ["test warning"],
                "character_id": "created-card",
            }

        self.generate_error = None
        studio.GenerationRequest = generation_request
        studio.generate = generate
        params_module = types.ModuleType("timbreforge.backend_indextts")
        params_module.SynthParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
        self.modules_patch = patch.dict(sys.modules, {
            "timbreforge.studio": studio,
            "timbreforge.backend_indextts": params_module,
        })
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        self.fake_studio = studio

    def test_consent_is_required_before_service_import_or_call(self):
        audio, status = webui.do_studio_synth(
            "clone", "indextts", "original.wav", "words", "", "hello", 4,
            "", "", False)
        self.assertIsNone(audio)
        self.assertIn("授权", status)
        self.assertEqual(self.calls, [])

    def test_clone_passes_original_reference_and_request_fields(self):
        audio, status = webui.do_studio_synth(
            "clone", "cosyvoice3", "C:/uploads/full_original.wav", "exact transcript",
            "", "new line", 42, "Reusable", "Description", True)
        request = self.calls[0]
        self.assertEqual(request.task, "clone")
        self.assertEqual(request.backend, "cosyvoice3")
        self.assertEqual(request.reference_audio, "C:/uploads/full_original.wav")
        self.assertEqual(request.reference_text, "exact transcript")
        self.assertEqual(request.instruction, "")
        self.assertEqual(request.text, "new line")
        self.assertEqual(request.params.seed, 42)
        self.assertEqual(request.save_character_name, "Reusable")
        self.assertEqual(request.description, "Description")
        self.assertEqual(audio, "output.wav")
        self.assertIn("output.json", status)
        self.assertIn("test warning", status)
        self.assertIn("不保证", status)

    def test_instruct_task_passes_instruction(self):
        webui.do_studio_synth("instruct", "indextts", "ref.wav", "transcript",
                              "speak gently", "line", 3, "", "", True)
        request = self.calls[0]
        self.assertEqual(request.task, "instruct")
        self.assertEqual(request.instruction, "speak gently")
        self.assertEqual(request.reference_text, "transcript")

    def test_qwen_clone_receives_own_parameters(self):
        webui.do_studio_synth('clone','qwen3','ref.wav','exact transcript','','line',123,'','',True)
        request=self.calls[0]
        self.assertIsNone(request.params)
        self.assertEqual(request.qwen_params['seed'],123)
        self.assertEqual(request.qwen_params['language'],'Chinese')

    def test_memory_error_does_not_blame_reference_transcript(self):
        audio,status=webui._studio_failure(RuntimeError('CUDA out of memory'))
        self.assertIsNone(audio)
        self.assertIn('自动释放',status)
        self.assertNotIn('检查参考音频',status)

    def test_invalid_seed_and_service_failure_are_actionable(self):
        audio, status = webui.do_studio_synth(
            "clone", "indextts", "ref.wav", "", "", "line", 1.5, "", "", True)
        self.assertIsNone(audio)
        self.assertIn("种子", status)
        self.generate_error = FileNotFoundError("ComfyUI is not reachable")
        audio, status = webui.do_studio_synth(
            "clone", "indextts", "ref.wav", "", "", "line", 1, "", "", True)
        self.assertIsNone(audio)
        self.assertIn("ComfyUI is not reachable", status)
        self.assertIn("检查参考音频", status)

    def test_character_generation_preserves_saved_params_and_optional_seed(self):
        audio, status = webui.do_character_synth("abc123", "new line", "", None, True)
        request = self.calls[0]
        self.assertEqual(request.task, "character")
        self.assertEqual(request.character_id, "abc123")
        self.assertEqual(request.text, "new line")
        self.assertEqual(request.instruction, "")
        self.assertIsNone(request.params)
        self.assertIsNone(request.seed_override)
        self.assertEqual(audio, "output.wav")
        self.assertIn("created-card", status)

    def test_character_explicit_cpu_device_and_invalid_device(self):
        webui.do_character_synth('abc123', 'line', '', None, True, 'cpu')
        self.assertEqual(self.calls[0].device, 'cpu')
        audio, status = webui.do_character_synth('abc123', 'line', '', None, True, 'auto')
        self.assertIsNone(audio)
        self.assertIn('设备', status)
        self.assertEqual(len(self.calls), 2)  # one request plus one mocked generate call

    def test_character_seed_override_is_passed_without_default_params(self):
        webui.do_character_synth("abc123", "line", "override", 21, True)
        request = self.calls[0]
        self.assertEqual(request.seed_override, 21)
        self.assertIsNone(request.params)
        self.assertEqual(request.instruction, "override")

    def test_character_store_list_and_preview_use_ids_and_metadata(self):
        characters = types.ModuleType("timbreforge.characters")
        card = {
            "id": "same-id", "name": "Repeated name", "reference_audio": "saved.wav",
            "reference_text": "saved words", "backend": "indextts",
            "instruction": "saved style", "description": "card note", "params": {"seed": 9},
        }

        class Store:
            def list(self):
                return [card]

            def load(self, character_id):
                self.last_id = character_id
                return card

        characters.CharacterStore = Store
        with patch.dict(sys.modules, {"timbreforge.characters": characters}):
            self.assertEqual(webui.studio_character_choices(), [("Repeated name", "same-id")])
            audio, metadata = webui.do_character_preview("same-id")
        self.assertEqual(audio, "saved.wav")
        self.assertIn("saved words", metadata)
        self.assertIn("saved style", metadata)
        self.assertIn('"seed": 9', metadata)

    @unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio not installed")
    def test_gradio_build_smoke_without_model_or_network(self):
        import gradio  # Load framework before forbidding network operations.
        with patch.object(webui, "get_bank", side_effect=AssertionError("no dataset")), \
             patch("requests.sessions.Session.request", side_effect=AssertionError("no network")), \
             patch("httpx.Client.send", side_effect=AssertionError("no network")):
            demo = webui.build_ui()
        self.assertTrue(demo.enable_queue)
        config = demo.get_config_file()
        self.assertIn("VoiceDesign", str(config))
        self.assertNotIn("数据集检索（原有功能）", str(config))
        self.assertNotIn("1336", str(config))
        tabs = [c["props"]["label"] for c in config["components"] if c["type"] == "tabitem"]
        self.assertEqual(tabs[0], "角色设定 → 声线推演")
        self.assertEqual(tabs[1], "简化版 · 一句话生成")
        self.assertEqual(tabs[2], "文字声线设计 / Qwen VoiceDesign")
        self.assertIn("studio_generate", str(config))
        self.assertIn("角色复用", str(config))
        self.assertIn("能力状态", str(config))

    @unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio not installed")
    def test_optional_legacy_build_is_lazy_and_errors_are_clear(self):
        with patch.object(webui, "get_bank", side_effect=FileNotFoundError("missing index")):
            demo = webui.build_ui(include_legacy=True)
            self.assertIn("数据集检索", str(demo.get_config_file()))
            rows, message = webui.do_search("voice", 1, 3, -1, "auto")
            self.assertEqual(rows, [])
            self.assertIn("数据集检索不可用", message)
            self.assertIn("missing index", message)
            self.assertIsNone(webui.do_preview("cid")[0])
            audio, message = webui.do_synth("voice", "cid", 1, "hello", .8, .8,
                                          30, 1, 3, 1, {}, "", 3, -1, "auto")
            self.assertIsNone(audio)
            self.assertIn("数据集检索不可用", message)

    def test_legacy_synth_uses_studio_validation_and_output_service(self):
        entry = types.SimpleNamespace(cid="cid", name="Voice")
        bank = types.SimpleNamespace(get=lambda cid: entry, to_wav=lambda e: "ref.wav")
        with patch.object(webui, "get_bank", return_value=bank):
            audio, message = webui.do_synth("", "cid", 7, "line", .8, .8, 30,
                                          1, 3, 1, {}, "http://custom", 3, -1, "auto")
        request = self.calls[0]
        self.assertEqual(request.reference_audio, "ref.wav")
        self.assertEqual(request.base_url, "http://custom")
        self.assertEqual(request.params.seed, 7)
        self.assertEqual(request.text, "line")
        self.assertEqual(audio, "output.wav")
        self.assertIn("cid", message)


if __name__ == "__main__":
    unittest.main()
