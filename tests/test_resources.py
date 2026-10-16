import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from timbreforge import resources as r


class ResourceTests(unittest.TestCase):
    def test_memory_errors_are_distinct_from_validation_errors(self):
        for message in ('CUDA out of memory', 'WinError 1455', 'MemoryError', '-1073741801'):
            self.assertTrue(r.is_memory_error(message))
        self.assertFalse(r.is_memory_error('Reference transcript missing'))

    def test_busy_comfy_never_interrupted_or_unloaded(self):
        with patch.object(r, '_comfy', return_value={'queue_running':[[1]],'queue_pending':[]}) as call:
            with self.assertRaisesRegex(RuntimeError,'未中断'):r.release_idle_models('http://127.0.0.1:8188')
        self.assertEqual(call.call_count,1)

    def test_idle_index_cache_and_executor_both_released(self):
        def api(base,path,data=None):
            if path=='/queue':return {'queue_running':[],'queue_pending':[]}
            if path.startswith('/object_info'):return {'IndexTTS25Unload':{}}
            if path=='/prompt':return {'prompt_id':'cleanup'}
            if path=='/history/cleanup':return {'cleanup':{'status':{'status_str':'success'}}}
            return {}
        # Node schema can be an empty mapping in mocks; a real schema is populated.
        def real_schema(*args):
            value=api(*args)
            return {'IndexTTS25Unload':{'input':{}}} if args[1].startswith('/object_info') else value
        with patch.object(r,'_comfy',side_effect=real_schema) as call:
            r.release_idle_models('http://127.0.0.1:8188')
        paths=[c.args[1] for c in call.call_args_list]
        self.assertIn('/prompt',paths)
        self.assertEqual(paths.count('/prompt'),2)
        self.assertLess(paths.index('/free'),paths.index('/prompt'))
        self.assertEqual(paths[-1],'/free')
        self.assertEqual(call.call_args.args[2],{'unload_models':True,'free_memory':True})

    def test_remote_comfy_not_modified(self):
        with patch.object(r,'_comfy') as call:
            r.release_idle_models('https://example.com')
        call.assert_not_called()

    def test_serialization_and_failed_cleanup_release_lock(self):
        with tempfile.TemporaryDirectory() as td:
            entered=threading.Event()
            def other():
                with r.model_slot(td,release_comfy=False):entered.set()
            with r.model_slot(td,release_comfy=False):
                thread=threading.Thread(target=other)
                thread.start()
                self.assertFalse(entered.wait(.1))
            thread.join(3)
            self.assertTrue(entered.is_set())
            with patch.object(r,'release_idle_models',side_effect=RuntimeError('cleanup failed')):
                with self.assertRaises(RuntimeError):
                    with r.model_slot(td):pass
            with r.model_slot(td,release_comfy=False,timeout=.2):pass

    def test_independent_process_cannot_load_model_while_slot_occupied(self):
        with tempfile.TemporaryDirectory() as td:
            script=('from timbreforge.resources import model_slot; import sys\n'
                    'with model_slot(sys.argv[1], release_comfy=False, timeout=.2): print("loaded")')
            with r.model_slot(td,release_comfy=False):
                process=subprocess.run([sys.executable,'-c',script,td],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(process.returncode,0)
            self.assertNotIn('loaded',process.stdout)
            self.assertIn('TimeoutError',process.stderr)


if __name__=='__main__':unittest.main()
