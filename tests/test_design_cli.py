import contextlib
import io
import unittest
from unittest.mock import patch
from timbreforge.design_cli import main

class DesignCLITests(unittest.TestCase):
    def test_canonical_route(self):
        from timbreforge.studio_cli import main as entry
        with patch('timbreforge.design_cli.main', return_value=0) as target:
            self.assertEqual(entry(['--config','local.toml','design','list']),0)
        target.assert_called_once_with(['--config','local.toml','list'])

    def test_create_params_are_real_and_greedy_disables_both(self):
        with patch('timbreforge.designs.DesignStore') as store, contextlib.redirect_stdout(io.StringIO()):
            store.return_value.create.return_value={'id':'mock'}
            self.assertEqual(main(['create','--prompt','温暖低沉','--text','你好。','--seed','7','--greedy','--confirm-design']),0)
        call=store.return_value.create.call_args
        self.assertEqual(call.args,('温暖低沉','你好。'))
        params=call.kwargs['params']
        self.assertFalse(params.do_sample)
        self.assertFalse(params.subtalker_dosample)
        self.assertEqual(params.seed,7)
        self.assertTrue(call.kwargs['confirmed'])

if __name__=='__main__':unittest.main()
