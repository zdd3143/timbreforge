"""Check an installed wheel from outside the source checkout (no models/network)."""
from pathlib import Path
import importlib.metadata
import sys

import timbreforge
from timbreforge import backend_cosyvoice, backend_qwen, role_planner, score, mosmodels
from timbreforge.studio import capabilities


def main():
    package = Path(timbreforge.__file__).resolve().parent
    version = importlib.metadata.version('timbreforge')
    assert version == timbreforge.__version__, (version, timbreforge.__version__)
    for worker in (backend_cosyvoice.WORKER, backend_qwen.WORKER, role_planner.WORKER,
                   score.CAMPP_SCRIPT, mosmodels.MOS_SCRIPT):
        path = Path(worker).resolve()
        assert path.is_file() and path.is_relative_to(package), path
    assert not any(x.get('backend') == 'qwen3_design' and x['available'] for x in capabilities())
    assert not any(name in sys.modules for name in ('torch', 'transformers', 'qwen_tts')), 'Model runtime imported into core'
    print('Installed package:', package)
    print('Version:', version)
    print('Packaged workers: OK; no model loaded')
    return 0


if __name__ == '__main__':
    sys.exit(main())
