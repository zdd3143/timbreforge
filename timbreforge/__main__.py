"""Canonical `python -m timbreforge` entry point: the voice Studio CLI.

The procedural synthesizer intentionally remains `timbreforge-synth` or
`python -m timbreforge.cli` so timbre experiments never masquerade as TTS.
"""
from timbreforge.studio_cli import main

if __name__ == '__main__':
    raise SystemExit(main())
