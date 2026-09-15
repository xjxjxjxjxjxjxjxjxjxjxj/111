#!/usr/bin/env python3
import os
import io
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
print('Repo root:', root)

for p in root.rglob('*.py'):
    if 'site-packages' in str(p) or '.venv' in str(p) or 'venv' in str(p):
        continue
    # skip large binary-like or generated folders
    try:
        text = p.read_text(encoding='utf-8')
    except Exception:
        continue
    if '\t' in text:
        bak = p.with_suffix(p.suffix + '.bak')
        if not bak.exists():
            bak.write_text(text, encoding='utf-8')
        new_text = text.replace('\t', '    ')
        p.write_text(new_text, encoding='utf-8')
        print('Fixed:', p)
print('Done')
