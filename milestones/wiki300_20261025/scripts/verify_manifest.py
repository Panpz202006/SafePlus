#!/usr/bin/env python3
"""Verify every listed file's bytes in a delivered milestone."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'manifest_sha256.json').read_text())
errors=[]
for name,digest in m['files'].items():
    p=ROOT/name
    if not p.is_file():errors.append({'file':name,'error':'missing'})
    elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:errors.append({'file':name,'error':'sha256 mismatch'})
print(json.dumps({'status':'PASS' if not errors else 'FAIL','checked_files':len(m['files']),'errors':errors},ensure_ascii=False,indent=2))
raise SystemExit(1 if errors else 0)
