from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'SHA256_MANIFEST.json').read_text())
for name,sha in manifest.items():
    f=root/name
    assert f.is_file(),f'MISSING {name}'
    assert hashlib.sha256(f.read_bytes()).hexdigest()==sha,f'SHA FAIL {name}'
print('PASS',len(manifest),'SHA-256 entries verified')
