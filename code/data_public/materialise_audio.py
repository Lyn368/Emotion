import json, os, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / 'audios'
DEST.mkdir(parents=True, exist_ok=True)
mode = sys.argv[1] if len(sys.argv) > 1 else 'auto'   # auto|link|copy
plan = [json.loads(l) for l in open(HERE / 'clip_sources.jsonl') if l.strip()]
made = skipped = copied = 0
for c in plan:
    src, dst = Path(c['source']), DEST / (c['stem'] + '.wav')
    if dst.exists():
        skipped += 1
        continue
    if not src.exists():
        print('[!] missing source:', src)
        continue
    if mode != 'copy':
        try:
            os.link(src, dst)
            made += 1
            continue
        except OSError:
            if mode == 'link':
                raise
    shutil.copy2(src, dst)
    copied += 1
print(f'{made} hardlinked, {copied} copied, {skipped} already there '
      f'-> {DEST}')
