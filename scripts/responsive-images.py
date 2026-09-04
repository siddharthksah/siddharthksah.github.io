#!/usr/bin/env python3
"""Generate 480/960px variants for post images and add srcset via kramdown IALs.

Idempotent: skips variants that exist, leaves lines that already carry srcset.
Run from the repo root after adding images to a post:  python3 scripts/responsive-images.py
"""
import glob
import os
import re
import subprocess

WIDTHS = [480, 960]
SIZES = '(max-width: 800px) 92vw, 770px'


def image_width(path):
    out = subprocess.run(["magick", "identify", "-format", "%w", path],
                         capture_output=True, text=True)
    return int(out.stdout.strip()) if out.returncode == 0 else 0


def ensure_variants(path):
    """Create -480/-960 files next to the original; return available srcset widths."""
    nat = image_width(path)
    if not nat:
        return None
    base, ext = os.path.splitext(path)
    entries = []
    for w in WIDTHS:
        if w >= nat:
            continue
        vpath = f"{base}-{w}{ext}"
        if not os.path.exists(vpath):
            q = ["-quality", "82"] if ext.lower() in (".jpg", ".jpeg") else []
            subprocess.run(["magick", path, "-resize", f"{w}x"] + q + [vpath], check=True)
        entries.append((vpath, w))
    entries.append((path, nat))
    return entries


def srcset_attr(entries):
    return ", ".join(f"{p.replace(os.getcwd() + '/', '/').lstrip('.')} {w}w" for p, w in entries)


def process_post(mdpath):
    src = open(mdpath).read()
    out_lines = []
    changed = False
    for line in src.split("\n"):
        m = re.match(r'^!\[([^\]]*)\]\((/images/[^)]+\.(?:jpg|jpeg|png))\)(\{:[^}]*\})?\s*$', line)
        if not m or "srcset" in (m.group(3) or ""):
            out_lines.append(line)
            continue
        alt, imgurl, ial = m.group(1), m.group(2), m.group(3) or ""
        fspath = imgurl.lstrip("/")
        if not os.path.exists(fspath):
            out_lines.append(line)
            continue
        entries = ensure_variants(fspath)
        if not entries or len(entries) < 2:
            out_lines.append(line)
            continue
        ss = ", ".join(f"/{p} {w}w" for p, w in entries)
        extra = ial[2:-1].strip() if ial else ""
        attrs = f'srcset="{ss}" sizes="{SIZES}"' + (f" {extra}" if extra else "")
        out_lines.append(f'![{alt}]({imgurl}){{: {attrs}}}')
        changed = True
    if changed:
        open(mdpath, "w").write("\n".join(out_lines))
    return changed


if __name__ == "__main__":
    for md in sorted(glob.glob("_posts/*.md")):
        if process_post(md):
            print("updated", md)
