#!/usr/bin/env python
"""Refuses to commit an image that carries GPS coordinates in its EXIF.

Every photo taken for this repository comes off a phone, and a phone writes where it was standing.
On 2026-09-14 a photograph of a power-supply label reached `git add` with the author's home to five
decimal places inside it, and was caught because someone happened to look. That is not a safety
check; this is. Scanning the history the same day found four more (`media/koch/*`, inherited from
upstream, a Paris office rather than a home, and already deleted from the tree) -- so the failure
mode is normal rather than exotic, and it does not announce itself.

Stripping is deliberately not automatic. A hook that silently rewrites what you committed teaches
you nothing about the file, and the right fix differs: usually re-export without EXIF, occasionally
keep the shot and publish nothing at all. So this reports and stops.

To strip one, keeping the pixels and dropping every tag:

    python -c "from PIL import Image; im=Image.open('IN.jpg').convert('RGB'); \
               out=Image.new('RGB', im.size); out.paste(im); out.save('OUT.jpg', quality=90)"

Only GPS fails the check. A date, a camera model and an exposure are provenance, and this project
wants them.
"""

import sys

GPS_IFD = 0x8825


def gps_in(path: str) -> bool:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as im:
            exif = im.getexif()
    except (UnidentifiedImageError, OSError):
        # Not a readable image (a Git LFS pointer, a test fixture, a truncated file). Nothing to
        # assert about it either way, and refusing to commit it is not this hook's business.
        return False
    return bool(exif and exif.get_ifd(GPS_IFD))


HINT = """
To strip one, keeping the pixels and dropping every tag:

    python -c "from PIL import Image; im = Image.open('IN.jpg').convert('RGB'); \
               out = Image.new('RGB', im.size); out.paste(im); out.save('OUT.jpg', quality=90)"

Only GPS fails this check. A date, a camera and an exposure are provenance; keep those.
"""


def main(paths: list[str]) -> int:
    hits = [p for p in paths if gps_in(p)]
    for p in hits:
        print(f"{p}: carries GPS coordinates in EXIF -- strip them before committing", file=sys.stderr)
    if hits:
        print(HINT, file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
