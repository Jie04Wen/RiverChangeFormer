# -*- coding: utf-8 -*-

import os
import urllib.request

from .paths import MIT_B2_URL, MIT_B2_PATH


def _report(block_num, block_size, total):
    if total <= 0:
        return
    done = block_num * block_size
    pct = min(100.0, 100.0 * done / total)
    if block_num % 256 == 0 or done >= total:
        print("\r[weights] downloading MiT-B2 ... %.1f%%" % pct, end="", flush=True)


def ensure_mit_b2(path=None, url=MIT_B2_URL):
    path = str(path or MIT_B2_PATH)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("[weights] not found, downloading from\n  %s" % url, flush=True)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=_report)
    print()
    if not os.path.exists(tmp) or os.path.getsize(tmp) < 1_000_000:
        raise RuntimeError("download failed: %s" % url)
    os.replace(tmp, path)
    print("[weights] saved to", path, "(%.1f MB)" % (os.path.getsize(path) / 1e6))
    return path


if __name__ == "__main__":
    ensure_mit_b2()
