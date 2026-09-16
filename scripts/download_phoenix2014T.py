import argparse
import os
import tarfile
import time
import urllib.error
import urllib.request

DEFAULT_URL = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"
DEFAULT_TGZ = "phoenix-2014-T.v3.tar.gz"
CHUNK = 1024 * 1024
MAX_ATTEMPTS = 5


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024


def download(url, dest):
    part = dest + ".part"
    existing = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", 200)
                if status == 200 and existing:
                    mode = "wb"
                    done = 0
                else:
                    mode = "ab"
                    done = existing
                total = done + int(resp.headers.get("Content-Length", 0))
                start = time.time()
                last = 0
                with open(part, mode) as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - last > 2:
                            last = now
                            rate = done / max(now - start, 1e-9)
                            eta = (total - done) / max(rate, 1e-9) if total > done else 0
                            print(f"\r{human(done)} / {human(total)}  {human(rate)}/s  ETA {int(eta)}s", end="", flush=True)
                print()
                if total and done != total:
                    raise IOError(f"Download incomplete: {human(done)} != {human(total)}")
                os.replace(part, dest)
                return dest
        except (urllib.error.URLError, OSError, IOError, TimeoutError) as e:
            print(f"\nAttempt {attempt}/{MAX_ATTEMPTS} failed: {e}", flush=True)
            time.sleep(5)

    raise RuntimeError("Download failed after retries; re-run to resume from the partial file.")


def extract(tgz_path, data_dir):
    print(f"Extracting {tgz_path} ...", flush=True)
    n = 0
    start = time.time()
    with tarfile.open(tgz_path, "r:gz") as tf:
        for member in tf:
            tf.extract(member, data_dir)
            n += 1
            if n % 500 == 0:
                print(f"\r{n} files extracted ...", end="", flush=True)
    print(f"\nExtracted {n} files in {time.time() - start:.1f}s", flush=True)


def locate_dataset(data_dir):
    for root, dirs, files in os.walk(data_dir):
        if "annotations" in dirs:
            manual = os.path.join(root, "annotations", "manual")
            if any(
                f.startswith("PHOENIX-2014-T") and f.endswith(".corpus.csv")
                for f in os.listdir(manual)
            ):
                return root
    return None


def verify(root):
    print(f"\nDataset root: {root}")
    anno = os.path.join(root, "annotations", "manual")
    frames = os.path.join(root, "features", "fullFrame-210x260px")
    ok = True
    for split in ("train", "dev", "test"):
        csv = os.path.join(anno, f"PHOENIX-2014-T.{split}.corpus.csv")
        split_dir = os.path.join(frames, split)
        n_dirs = len(os.listdir(split_dir)) if os.path.isdir(split_dir) else 0
        print(f"  {split:5s} csv={os.path.exists(csv)!s:5s}  video dirs={n_dirs}")
        ok = ok and os.path.exists(csv) and n_dirs > 0
    return ok


def main():
    parser = argparse.ArgumentParser(description="Download and extract the RWTH-PHOENIX-2014-T v3 dataset.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tgz", default=DEFAULT_TGZ)
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    dest = os.path.join(args.data_dir, args.tgz)
    print(f"Downloading\n  {args.url}\n  -> {dest}", flush=True)

    stat = os.statvfs(args.data_dir)
    free = stat.f_bavail * stat.f_frsize
    if free < 80 * 1024 ** 3:
        print(f"WARNING: only {human(free)} free on this filesystem. The archive is ~39 GB and needs "
              f"roughly 80 GB total after extraction.", flush=True)

    if os.path.exists(dest):
        print(f"{dest} already exists, skipping download.", flush=True)
    else:
        download(args.url, dest)
        print(f"Saved to {dest}", flush=True)

    if args.no_extract:
        print("Skipping extraction as requested.", flush=True)
        return

    if not tarfile.is_tarfile(dest):
        raise SystemExit("Downloaded file is not a valid tar.gz archive; please re-run to resume/re-download.")

    extract(dest, args.data_dir)
    root = locate_dataset(args.data_dir)
    if not root:
        raise SystemExit("Could not locate PHOENIX-2014-T annotations after extraction.")
    passed = verify(root)
    print("\nVerification " + ("PASSED" if passed else "FAILED"), flush=True)


if __name__ == "__main__":
    main()
