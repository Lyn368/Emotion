# -----------------------
# Import all the libraries
# -----------------------
import argparse
import subprocess
from pathlib import Path

# -----------------------
# Conversion function (for one, repeat ths function for all the videos)
# -----------------------
def convert_one(src: Path, dst: Path, sample_rate: int, overwrite: bool) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Start the comand
    cmd = ["ffmpeg", "-y" if overwrite else "-n"]
    cmd += ["-i", str(src)]
    """
    cmd += ["-i", str(src), "-vn", "-b:a", bitrate, "-ar", str(sample_rate)]
    # keep original channels
    cmd += ["-codec:a", "libmp3lame", str(dst)]
    """
    # Extract audio, mono, resample, WAV PCM 16-bit
    cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True, str(dst)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")[-1200:]
        return False, f"{src.name} failed:\n{err}"

# -----------------------
# Main
# -----------------------
def main():
    p = argparse.ArgumentParser(description="Convert .webm videos to .mp3 audio")
    p.add_argument("--root", default="/Users/ytinglin/Downloads/ProjectCE903/25-26_CE903-SP_team01/MLmodel/data", help="Project root folder")
    p.add_argument("--bitrate", default="192k", help="MP3 bitrate (default: 192k)")
    p.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate (default: 44100)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing mp3 files")
    p.add_argument("--recursive", action="store_true", help="Search .webm files recursively")
    args = p.parse_args()

    root = Path(args.root)
    video_dir = root / "videos"
    audio_dir = root / "audios"

    if not video_dir.exists():
        raise SystemExit(f"Video folder not found: {video_dir}")

    pattern = "**/*.webm" if args.recursive else "*.webm"
    webm_files = sorted(video_dir.glob(pattern))
    if not webm_files:
        print(f"No .webm files found in: {video_dir} (recursive={args.recursive})")
        return

    ok = 0
    fail = 0

    for src in webm_files:
        # Keep relative subfolder structure from video/ into audio/
        rel = src.relative_to(video_dir).with_suffix(".wav")
        dst = audio_dir / rel

        success, msg = convert_one(src, dst, args.sample_rate, args.overwrite)
        if success:
            ok += 1
            print(f"✅ {src.relative_to(root)} -> {dst.relative_to(root)}")
        else:
            fail += 1
            print(f"❌ {msg}\n")

    print("\nDone.")
    print(f"Output folder: {audio_dir}")
    print(f"Success: {ok}, Failed: {fail}")

if __name__ == "__main__":
    main()


