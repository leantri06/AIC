"""CPU keyframe candidates with original frame indices, PTS and per-video resume.

Requires av, numpy and Pillow. See docs/CPU_KEYFRAMES.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

VERSION = 1
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'data/batches/batch2/extracted'


def atomic_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def extract(video, root, settings, threads, reserve_gib):
    import av
    import numpy as np
    from PIL import Image

    started = time.monotonic()
    identity = {'path': str(video.resolve()), 'bytes': video.stat().st_size,
                'mtime_ns': video.stat().st_mtime_ns}
    images = root / 'Keyframes' / video.stem
    maps = root / 'map-keyframes'
    reports = root / 'keyframe-reports'
    for directory in (images.parent, maps, reports):
        directory.mkdir(parents=True, exist_ok=True)
    done = reports / f'{video.stem}.json'
    csv_path = maps / f'{video.stem}.csv'
    if done.exists():
        report = json.loads(done.read_text(encoding='utf-8'))
        if report['source'] != identity or report['settings'] != settings:
            raise RuntimeError('Source/settings changed: use a new --output-root.')
        with csv_path.open(newline='', encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        actual = {p.name for p in images.glob('*.jpg')}
        expected = {f"{int(row['frame_idx']):09d}.jpg" for row in rows}
        if len(rows) != report['kept_frames'] or actual != expected:
            raise RuntimeError('Completed output is inconsistent; use a new output root.')
        if any((images / name).stat().st_size == 0 for name in expected):
            raise RuntimeError('Empty image in completed output.')
        print(f'SKIP {video.stem}: {len(rows)} images', flush=True)
        return

    owner = images / '.owner.json'
    ownership = {'source': identity, 'settings': settings}
    if images.exists():
        if not owner.exists() or json.loads(owner.read_text()) != ownership:
            raise RuntimeError('Existing output not owned by this run; use a new --output-root.')
        # There is no completion report, so these are generated remnants from
        # an interrupted/failed attempt. Delete only files covered by our owner marker.
        for generated in images.glob('*.jpg'):
            generated.unlink()
        for generated in images.glob('*.jpg.tmp'):
            generated.unlink()
    else:
        images.mkdir()
        atomic_json(owner, ownership)

    rows = []
    previous_small = None
    previous_kept_time = -math.inf
    previous_time = -math.inf
    previous_raw_time = -math.inf
    timeline_corrections = 0
    origin = None
    next_sample = 0.0
    decoded = 0
    last_progress = 0.0
    last_frame = None

    def keep(frame, index, elapsed, rate, pts_seconds):
        if shutil.disk_usage(root).free < reserve_gib * 1024**3:
            raise RuntimeError(f'Less than {reserve_gib} GiB free; stopped safely.')
        width = min(settings['max_width'], frame.width)
        height = max(1, round(frame.height * width / frame.width))
        image = frame.reformat(width=width, height=height, format='rgb24').to_image()
        target = images / f'{index:09d}.jpg'
        temporary = target.with_suffix('.jpg.tmp')
        image.save(temporary, format='JPEG', quality=settings['quality'])
        temporary.replace(target)
        rows.append({'n': len(rows) + 1, 'pts_time': f'{elapsed:.9f}',
                     'fps': rate, 'frame_idx': index, 'source_pts_time': pts_seconds})

    with av.open(str(video)) as container:
        if not container.streams.video:
            raise RuntimeError('No video stream.')
        stream = container.streams.video[0]
        stream.codec_context.thread_count = threads
        stream.thread_type = 'AUTO'
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        declared_frames = stream.frames
        for index, frame in enumerate(container.decode(stream)):
            if frame.time is None:
                raise RuntimeError('Missing PTS; refusing to invent timestamps.')
            pts_seconds = float(frame.time)
            if origin is None:
                origin = pts_seconds
            raw_elapsed = pts_seconds - origin
            if raw_elapsed < previous_raw_time:
                timeline_corrections += 1
            previous_raw_time = raw_elapsed
            # Some official MP4/MOV files contain backward PTS or edit-list
            # discontinuities. Keep the original PTS in source_pts_time while
            # deriving a monotonic relative timeline for sampling and UI.
            minimum_step = (1.0 / fps) if fps > 0 else 0.0
            elapsed = (raw_elapsed if raw_elapsed >= previous_time
                       else previous_time + minimum_step)
            previous_time = elapsed
            decoded += 1
            last_frame = (frame, index, elapsed, pts_seconds)
            if time.monotonic() - last_progress >= 15:
                print(f'{video.stem}: {elapsed / 60:.1f} min video, '
                      f'{decoded} decoded, {len(rows)} saved', flush=True)
                last_progress = time.monotonic()
            if elapsed + 1e-8 < next_sample:
                continue
            next_sample = (math.floor(elapsed * settings['sample_fps'] + 1e-8) + 1) / settings['sample_fps']
            small = frame.reformat(width=160, height=90, format='rgb24').to_ndarray()
            difference = math.inf if previous_small is None else float(
                np.abs(small.astype(np.int16) - previous_small.astype(np.int16)).mean())
            if (difference >= settings['pixel_threshold'] or
                    elapsed - previous_kept_time >= settings['max_gap']):
                keep(frame, index, elapsed, fps, pts_seconds)
                previous_small = small
                previous_kept_time = elapsed

        if last_frame is None:
            raise RuntimeError('Video decoded zero frames.')
        if declared_frames and decoded != declared_frames:
            print(f'WARN {video.stem}: decoded {decoded}, container declares '
                  f'{declared_frames}; decoder EOS is authoritative', flush=True)
        if not rows or rows[-1]['frame_idx'] != last_frame[1]:
            frame, index, elapsed, pts_seconds = last_frame
            keep(frame, index, elapsed, fps, pts_seconds)

    # A rerun with identical source/settings deterministically rewrites partial images.
    # Reject unexpected leftovers rather than delete user data.
    expected = {f"{row['frame_idx']:09d}.jpg" for row in rows}
    if {p.name for p in images.glob('*.jpg')} != expected:
        raise RuntimeError('Unexpected JPG files; use a new --output-root.')
    for name in expected:
        with Image.open(images / name) as im:
            im.verify()
    temporary_csv = csv_path.with_suffix('.csv.tmp')
    with temporary_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['n', 'pts_time', 'fps', 'frame_idx', 'source_pts_time'])
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)
    report = {'source': identity, 'settings': settings, 'decoded_frames': decoded,
              'kept_frames': len(rows), 'video_seconds': previous_time,
              'elapsed_seconds': round(time.monotonic() - started, 3),
              'image_bytes': sum((images / name).stat().st_size for name in expected),
              'timestamp_origin': origin, 'av_version': av.__version__}
    report['declared_frames'] = declared_frames
    report['declared_frame_delta'] = decoded - declared_frames if declared_frames else None
    report['timeline_corrections'] = timeline_corrections
    atomic_json(done, report)
    print(f'DONE {video.stem}: {len(rows)} images in {report["elapsed_seconds"]:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos-dir', type=Path, default=DEFAULT_ROOT / 'Videos')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--video-id', action='append', help='Exact filename stem; repeat to select multiple videos.')
    parser.add_argument('--sample-fps', type=float, default=2.0)
    parser.add_argument('--max-gap', type=float, default=4.0)
    parser.add_argument('--pixel-threshold', type=float, default=12.0)
    parser.add_argument('--max-width', type=int, default=1280)
    parser.add_argument('--quality', type=int, default=90)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reserve-gib', type=float, default=30.0)
    parser.add_argument('--list', action='store_true', help='List selected inputs without decoding.')
    args = parser.parse_args()
    if (not all(math.isfinite(v) and v > 0 for v in
                (args.sample_fps, args.max_gap, args.reserve_gib)) or
            not math.isfinite(args.pixel_threshold) or args.pixel_threshold < 0 or
            args.max_width < 16 or args.threads < 1 or not 1 <= args.quality <= 100):
        parser.error('Invalid extraction settings.')
    videos = sorted(p for p in args.videos_dir.rglob('*')
                    if p.is_file() and p.suffix.lower() in ('.mp4', '.mov'))
    if len({p.stem.casefold() for p in videos}) != len(videos):
        parser.error('Duplicate video IDs.')
    if args.video_id:
        missing = set(args.video_id) - {p.stem for p in videos}
        if missing:
            parser.error(f'Unknown video IDs: {sorted(missing)}')
        videos = [p for p in videos if p.stem in args.video_id]
    if not videos:
        parser.error('No videos found.')
    if args.list:
        print('\n'.join(p.name for p in videos))
        print(f'{len(videos)} videos')
        return 0
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = {key: getattr(args, key) for key in
                ('sample_fps', 'max_gap', 'pixel_threshold', 'max_width', 'quality')}
    settings['version'] = VERSION
    # OS lock releases automatically after interruption. Never run simultaneous writers.
    with (root / '.keyframe-cpu.lock').open('a+b') as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except ImportError:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = []
        for number, video in enumerate(videos, 1):
            print(f'[{number}/{len(videos)}] {video.name}', flush=True)
            try:
                extract(video, root, settings, args.threads, args.reserve_gib)
            except KeyboardInterrupt:
                print('Stopped. Run the same command to resume. Current video restarts.')
                return 130
            except Exception as exc:
                failures.append({'video': video.name, 'error': str(exc)})
                print(f'ERROR: {video.name}: {exc}', flush=True)
                if shutil.disk_usage(root).free < args.reserve_gib * 1024**3:
                    break
        atomic_json(root / 'keyframe-last-run.json', {'selected': len(videos), 'failures': failures})
        return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
