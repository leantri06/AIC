"""Benchmark a single video: CPU decode, CUDA TransNet V2, then JPEG export.

Uses official PyTorch architecture (85cef72) and a pinned community-converted
state dict. Cache is outside Git; weights load with weights_only=True.
This one-video benchmark buffers reduced 48x27 RGB frames in RAM.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=ROOT / 'data/batches/batch2/extracted/Videos/M05_V001.mp4')
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/.cache/transnetv2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('This GPU benchmark requires CUDA.')
    torch.set_num_threads(4)
    # FP32 is appropriate for Pascal; no mixed precision or Tensor Core assumptions.
    start = time.perf_counter()
    spec = importlib.util.spec_from_file_location('transnet_model', args.cache / 'transnetv2_pytorch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TransNetV2()
    weights_path = args.cache / 'transnetv2-pytorch-weights.pth'
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=True)
    model.eval().cuda()
    with torch.inference_mode():
        for _ in range(3):
            model(torch.zeros((1, 100, 27, 48, 3), dtype=torch.uint8, device='cuda'))
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - start
    print(f'Model ready on {torch.cuda.get_device_name(0)}, setup {setup_seconds:.2f}s', flush=True)

    start = time.perf_counter()
    small_frames, times = [], []
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 4
        fps = float(stream.average_rate)
        declared = stream.frames
        for frame in container.decode(stream):
            if frame.time is None:
                raise RuntimeError('Missing PTS.')
            times.append(float(frame.time))
            small_frames.append(frame.reformat(width=48, height=27, format='rgb24').to_ndarray())
            if len(times) % 5000 == 0:
                print(f'Decode {len(times)} frames', flush=True)
    if not times or (declared and len(times) != declared):
        raise RuntimeError('Incomplete decode.')
    frames = np.stack(small_frames)
    del small_frames
    timestamps = np.asarray(times) - times[0]
    if np.any(np.diff(timestamps) < 0):
        raise RuntimeError('Non-monotonic PTS.')
    decode_seconds = time.perf_counter() - start
    print(f'Decoded {len(frames)} frames in {decode_seconds:.2f}s', flush=True)

    # Official inference: 100-frame windows, stride 50, only central 50 outputs.
    padding_end = 25 + (50 - len(frames) % 50) % 50
    padded = np.concatenate((np.repeat(frames[:1], 25, axis=0), frames,
                             np.repeat(frames[-1:], padding_end, axis=0)))
    predictions = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(padded) - 99, 50):
            tensor = torch.from_numpy(padded[offset:offset+100][None]).cuda()
            single, _ = model(tensor)
            predictions.append(torch.sigmoid(single)[0, 25:75, 0].cpu().numpy())
            if len(predictions) % 50 == 0:
                print(f'Inference {min(len(predictions)*50, len(frames))}/{len(frames)}', flush=True)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    probs = np.concatenate(predictions)[:len(frames)]
    assert len(probs) == len(frames) and np.isfinite(probs).all()
    np.save(args.output / 'transition_probabilities.npy', probs)
    # Stable intervals exclude transition-positive frames; indices are inclusive.
    stable = probs <= 0.5
    edges = np.diff(np.r_[False, stable, False].astype(np.int8))
    scenes = list(zip(np.flatnonzero(edges == 1).tolist(),
                      (np.flatnonzero(edges == -1) - 1).tolist()))
    if not scenes:
        raise RuntimeError('No stable shots: inspect predictions before using output.')
    (args.output / 'shots.json').write_text(json.dumps(scenes), encoding='utf-8')
    selected = set()
    for first, last in scenes:
        begin, end = timestamps[first], timestamps[last]
        count = max(1, int(np.ceil((end - begin) / 3.0)))
        for t in begin + (np.arange(count) + 0.5) * ((end - begin) / count):
            selected.add(min(last, max(first, int(np.searchsorted(timestamps, t)))))

    image_dir = args.output / 'Keyframes' / args.video.stem
    image_dir.mkdir(parents=True)
    rows = []
    start = time.perf_counter()
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 4
        for index, frame in enumerate(container.decode(stream)):
            if index not in selected:
                continue
            if abs(float(frame.time) - times[index]) > 1e-6:
                raise RuntimeError('PTS mismatch between decoding passes.')
            width = min(1280, frame.width)
            height = max(1, round(frame.height * width / frame.width))
            im = frame.reformat(width=width, height=height, format='rgb24').to_image()
            im.save(image_dir / f'{index:09d}.jpg', quality=90)
            rows.append([len(rows)+1, timestamps[index], fps, index])
    if len(rows) != len(selected):
        raise RuntimeError('Missing selected keyframes.')
    export_seconds = time.perf_counter() - start
    map_dir = args.output / 'map-keyframes'
    map_dir.mkdir()
    with (map_dir / f'{args.video.stem}.csv').open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['n', 'pts_time', 'fps', 'frame_idx'])
        writer.writerows(rows)
    report = dict(video=str(args.video), gpu=torch.cuda.get_device_name(0),
                  torch=torch.__version__, av=av.__version__, frames=len(frames),
                  video_seconds=float(timestamps[-1]), fps=fps, shots=len(scenes),
                  keyframes=len(rows), setup_seconds=setup_seconds,
                  decode_seconds=decode_seconds, inference_seconds=inference_seconds,
                  export_seconds=export_seconds,
                  processing_seconds=decode_seconds+inference_seconds+export_seconds,
                  peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  image_bytes=sum(p.stat().st_size for p in image_dir.glob('*.jpg')),
                  precision='FP32', window=100, stride=50, threshold=0.5,
                  selection='midpoint of <=3-second bins in stable shots',
                  architecture_commit='85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed',
                  weights_source='magnusdtd/TransNetV2@c90da43a211d390a5fe2603f78f82419b7e6cd9c',
                  weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest())
    (args.output / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
