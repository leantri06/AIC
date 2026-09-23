import importlib.util
import json
import csv
from fractions import Fraction
from pathlib import Path

import pytest

av = pytest.importorskip('av')
np = pytest.importorskip('numpy')
spec = importlib.util.spec_from_file_location(
    'cpu_frames', Path(__file__).parents[1] / 'scripts/extract_keyframes_cpu.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_vfr_mapping_resume_and_changed_config(tmp_path):
    source = tmp_path / 'test.mov'
    with av.open(str(source), 'w') as out:
        stream = out.add_stream('mpeg4', rate=10)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = 'yuv420p'
        stream.codec_context.time_base = Fraction(1, 10)
        for i, pts in enumerate([0, 1, 2, 5, 10, 15, 20, 30]):
            data = np.full((48, 64, 3), 0 if i < 4 else 240, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(data, format='rgb24')
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    settings = dict(sample_fps=2., max_gap=1., pixel_threshold=12.,
                    max_width=1280, quality=90, version=1)
    root = tmp_path / 'output'
    module.extract(source, root, settings, threads=1, reserve_gib=0)
    with (root / 'map-keyframes/test.csv').open() as fh:
        rows = list(csv.DictReader(fh))
    with av.open(str(source)) as video:
        times = [float(frame.time) for frame in video.decode(video=0)]
    for row in rows:
        assert float(row['pts_time']) == pytest.approx(times[int(row['frame_idx'])] - times[0])
    assert int(rows[-1]['frame_idx']) == 7
    report = root / 'keyframe-reports/test.json'
    before = report.stat().st_mtime_ns
    module.extract(source, root, settings, threads=1, reserve_gib=0)
    assert report.stat().st_mtime_ns == before
    # Simulate interruption before the completion marker was saved.
    report.unlink()
    module.extract(source, root, settings, threads=1, reserve_gib=0)
    assert json.loads(report.read_text())['decoded_frames'] == 8
    with pytest.raises(RuntimeError, match='settings changed'):
        module.extract(source, root, {**settings, 'sample_fps': 1.}, 1, 0)
