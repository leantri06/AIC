import json

import pytest

from aic2026.ingestion.cpu_keyframes import cpu_video_records


def fixture_root(tmp_path):
    for name in ("Videos", "Keyframes/N001-V001", "map-keyframes", "keyframe-reports"):
        (tmp_path / name).mkdir(parents=True)
    video = tmp_path / "Videos/N001-V001.mov"
    video.write_bytes(b"video")
    identity = {"path": str(video), "bytes": 5, "mtime_ns": video.stat().st_mtime_ns}
    report = {"source": identity, "settings": {}, "kept_frames": 2, "decoded_frames": 101}
    (tmp_path / "keyframe-reports/N001-V001.json").write_text(json.dumps(report))
    (tmp_path / "Keyframes/N001-V001/.owner.json").write_text(json.dumps({"source": identity, "settings": {}}))
    for frame in (0, 100):
        (tmp_path / f"Keyframes/N001-V001/{frame:09d}.jpg").write_bytes(b"image")
    (tmp_path / "map-keyframes/N001-V001.csv").write_text(
        "n,frame_idx,pts_time,source_pts_time\n1,0,0,0.2\n2,100,4.001,4.201\n")
    return tmp_path


def test_cpu_names_are_frame_indices_and_vfr_pts_are_preserved(tmp_path):
    records = cpu_video_records(fixture_root(tmp_path), "N001-V001")
    assert [r.frame_id for r in records] == [0, 100]
    assert records[1].pts_time == 4.001
    assert records[1].source_pts_time == 4.201
    assert records[1].video_path.endswith(".mov")


def test_incomplete_cpu_output_rejected(tmp_path):
    root = fixture_root(tmp_path)
    (root / "Keyframes/N001-V001/000000100.jpg").unlink()
    with pytest.raises(ValueError, match="missing/empty"):
        cpu_video_records(root, "N001-V001")


def test_duplicate_frame_rejected(tmp_path):
    root = fixture_root(tmp_path)
    (root / "map-keyframes/N001-V001.csv").write_text(
        "n,frame_idx,pts_time,source_pts_time\n1,0,0,0.2\n2,0,4.001,4.201\n")
    with pytest.raises(ValueError, match="invalid mapping"):
        cpu_video_records(root, "N001-V001")


def test_source_change_rejected(tmp_path):
    root = fixture_root(tmp_path)
    (root / "Videos/N001-V001.mov").write_bytes(b"different video")
    with pytest.raises(ValueError, match="source changed"):
        cpu_video_records(root, "N001-V001")


@pytest.mark.parametrize("pts", ["nan", "-1", "inf"])
def test_invalid_timestamp_rejected(tmp_path, pts):
    root = fixture_root(tmp_path)
    (root / "map-keyframes/N001-V001.csv").write_text(
        f"n,frame_idx,pts_time,source_pts_time\n1,0,0,0.2\n2,100,{pts},4.201\n")
    with pytest.raises(ValueError, match="invalid mapping"):
        cpu_video_records(root, "N001-V001")
