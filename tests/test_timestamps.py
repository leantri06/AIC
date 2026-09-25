from pathlib import Path

from aic2026.data_platform.timestamps import load_timestamp_map


def test_duplicate_frames_keep_image_timestamps(tmp_path: Path):
    csv = tmp_path / "video.csv"
    csv.write_text("n,frame_idx,pts_time\n1,0,0\n2,0,0.0333333\n3,25,1\n", encoding="utf-8-sig")
    table = load_timestamp_map(csv)
    assert table.timestamp(0, r"D:\old\video\001.jpg") == 0
    assert table.timestamp(0, "/data/video/002.jpg") == 0.0333333
    assert table.timestamp(25, "003.jpg") == 1
    # A mismatching ordinal must not override the requested frame.
    assert table.timestamp(25, "001.jpg") == 1
    assert table.timestamp(99, "001.jpg") is None


def test_cache_separates_directories_and_refreshes_changed_file(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for folder, pts in ((a, 1), (b, 2)):
        (folder / "video.csv").write_text(f"n,frame_idx,pts_time\n1,25,{pts}\n")
    assert load_timestamp_map(a / "video.csv").timestamp(25, "001.jpg") == 1
    assert load_timestamp_map(b / "video.csv").timestamp(25, "001.jpg") == 2
    (a / "video.csv").write_text("n,frame_idx,pts_time\n1,25,3.125\n")
    assert load_timestamp_map(a / "video.csv").timestamp(25, "001.jpg") == 3.125
