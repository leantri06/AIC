import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "kaggle"))
from siglip2_runner import (WEIGHTS_SHA, balanced, fingerprint, save_result,
                           shard_records, validate_result)
from aic2026.models import FrameRecord

spec = importlib.util.spec_from_file_location("prepare_kaggle_batch2", REPO / "scripts/prepare_kaggle_batch2.py")
packer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packer)


def fixture_shard(tmp_path):
    image = tmp_path / "000000100.jpg"
    image.write_bytes(b"unchanged-jpeg-bytes")
    record = FrameRecord(vector_id=0, video_id="N001-V001", frame_id=100,
        keyframe_path=str(image), video_path=str(tmp_path / "N001-V001.mov"),
        pts_time=4.001, source_pts_time=4.201)
    path = tmp_path / "N001-V001-0000.aicshard"
    return path, packer.write_shard(path, [record])


def fixture_result(tmp_path):
    import zipfile
    path, job = fixture_shard(tmp_path)
    with zipfile.ZipFile(path) as z:
        payload = z.read("records.jsonl").decode()
    config = {"checkpoint_sha256": WEIGHTS_SHA, "precision": "fp16"}
    meta = dict(schema=1, input_sha256=job["sha256"], records_sha256=job["records_sha256"],
        config=config, config_id=fingerprint(config), job_id=job["id"], count=job["count"])
    values = np.zeros((1, 1152), dtype=np.float32)
    values[0, 0] = 1
    result = tmp_path / f"{job['id']}.npz"
    save_result(result, values, payload, meta)
    return result, job, meta, values, payload


def test_shard_keeps_exact_images_and_vfr_mapping(tmp_path):
    import zipfile
    path, job = fixture_shard(tmp_path)
    rows = shard_records(path, job)
    assert rows[0]["frame_id"] == 100
    assert rows[0]["pts_time"] == 4.001
    assert rows[0]["source_pts_time"] == 4.201
    assert rows[0]["video_path"] == "Videos/N001-V001.mov"
    assert rows[0]["keyframe_path"] == "Keyframes/N001-V001/000000100.jpg"
    with zipfile.ZipFile(path) as z:
        assert z.read(rows[0]["member"]) == b"unchanged-jpeg-bytes"


def test_corrupt_shard_rejected(tmp_path):
    path, job = fixture_shard(tmp_path)
    with path.open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt input"):
        shard_records(path, job)


def test_balances_frames_without_duplicate_jobs():
    jobs = [dict(id=str(i), count=n) for i, n in enumerate([100, 90, 20, 10])]
    groups = balanced(jobs, 2)
    assert [sum(j["count"] for j in group) for group in groups] == [110, 110]
    assert sorted(j["id"] for group in groups for j in group) == ["0", "1", "2", "3"]


def test_resume_valid_result(tmp_path):
    path, job, meta, _, _ = fixture_result(tmp_path)
    assert validate_result(path, job, meta["config_id"]) == meta
    assert not path.with_suffix(".npz.tmp").exists()


def test_resume_different_precision_rejected(tmp_path):
    path, job, _, _, _ = fixture_result(tmp_path)
    with pytest.raises(ValueError, match="config mismatch"):
        validate_result(path, job, "different-config")


@pytest.mark.parametrize("problem", ["nan", "wrong-norm", "wrong-mapping", "wrong-input"])
def test_bad_result_rejected(tmp_path, problem):
    path, job, meta, values, payload = fixture_result(tmp_path)
    if problem == "nan":
        values[0, 0] = np.nan
    elif problem == "wrong-norm":
        values[0, 0] = 2
    elif problem == "wrong-mapping":
        payload = payload.replace('"frame_id": 100', '"frame_id": 99')
    else:
        meta["input_sha256"] = "changed-source"
    save_result(path, values, payload, meta)
    with pytest.raises(ValueError):
        validate_result(path, job)


def test_merge_rebuilds_paths_and_preserves_vector_order(tmp_path):
    import subprocess
    path, job, _, _, _ = fixture_result(tmp_path)
    extracted = tmp_path / "extracted"
    (extracted / "Keyframes/N001-V001").mkdir(parents=True)
    (extracted / "Videos").mkdir()
    (extracted / "Keyframes/N001-V001/000000100.jpg").write_bytes(b"image")
    (extracted / "Videos/N001-V001.mov").write_bytes(b"video")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(dict(schema=1, sample=False, frames=1, shards=[job])))
    output = tmp_path / "merged"
    subprocess.run([sys.executable, str(REPO / "scripts/merge_kaggle_batch2.py"),
        "--catalog", str(catalog), "--results", str(path.parent),
        "--extracted", str(extracted), "--output", str(output)], check=True)
    row = json.loads((output / "siglip2_manifest.jsonl").read_text())
    assert row["frame_id"] == 100
    assert row["vector_id"] == row["clip_feature_index"] == 0
    assert row["pts_time"] == 4.001
    assert Path(row["video_path"]).is_file()
    assert np.load(output / "siglip2_features.npy")[0, 0] == 1
    assert json.loads((output / "merge-report.json").read_text())["complete"]


def test_notebook_code_compiles_and_has_no_outputs():
    notebook = json.loads((REPO / "kaggle/batch2_siglip2_t4x2.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            compile("".join(cell["source"]), "notebook", "exec")
