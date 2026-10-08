"""API integration tests: the whole labeling workflow through HTTP (small DINOv3 for speed)."""
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from semlabel.server.app import create_app  # noqa: E402

from conftest import needs_weights  # noqa: E402

# optional: a ROS 2 bag (.mcap file or folder) and one of its image topics for the bag-import test
BAG = Path(os.environ.get("SEMLABEL_TEST_BAG", "/nonexistent"))
BAG_TOPIC = os.environ.get("SEMLABEL_TEST_BAG_TOPIC", "")


def scene(seed, swap=False, H=120, W=160):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    stripes = np.stack([150 + 60 * np.sin(xx / 4.0), 90 + 40 * np.sin(xx / 4.0), 40 + 0 * xx], -1)
    carpet = np.stack([30 + 0 * xx, 60 + 0 * xx, 170 + 0 * xx], -1) + rng.randint(-40, 40, (H, W, 3))
    left = (xx < W // 2) if not swap else (xx >= W // 2)
    return np.where(left[..., None], stripes, carpet).clip(0, 255).astype(np.uint8), np.where(left, 1, 2).astype(np.uint8)


def png_bytes(a):
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return b.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("sl")
    return TestClient(create_app(d)), d


def wait(c, jid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/jobs/{jid}").json()
        if r["status"] in ("done", "failed", "cancelled"):
            return r
        time.sleep(0.2)
    raise TimeoutError(jid)


@needs_weights
def test_full_workflow(client, tmp_path):
    c, d = client
    p = c.post("/api/projects", json={"name": "Test", "template": "empty", "classes": ["wood", "carpet"],
                                      "config": {"model": "small", "image_size": 256}}).json()
    pid = p["id"]
    assert [x["name"] for x in p["classes"]] == ["wood", "carpet"]
    # upload 4 images (one duplicate)
    files = [("files", (f"img{i}.png", png_bytes(scene(i, swap=i % 2 == 1)[0]), "image/png")) for i in range(4)]
    files.append(("files", ("dup.png", png_bytes(scene(0)[0]), "image/png")))
    r = c.post(f"/api/projects/{pid}/images", files=files).json()
    assert len(r["added"]) == 4 and r["duplicates"] == 1
    ids = r["added"]
    im = c.get(f"/api/projects/{pid}/images/{ids[0]}").json()
    assert (im["width"], im["height"], im["status"]) == (160, 120, "new")
    assert c.get(f"/api/projects/{pid}/images/{ids[0]}/thumb").status_code == 200
    # empty label is all zeros
    lab = c.get(f"/api/projects/{pid}/images/{ids[0]}/label")
    assert lab.headers["x-empty"] == "1" and len(lab.content) == 160 * 120
    # wrong size is rejected
    assert c.put(f"/api/projects/{pid}/images/{ids[0]}/label", content=b"\x00" * 10).status_code == 400
    # label image 0 and submit to memory
    m0 = scene(0)[1]
    r = c.put(f"/api/projects/{pid}/images/{ids[0]}/label?status=reviewed&add_to_memory=1", content=m0.tobytes())
    assert r.status_code == 200 and r.json()["status"] == "reviewed" and r.json()["in_memory"] == 1
    back = np.frombuffer(c.get(f"/api/projects/{pid}/images/{ids[0]}/label").content, np.uint8).reshape(120, 160)
    assert (back == m0).all()
    mem = c.get(f"/api/projects/{pid}/memory").json()
    assert mem["references"] == 1 and mem["size"] > 0 and set(mem["per_class"]) == {"1", "2"}
    # predict a swapped scene
    r = c.post(f"/api/projects/{pid}/images/{ids[1]}/predict", json={}).json()
    assert 0 <= r["conf"] <= 1 and r["image"]["status"] == "predicted"
    pred = np.frombuffer(c.get(f"/api/projects/{pid}/images/{ids[1]}/pred").content, np.uint8).reshape(120, 160)
    acc = (pred == scene(1, swap=True)[1]).mean()
    assert acc > 0.85, acc
    assert len(c.get(f"/api/projects/{pid}/images/{ids[1]}/conf").content) == 160 * 120
    # batch auto-label the rest
    j = wait(c, c.post(f"/api/projects/{pid}/jobs", json={"kind": "predict", "params": {"status": "new"}}).json()["job"])
    assert j["status"] == "done" and j["result"]["predicted"] == 2
    counts = c.get(f"/api/projects/{pid}").json()["counts"]
    assert counts["predicted"] == 3 and counts["reviewed"] == 1
    # accept a prediction -> reviewed; evaluate on it (not a reference)
    assert c.post(f"/api/projects/{pid}/images/{ids[2]}/accept", json={}).json()["status"] == "reviewed"
    c.put(f"/api/projects/{pid}/images/{ids[3]}/label?status=reviewed", content=scene(3, swap=True)[1].tobytes())
    ev = wait(c, c.post(f"/api/projects/{pid}/jobs", json={"kind": "evaluate", "params": {}}).json()["job"])
    assert ev["status"] == "done" and ev["result"]["n_images"] == 2 and ev["result"]["miou"] > 70
    assert len(c.get(f"/api/projects/{pid}/evals").json()) == 1
    # tools
    w = c.post(f"/api/projects/{pid}/images/{ids[1]}/wand", json={"x": 140, "y": 60, "thresh": 0.5})
    wm = np.frombuffer(w.content, np.uint8).reshape(120, 160)
    assert wm[60, 140] == 1 and int(w.headers["x-count"]) > 500
    sp = c.get(f"/api/projects/{pid}/images/{ids[1]}/superpixels")
    assert len(sp.content) == 160 * 120 * 2
    # memory: add second reference, compress, remove
    c.put(f"/api/projects/{pid}/images/{ids[1]}/label?status=reviewed", content=scene(1, swap=True)[1].tobytes())
    r = c.post(f"/api/projects/{pid}/memory/add", json={"ids": [ids[1]]}).json()
    assert r["size"] > mem["size"]
    j = wait(c, c.post(f"/api/projects/{pid}/memory/compress", json={"ratio": 0.5}).json()["job"])
    assert j["status"] == "done" and j["result"]["after"] <= j["result"]["before"] * 0.5 + 2
    r = c.post(f"/api/projects/{pid}/memory/remove", json={"ids": [ids[1]]}).json()
    assert c.get(f"/api/projects/{pid}/images/{ids[1]}").json()["in_memory"] == 0
    # export all formats and validate
    j = wait(c, c.post(f"/api/projects/{pid}/jobs", json={"kind": "export", "params": {
        "formats": ["index_png", "color_png", "coco", "cvat_mask"], "include": "reviewed"}}).json()["job"])
    assert j["status"] == "done", j
    name = j["result"]["file"]
    data = c.get(f"/api/projects/{pid}/exports/{name}").content
    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    assert "annotations/coco_semantic.json" in names and "cvat_mask/labelmap.txt" in names
    coco = json.loads(z.read("annotations/coco_semantic.json"))
    assert len(coco["images"]) == 4 and {c_["name"] for c_ in coco["categories"]} == {"wood", "carpet"}
    import pycocotools.mask as mu
    ann = coco["annotations"][0]
    dec = mu.decode({"size": ann["segmentation"]["size"], "counts": ann["segmentation"]["counts"].encode()})
    assert dec.sum() == ann["area"]
    idx_png = [n for n in names if n.startswith("masks/")][0]
    assert set(np.unique(np.array(Image.open(io.BytesIO(z.read(idx_png)))))) <= {0, 1, 2}
    # re-import the CVAT export into a fresh project (round trip)
    out = tmp_path / "exp.zip"
    out.write_bytes(data)
    p2 = c.post("/api/projects", json={"name": "Roundtrip", "template": "empty", "config": {"model": "small", "image_size": 256}}).json()
    for n_ in [n for n in names if n.startswith("images/")]:
        c.post(f"/api/projects/{p2['id']}/images", files=[("files", (Path(n_).name, z.read(n_), "image/png"))])
    j = wait(c, c.post(f"/api/projects/{p2['id']}/jobs", json={"kind": "import_masks", "params": {"path": str(out)}}).json()["job"])
    assert j["status"] == "done" and j["result"]["imported"] == 4, j
    assert {x["name"] for x in c.get(f"/api/projects/{p2['id']}/classes").json()} == {"wood", "carpet"}
    # class delete merges pixels
    r = c.delete(f"/api/projects/{pid}/classes/2?merge_into=1").json()
    assert [x["idx"] for x in r["classes"]] == [1]
    lab = np.frombuffer(c.get(f"/api/projects/{pid}/images/{ids[0]}/label").content, np.uint8)
    assert set(np.unique(lab)) == {1}
    # delete an image
    assert c.delete(f"/api/projects/{pid}/images/{ids[3]}").json()["ok"]
    assert c.get(f"/api/projects/{pid}/images/{ids[3]}").status_code == 404
    # events were recorded
    kinds = {e["kind"] for e in c.get(f"/api/projects/{pid}/events?limit=200").json()}
    assert {"label_save", "memory_add", "memory_compress", "export", "class_delete"} <= kinds


@needs_weights
def test_predict_without_memory_fails_cleanly(client):
    c, _ = client
    p = c.post("/api/projects", json={"name": "Empty mem", "template": "empty", "classes": ["a"], "config": {"model": "small", "image_size": 256}}).json()
    iid = c.post(f"/api/projects/{p['id']}/images", files=[("files", ("x.png", png_bytes(scene(5)[0]), "image/png"))]).json()["added"][0]
    r = c.post(f"/api/projects/{p['id']}/images/{iid}/predict", json={})
    assert r.status_code == 400 and "memory bank is empty" in r.json()["detail"]


def test_job_cancel(client, tmp_path):
    c, _ = client
    p = c.post("/api/projects", json={"name": "Cancel", "template": "empty", "classes": ["a"], "config": {"model": "small", "image_size": 256}}).json()
    folder = tmp_path / "imgs"
    folder.mkdir()
    for i in range(60):
        Image.fromarray(scene(100 + i)[0]).save(folder / f"{i:03d}.png")
    jid = c.post(f"/api/projects/{p['id']}/jobs", json={"kind": "import_folder", "params": {"path": str(folder)}}).json()["job"]
    c.post(f"/api/jobs/{jid}/cancel")
    r = wait(c, jid)
    assert r["status"] in ("cancelled", "done")
    assert c.get(f"/api/projects/{p['id']}").json()["total"] <= 60


@pytest.mark.skipif(not (BAG.exists() and BAG_TOPIC), reason="set SEMLABEL_TEST_BAG and SEMLABEL_TEST_BAG_TOPIC")
def test_bag_import(client):
    c, _ = client
    topics = c.get(f"/api/bag_topics?path={BAG}").json()
    assert BAG_TOPIC in [t["topic"] for t in topics]
    p = c.post("/api/projects", json={"name": "Bag", "template": "indoor", "config": {"model": "small", "image_size": 256}}).json()
    j = wait(c, c.post(f"/api/projects/{p['id']}/jobs", json={"kind": "import_bag", "params": {
        "path": str(BAG), "topics": [BAG_TOPIC], "every_s": 2.0, "max": 3}}).json()["job"])
    assert j["status"] == "done" and j["result"]["added"] == 3, j
    im = c.get(f"/api/projects/{p['id']}/images").json()["items"][0]
    assert im["width"] > 0 and im["height"] > 0


def test_sam_tool(client):
    from semlabel.engine import sam
    if not sam.available():
        pytest.skip("SAM weights not cached")
    c, _ = client
    p = c.post("/api/projects", json={"name": "SAM", "template": "empty", "classes": ["a"], "config": {"model": "small", "image_size": 256}}).json()
    img = np.full((120, 160, 3), 230, np.uint8)
    img[30:90, 40:120] = (200, 30, 30)                      # a red box on a white background
    iid = c.post(f"/api/projects/{p['id']}/images", files=[("files", ("box.png", png_bytes(img), "image/png"))]).json()["added"][0]
    r = c.post(f"/api/projects/{p['id']}/images/{iid}/sam", json={"points": [[80, 60, 1]]})
    assert r.status_code == 200, r.text
    m = np.frombuffer(r.content, np.uint8).reshape(120, 160)
    inside = m[30:90, 40:120].mean()
    outside = (m.sum() - m[30:90, 40:120].sum()) / (m.size - 60 * 80)
    assert inside > 0.9 and outside < 0.05
