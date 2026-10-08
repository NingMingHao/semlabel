"""End-to-end UI flow on a running server with a throwaway project (deleted at the end).

python tests/ui_flow.py http://127.0.0.1:8765 <image_folder> <screenshot_dir>
Steps: create project via the UI dialog, import a folder, pre-label + correct the first image with the
brush / SAM / bucket tools, Submit + add to memory, Suggest, auto-label the rest from the gallery,
review a low-confidence image, compress the bank, export, delete the project.
"""
import json
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

base, folder, out = sys.argv[1], sys.argv[2], sys.argv[3]
errors, steps = [], []


def api(path, method="GET", body=None):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read() or "null")


def wait_jobs():
    for _ in range(600):
        if not [j for j in api("/api/jobs?limit=5") if j["status"] in ("queued", "running")]:
            return
        time.sleep(0.5)


def step(name):
    steps.append(name)
    print("·", name, flush=True)


with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1500, "height": 950})
    pg.on("console", lambda m: errors.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.goto(base + "/#/")
    pg.wait_for_selector("text=New project")
    step("create project in the dialog")
    pg.click("text=New project")
    pg.fill(".modal input[type=text] >> nth=0", "UI flow test")
    pg.click(".modal button:has-text('Create')")
    pg.wait_for_selector(".tabs")
    pid = int(pg.url.split("/p/")[1].split("/")[0].split("?")[0])
    step("import a folder from the Import / Export tab")
    pg.click("text=Import / Export")
    pg.fill("input[placeholder^='/path/to/folder']", folder)
    pg.click("button:has-text('Import') >> nth=0")
    wait_jobs()
    time.sleep(1.5)
    assert api(f"/api/projects/{pid}")["total"] > 3
    step("open the first image in the editor")
    pg.goto(base + f"/#/p/{pid}/images")
    pg.wait_for_selector(".thumb")
    pg.click(".thumb >> nth=0")
    pg.wait_for_selector(".editor canvas")
    time.sleep(1.2)
    step("pre-label with SegFormer")
    pg.click("text=Pre-label (SegFormer)")
    pg.wait_for_selector(".busy", state="hidden", timeout=60000)
    time.sleep(0.5)
    st = pg.evaluate("() => { const m = window.__editor.mask; let n = 0; for (const v of m) if (v) n++; return n / m.length; }")
    assert st > 0.5, st
    box = pg.locator(".stage canvas").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    step("SAM click on the left half, apply class 'chair' (key 7)")
    pg.keyboard.press("7")
    pg.keyboard.press("s")
    pg.mouse.click(cx - 200, cy + 60)
    pg.wait_for_selector(".busy", state="hidden", timeout=60000)
    time.sleep(0.4)
    has_mask = pg.evaluate("() => !!window.__editor.sam.mask")
    assert has_mask, "SAM returned no mask"
    pg.keyboard.press("Enter")
    step("bucket-fill a region with 'floor' (key 1)")
    pg.keyboard.press("1"); pg.keyboard.press("g"); pg.mouse.click(cx + 120, cy + 150)
    step("brush stroke")
    pg.keyboard.press("2"); pg.keyboard.press("b")
    pg.mouse.move(cx + 150, cy - 120); pg.mouse.down(); pg.mouse.move(cx + 250, cy - 120, steps=8); pg.mouse.up()
    pg.screenshot(path=f"{out}/flow_01_editor.png")
    step("Submit + add to memory (Ctrl+Shift+Enter) -> next image")
    first_url = pg.url
    pg.keyboard.press("Control+Shift+Enter")
    pg.wait_for_function("u => location.href !== u", arg=first_url, timeout=20000)
    time.sleep(1.5)
    mem = api(f"/api/projects/{pid}/memory")
    assert mem["references"] == 1 and mem["size"] > 0, mem
    step("auto-label the opened image (A)")
    pg.keyboard.press("a")
    pg.wait_for_selector(".busy", state="hidden", timeout=60000)
    time.sleep(0.5)
    pg.screenshot(path=f"{out}/flow_02_autolabel.png")
    pg.keyboard.press("Control+s")
    time.sleep(1.0)
    step("gallery: suggest images, then auto-label all new")
    pg.goto(base + f"/#/p/{pid}/images")
    pg.wait_for_selector(".thumb")
    pg.click("text=Suggest images to label")
    pg.fill(".modal input[type=number]", "3")
    pg.click(".modal button:has-text('Suggest')")
    wait_jobs(); time.sleep(1.5)
    assert pg.locator(".thumb.sel").count() == 3, pg.locator(".thumb.sel").count()
    pg.screenshot(path=f"{out}/flow_03_suggest.png", full_page=True)
    pg.click(".chips button >> nth=0")
    time.sleep(0.8)
    pg.click("button:has-text('Auto-label new')")
    wait_jobs(); time.sleep(2.0)
    counts = api(f"/api/projects/{pid}")["counts"]
    assert counts["new"] == 0, counts
    step("review lowest-confidence auto-label")
    pg.goto(base + f"/#/p/{pid}/images?status=predicted&sort=conf")
    pg.wait_for_selector(".thumb")
    pg.screenshot(path=f"{out}/flow_04_review_queue.png", full_page=True)
    pg.click(".thumb >> nth=0")
    pg.wait_for_selector(".editor canvas"); time.sleep(1.2)
    pg.keyboard.press("c"); time.sleep(0.3)
    pg.screenshot(path=f"{out}/flow_05_lowconf.png")
    pg.keyboard.press("c")
    step("viewing an auto-label and moving on does not save it")
    viewed = int(pg.url.split("/edit/")[1].split("?")[0])
    pg.keyboard.press("Alt+ArrowRight")
    pg.wait_for_function("u => !location.href.includes(u)", arg=f"/edit/{viewed}?", timeout=20000)
    time.sleep(1.2)
    st = api(f"/api/projects/{pid}/images/{viewed}")
    assert st["status"] == "predicted" and not st["has_label"], st
    step("an edit is kept as in progress when leaving through the gallery link")
    edited = int(pg.url.split("/edit/")[1].split("?")[0])
    pg.keyboard.press("3"); pg.keyboard.press("b")
    box = pg.locator(".stage canvas").bounding_box()
    pg.mouse.move(box["x"] + 300, box["y"] + 300); pg.mouse.down(); pg.mouse.move(box["x"] + 420, box["y"] + 320, steps=6); pg.mouse.up()
    pg.click("text=← Gallery"); time.sleep(1.5)
    st = api(f"/api/projects/{pid}/images/{edited}")
    assert st["status"] == "in_progress" and st["has_label"], st
    pg.goto(base + f"/#/p/{pid}/edit/{viewed}?status=predicted&sort=conf"); pg.wait_for_selector(".editor canvas"); time.sleep(1.2)
    pg.keyboard.press("Control+Enter")
    time.sleep(1.5)
    assert api(f"/api/projects/{pid}/images/{viewed}")["status"] == "reviewed"
    step("memory bank: compress to 20%")
    pg.goto(base + f"/#/p/{pid}/memory"); pg.wait_for_selector("text=Coreset compression")
    before = api(f"/api/projects/{pid}/memory")["size"]
    pg.click("button:has-text('Compress')")
    wait_jobs(); time.sleep(1.5)
    after = api(f"/api/projects/{pid}/memory")["size"]
    assert after < before, (before, after)
    pg.screenshot(path=f"{out}/flow_06_memory.png", full_page=True)
    step("export from the UI")
    pg.goto(base + f"/#/p/{pid}/data"); pg.wait_for_selector("text=Export zip")
    pg.click("text=Export zip")
    wait_jobs(); time.sleep(1.5)
    ex = api(f"/api/projects/{pid}/exports")
    assert ex and ex[0]["size"] > 0, ex
    step("delete the project from Settings")
    pg.goto(base + f"/#/p/{pid}/settings"); pg.wait_for_selector("text=Delete project")
    pg.click("text=Delete project")
    pg.click(".modal button:has-text('Delete project')")
    time.sleep(1.0)
    assert all(p["id"] != pid for p in api("/api/projects"))
    b.close()
print("STEPS OK:", len(steps))
print("ERRORS:", len(errors))
for e in errors[:20]:
    print(" ", e)
