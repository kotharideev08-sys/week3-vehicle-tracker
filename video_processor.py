"""
Vehicle Motion Tracker - classical CV, works with STATIC and MOVING cameras
===========================================================================
Pipeline
  1. Ego-motion  : ORB keypoints -> Hamming BFMatcher -> Lowe ratio test ->
                   RANSAC homography (frame t-1 -> t) = camera motion model.
  2. Detection   : --detector yolo   : YOLO (ultralytics) finds vehicles every
                                       few frames (best for dashcams: cars ahead
                                       move WITH the camera, so motion cues fail)
                   --detector motion : static camera -> MOG2 background subtraction
                                       moving camera -> LK flow vs camera model
                   --detector auto   : YOLO if installed, else motion (default)
  3. Tracking    : per-vehicle Shi-Tomasi features + pyramidal Lucas-Kanade,
                   forward-backward check, median outlier rejection, box
                   translation + scale update, IoU re-association.
  4. Output      : annotated MP4 + animated GIF for the README.

Usage:
    * Press the Run button in VS Code (edit the SETTINGS block below first if you like), or
    * python video_processor.py                      (same, uses SETTINGS)
    * python video_processor.py --video clip.mp4 ... (command-line options override SETTINGS)
Or:
    python3 video_processor.py --video clip.mp4 [--mode auto|moving|static]
Requires: opencv-python (or python3-opencv), numpy, pillow
Optional: ultralytics (pip install ultralytics) for the YOLO detector
"""
# ============================== SETTINGS ====================================
# Edit these, then just press the Run button. Command-line options override them.
VIDEO = ""            # "" = a file-chooser window opens;  or e.g. "~/Downloads/video1.mp4"
CAMERA = "auto"       # "auto" | "moving" (dashcam) | "static" (CCTV)
HOOD_PERCENT = 45     # % of the frame bottom covered by your own car's hood (0 = none)
QUALITY = "balanced"  # "fast" | "balanced" | "best"
PREVIEW = "browser"   # "none" | "browser" (safe live preview) | "window"
ASK = False           # True = ask all of the above in the terminal instead
# ============================================================================

# --- Auto-switch to the virtualenv that has ultralytics (so Run always works) --
import glob as _glob
import importlib.util as _ilu
import os as _os
import sys as _sys

if _ilu.find_spec("ultralytics") is None and not _os.environ.get("VP_REEXEC"):
    _home = _os.path.expanduser("~")
    for _v in ("cvenv", ".venv", "venv"):
        _py = _os.path.join(_home, _v, "bin", "python")
        _has = _glob.glob(_os.path.join(_home, _v, "lib", "python*", "site-packages", "ultralytics"))
        if _has and _os.path.exists(_py) and \
                _os.path.realpath(_sys.prefix) != _os.path.realpath(_os.path.join(_home, _v)):
            _os.environ["VP_REEXEC"] = "1"
            print(f"[auto] ultralytics not found in {_sys.executable}; switching to {_py}")
            _os.execv(_py, [_py] + _sys.argv)

import argparse
import os
import subprocess
import sys
import threading
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --- Linux/GNOME crash guard for cv2.imshow (pip OpenCV bundles Qt). ---------
# Must be set BEFORE cv2 is imported. Only matters for --preview window.
if sys.platform.startswith("linux"):
    os.environ["QT_QPA_PLATFORMTHEME"] = ""          # don't load GNOME/GTK theme plugin
    os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")
    if os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # X11 / XWayland

import cv2
import numpy as np
from PIL import Image

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


# ----------------------------------------------------------------------------
# Input helpers
# ----------------------------------------------------------------------------
def clean_path(s):
    return os.path.expanduser(s.strip().strip("'\""))


def find_videos():
    roots = {Path.cwd(), Path(__file__).resolve().parent, Path.home(),
             Path.home() / "Downloads", Path.home() / "Videos"}
    found = []
    for r in roots:
        if r.is_dir():
            for p in sorted(r.iterdir()):
                if p.suffix.lower() in VIDEO_EXT and p.is_file() and p not in found \
                        and not p.stem.endswith(("_tracked", "_demo")):
                    found.append(p)
    return found


def pick_video():
    vids = find_videos()
    print("\nSelect input video:")
    for i, v in enumerate(vids, 1):
        print(f"  [{i}] {v}")
    print("  [b] browse with file dialog")
    print("  [0] webcam")
    print("  or type / drag-drop a full path")
    while True:
        ans = input("> ").strip()
        if ans.isdigit() and 1 <= int(ans) <= len(vids):
            return str(vids[int(ans) - 1])
        if ans == "0":
            return 0
        if ans.lower() == "b":
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                f = filedialog.askopenfilename(
                    title="Choose a video",
                    filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v")])
                root.destroy()
                if f:
                    return f
            except Exception:
                print("File dialog unavailable (sudo apt install python3-tk). Type a path instead.")
            continue
        p = clean_path(ans)
        if os.path.isfile(p):
            return p
        print(f"Not found: {p}  - try again.")


def gui_pick_video():
    """File-chooser window (no typing). Returns a path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        f = filedialog.askopenfilename(
            title="Choose a video",
            initialdir=str(Path.home() / "Downloads"),
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v")])
        root.destroy()
        return f or None
    except Exception:
        return None


def ask_mode():
    print("\nCamera type:\n  [1] auto-detect (default)\n  [2] moving (dashcam, drone, handheld)\n  [3] static (CCTV, tripod)")
    a = input("> ").strip()
    return {"2": "moving", "3": "static"}.get(a, "auto")


def ask_yes_no(q, default=False):
    d = "Y/n" if default else "y/N"
    a = input(f"{q} [{d}]: ").strip().lower()
    return default if not a else a.startswith("y")


def ask_preview():
    print("\nLive preview:\n  [1] none (default, fastest)\n"
          "  [2] browser  (recommended - works on any Linux, cannot crash)\n"
          "  [3] window   (OpenCV window - may crash on some GNOME setups)")
    return {"2": "browser", "3": "window"}.get(input("> ").strip(), "none")


class BrowserPreview:
    """Live MJPEG preview served on http://127.0.0.1:<port> (stdlib only)."""
    PAGE = (b"<!doctype html><title>Vehicle tracker</title>"
            b"<body style='margin:0;background:#111;color:#ccc;font-family:sans-serif;text-align:center'>"
            b"<p>Live preview - stop with Ctrl+C in the terminal</p>"
            b"<img src='/stream' style='max-width:98vw;max-height:88vh'></body>")

    def __init__(self, port=8080):
        self.cond = threading.Condition()
        self.jpg, self.seq, self.clients, self.closed = None, 0, 0, False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if not self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(outer.PAGE)))
                    self.end_headers()
                    self.wfile.write(outer.PAGE)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with outer.cond:
                    outer.clients += 1
                last = -1
                try:
                    while not outer.closed:
                        with outer.cond:
                            outer.cond.wait_for(lambda: outer.seq != last or outer.closed, timeout=2)
                            if outer.closed:
                                break
                            jpg, last = outer.jpg, outer.seq
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                except OSError:  # browser tab closed
                    pass
                finally:
                    with outer.cond:
                        outer.clients -= 1

        for p in range(port, port + 20):
            try:
                self.server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
                self.port = p
                break
            except OSError:
                continue
        else:
            raise OSError("no free port for preview server")
        self.server.daemon_threads = True

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{self.port}"
        print(f"Live preview: {url}  (opening browser...)")
        try:
            webbrowser.open(url)
        except Exception:
            print("Could not auto-open the browser - open the URL above manually.")

    def push(self, frame):
        if self.clients == 0:
            return
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.cond:
                self.jpg, self.seq = buf.tobytes(), self.seq + 1
                self.cond.notify_all()

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        self.server.shutdown()


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def apply_h(H, pt):
    v = H @ np.array([pt[0], pt[1], 1.0])
    return v[:2] / v[2]


def corner_shift(H, w, h):
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    p = cv2.perspectiveTransform(c, H)
    return float(np.linalg.norm(p - c, axis=2).mean())


# ----------------------------------------------------------------------------
# 1. Ego-motion: ORB + ratio test + RANSAC homography
# ----------------------------------------------------------------------------
class EgoMotion:
    def __init__(self, nfeatures=2000, ratio=0.75, ransac_thr=3.0):
        self.orb = cv2.ORB_create(nfeatures)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)  # Hamming distance for binary ORB
        self.ratio = ratio
        self.thr = ransac_thr
        self.prev = None

    def update(self, gray, mask=None):
        """Returns (H prev->curr, ok, n_inliers). mask: 255 = usable region."""
        h, w = gray.shape
        kp, des = self.orb.detectAndCompute(gray, mask)
        H, ok, n_inl = np.eye(3), False, 0
        if (self.prev is not None and des is not None and self.prev[1] is not None
                and len(kp) >= 12 and len(self.prev[0]) >= 12):
            pk, pd = self.prev
            pairs = [m for m in self.bf.knnMatch(pd, des, k=2) if len(m) == 2]
            good = [a for a, b in pairs if a.distance < self.ratio * b.distance]  # Lowe
            if len(good) >= 12:
                src = np.float32([pk[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
                dst = np.float32([kp[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
                Hc, inl = cv2.findHomography(src, dst, cv2.RANSAC, self.thr)  # RANSAC
                if Hc is not None and inl is not None and inl.sum() >= 10 \
                        and corner_shift(Hc, w, h) < 0.25 * w:
                    H, ok, n_inl = Hc, True, int(inl.sum())
        self.prev = (kp, des)
        return H, ok, n_inl


# ----------------------------------------------------------------------------
# 2. Detection
# ----------------------------------------------------------------------------
def boxes_from_binary(binary, min_area, shape):
    h, w = shape[:2]
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / float(bh)
        if bw * bh > 0.5 * w * h or ar < 0.25 or ar > 5.0:
            continue
        boxes.append((x, y, bw, bh))
    return boxes


def detect_static(fg_raw, min_area, shape):
    fg = cv2.threshold(fg_raw, 200, 255, cv2.THRESH_BINARY)[1]  # drop shadows (127)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    fg = cv2.dilate(fg, k, iterations=1)
    return boxes_from_binary(fg, min_area, shape), fg


class MovingDetector:
    """Independent-motion detector: LK flow minus camera-model prediction."""

    def __init__(self, shape, thr=2.0, valid=None):
        self.thr = thr
        self.valid = valid
        self.acc = np.zeros(shape[:2], np.float32)
        self.H = np.eye(3)

    def update(self, prev_gray, gray, H, h_ok):
        if prev_gray is None:
            return
        if not h_ok:  # no reliable camera model: just fade the memory
            self.acc *= 0.6
            return
        h, w = gray.shape
        self.acc = cv2.warpPerspective(self.acc, H, (w, h)) * 0.6  # keep accumulator aligned
        pts = cv2.goodFeaturesToTrack(prev_gray, 1500, 0.005, 6, mask=self.valid)
        if pts is None or len(pts) < 20:
            return
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK_PARAMS)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **LK_PARAMS)
        fb = np.linalg.norm(pts - back, axis=2).ravel()
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        # refine the ORB camera model with the dense LK correspondences
        pred = cv2.perspectiveTransform(pts, H)
        res = np.linalg.norm(nxt - pred, axis=2).ravel()
        seed = good & (res < 8.0)
        if seed.sum() >= 30:
            Hr, inl = cv2.findHomography(pts[seed], nxt[seed], cv2.RANSAC, 1.5)
            if Hr is not None and inl is not None and inl.sum() >= 25:
                H = Hr
                pred = cv2.perspectiveTransform(pts, H)
                res = np.linalg.norm(nxt - pred, axis=2).ravel()
        self.H = H
        cand = good & (res > self.thr) & (res < 60)
        if good.sum() > 0 and cand.sum() > 0.4 * good.sum():
            self.acc *= 0.6  # camera model unreliable this frame (parallax / zoom)
            return
        m = np.zeros((h, w), np.uint8)
        for p in nxt[cand].reshape(-1, 2):
            cv2.circle(m, (int(p[0]), int(p[1])), 6, 255, -1)
        self.acc += (m > 0).astype(np.float32)

    def boxes(self, min_area, shape):
        binary = (self.acc > 1.0).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        return boxes_from_binary(binary, min_area, shape), binary


class YoloDetector:
    """Appearance-based vehicle detector (COCO car/motorcycle/bus/truck).

    Runs on the ORIGINAL-resolution frame (small, distant cars keep their pixels),
    with your own car's hood cropped away, and returns boxes in processing coordinates.
    """
    VEHICLE_CLASSES = [2, 3, 5, 7]

    def __init__(self, weights="yolov8s.pt", conf=0.15, hood=0.0, imgsz=1280):
        from ultralytics import YOLO  # raises ImportError if not installed
        self.model = YOLO(weights)
        self.conf, self.hood, self.imgsz = conf, hood, imgsz
        self.total_raw = 0

    def detect(self, raw, sx=1.0, sy=1.0, return_all=False):
        Hr = raw.shape[0]
        crop = raw[: max(int(Hr * (1 - self.hood)), 32)] if self.hood > 0 else raw
        r = self.model(crop, conf=self.conf, classes=self.VEHICLE_CLASSES,
                       imgsz=self.imgsz, agnostic_nms=True, verbose=False)[0]
        out = []
        for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy():
            x, y, w, h = x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy
            if w < 6 or h < 6:
                continue
            out.append((int(x), int(y), int(w), int(h)))
        self.total_raw += len(out)
        return out


# ----------------------------------------------------------------------------
# 3. Tracking
# ----------------------------------------------------------------------------
class Track:
    _next_id = 1

    def __init__(self, box, gray, mask, shrink=False):
        self.shrink = shrink  # use only the box core for features (avoids background points)
        self.unseen = 0
        self.id = Track._next_id
        Track._next_id += 1
        self.box = np.array(box, dtype=float)  # x, y, w, h
        self.pts = None
        self.trail = deque(maxlen=40)
        self.missed = 0
        self.hits = 1
        self.speed = 0.0  # camera-compensated px / frame
        self.refresh_features(gray, mask)
        self.trail.append(self.center())

    def center(self):
        x, y, w, h = self.box
        return int(x + w / 2), int(y + h / 2)

    def refresh_features(self, gray, mask=None):
        x, y, w, h = [int(v) for v in self.box]
        if self.shrink and mask is None:
            x, y, w, h = int(x + 0.15 * w), int(y + 0.15 * h), int(0.7 * w), int(0.7 * h)
        roi = np.zeros_like(gray)
        roi[max(y, 0):max(y + h, 0), max(x, 0):max(x + w, 0)] = 255
        if mask is not None:
            m2 = cv2.bitwise_and(roi, mask)
            if cv2.countNonZero(m2) > 50:
                roi = m2
        if w * h < 40 * 40:  # distant car: allow denser / weaker corners
            self.pts = cv2.goodFeaturesToTrack(gray, 25, 0.003, 2, mask=roi)
        else:
            self.pts = cv2.goodFeaturesToTrack(gray, 40, 0.01, 5, mask=roi)

    def update_flow(self, prev_gray, gray, H):
        if self.pts is None or len(self.pts) < 4:
            return False
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, self.pts, None, **LK_PARAMS)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **LK_PARAMS)
        fb = np.linalg.norm(self.pts - back, axis=2).ravel()
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        if ok.sum() < 4:
            return False
        old = self.pts[ok].reshape(-1, 2)
        new = nxt[ok].reshape(-1, 2)
        d = new - old
        med = np.median(d, axis=0)
        inl = np.linalg.norm(d - med, axis=1) < 3.0
        if inl.sum() < 4:
            return False
        old_i, new_i = old[inl], new[inl]
        shift = (new_i - old_i).mean(axis=0)

        c0 = np.array(self.center(), float)
        rel = shift - (apply_h(H, c0) - c0)  # motion not explained by camera
        self.speed = float(np.linalg.norm(rel))

        self.box[:2] += shift
        if len(new_i) >= 6:  # scale update (approaching / receding vehicles)
            s_old = np.linalg.norm(old_i - old_i.mean(0), axis=1).mean()
            s_new = np.linalg.norm(new_i - new_i.mean(0), axis=1).mean()
            if s_old > 2:
                r = float(np.clip(s_new / s_old, 0.97, 1.03))
                cen = self.box[:2] + self.box[2:] / 2
                self.box[2:] *= r
                self.box[:2] = cen - self.box[2:] / 2
        self.pts = new_i.reshape(-1, 1, 2).astype(np.float32)
        self.trail.append(self.center())
        return True


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def detect_mode(cap, scale, hood=0.0):
    """Pre-scan first frames: is the camera moving?"""
    ego = EgoMotion(1000)
    shifts, n = [], 0
    while n < 25:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                         if scale < 1 else f, cv2.COLOR_BGR2GRAY)
        vm = None
        if hood > 0:
            vm = np.full(g.shape, 255, np.uint8)
            vm[int(g.shape[0] * (1 - hood)):, :] = 0
        H, good, _ = ego.update(g, vm)
        if n > 0 and good:
            shifts.append(corner_shift(H, g.shape[1], g.shape[0]))
        n += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    med = float(np.median(shifts)) if shifts else 0.0
    return ("moving" if med > 1.0 else "static"), med


def main():
    ap = argparse.ArgumentParser(description="Classical vehicle tracker (static + moving cameras)")
    ap.add_argument("--video", default=None, help="video path, or 0 for webcam (omit to be prompted)")
    ap.add_argument("--mode", choices=["auto", "moving", "static"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--width", type=int, default=None, help="processing width (smaller = faster)")
    ap.add_argument("--min-area", type=int, default=400, help="min blob area at processing width")
    ap.add_argument("--motion-thr", type=float, default=2.0, help="moving mode: px residual vs camera model")
    ap.add_argument("--detect-every", type=int, default=None)
    ap.add_argument("--confirm", type=int, default=None, help="detections needed before a track is shown")
    ap.add_argument("--detector", choices=["auto", "yolo", "motion"], default="auto")
    ap.add_argument("--weights", default=None, help="YOLO weights (auto-downloaded by ultralytics)")
    ap.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold")
    ap.add_argument("--quality", choices=["fast", "balanced", "best"], default="balanced",
                    help="accuracy/speed preset (best = bigger model, higher resolution, detect every frame)")
    ap.add_argument("--imgsz", type=int, default=None, help="YOLO input size override")
    ap.add_argument("--hood", type=float, default=None,
                    help="fraction (0-0.6) of frame bottom covered by your own car's hood")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--gif-width", type=int, default=480)
    ap.add_argument("--gif-max-frames", type=int, default=120)
    ap.add_argument("--gif-step", type=int, default=2)
    ap.add_argument("--preview", choices=["none", "browser", "window"], default=None,
                    help="live preview: browser (safe), window (OpenCV, may crash on some Linux), none")
    ap.add_argument("--check", action="store_true",
                    help="detector self-test: run YOLO on one frame, save <name>_check.jpg and exit")
    ap.add_argument("--debug", action="store_true", help="also draw raw detector boxes (white)")
    ap.add_argument("--port", type=int, default=8080, help="browser preview port")
    ap.add_argument("--show", action="store_true", help="same as --preview browser")
    args = ap.parse_args()
    settings_mode = (len(sys.argv) == 1) and not ASK  # plain Run button press
    if settings_mode:
        args.video = clean_path(VIDEO) if VIDEO else None
        args.mode = CAMERA
        args.hood = HOOD_PERCENT / 100.0
        args.quality = QUALITY
        args.preview = PREVIEW
    preset = {  # weights, imgsz, width, detect_every, conf
        "fast":     ("yolov8n.pt", 960, 960, 3, 0.25),
        "balanced": ("yolov8s.pt", 1280, 1280, 2, 0.15),
        "best":     ("yolov8m.pt", 1280, 1280, 1, 0.10),
    }[args.quality]
    args.weights = args.weights or preset[0]
    args.imgsz = args.imgsz or preset[1]
    args.width = args.width or preset[2]
    args.detect_every = args.detect_every or preset[3]
    args.conf = args.conf if args.conf is not None else preset[4]

    interactive = args.video is None and not settings_mode
    if args.video is None:
        video = (gui_pick_video() if settings_mode else None) or pick_video()
    elif args.video == "0":
        video = 0
    else:
        video = clean_path(args.video)
        if not os.path.isfile(video):
            print(f"File not found: {video}")
            video = pick_video()
            interactive = True
    if args.mode is None:
        args.mode = ask_mode() if interactive and sys.stdin.isatty() else "auto"
    if args.hood is None:
        args.hood = 0.0
        if interactive and sys.stdin.isatty():
            a = input("\nPercent of frame bottom covered by your own car's hood (Enter = 0): ").strip()
            try:
                args.hood = float(a) / 100.0 if a else 0.0
            except ValueError:
                args.hood = 0.0
    args.hood = float(np.clip(args.hood, 0.0, 0.6))
    if args.preview is None:
        if args.show:
            args.preview = "browser"
        elif interactive and sys.stdin.isatty():
            args.preview = ask_preview()
        else:
            args.preview = "none"

    stem = "webcam" if video == 0 else Path(video).stem
    outdir = Path(video).resolve().parent if video != 0 else Path.cwd()
    args.out = args.out or str(outdir / f"{stem}_tracked.mp4")
    args.gif = args.gif or str(outdir / f"{stem}_demo.gif")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}. Try: ffmpeg -i in.mp4 -c:v libx264 out.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps > 120:
        fps = 30
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, args.width / W0)
    W, H = int(W0 * scale) // 2 * 2, int(H0 * scale) // 2 * 2

    mode = args.mode
    if mode == "auto":
        if video == 0:
            mode, cam = "moving", 0.0
        else:
            mode, cam = detect_mode(cap, scale, args.hood)
        print(f"Auto-detected camera: {mode} (median background shift {cam:.2f}px/frame)")

    yolo = None
    if args.detector in ("auto", "yolo"):
        try:
            yolo = YoloDetector(args.weights, args.conf, args.hood, args.imgsz)
        except Exception as e:  # ImportError, download failure, ...
            msg = (f"\n*** YOLO detector unavailable: {e}\n"
                   f"*** Python in use: {sys.executable}\n"
                   f"*** Fix: activate the venv that has ultralytics (source ~/cvenv/bin/activate)\n"
                   f"***      or install it:  {sys.executable} -m pip install ultralytics\n"
                   f"***      (in VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter' -> ~/cvenv/bin/python)")
            if args.detector == "yolo" or mode == "moving":
                # motion-only detection cannot see cars driving with the camera, so stop loudly
                raise SystemExit(msg + "\n*** (use --detector motion to force the weak motion-only detector)")
            print(msg + "\n*** Static camera -> continuing with background subtraction.")
    det_name = "yolo" if yolo else "motion"
    if args.confirm is None:
        args.confirm = 1 if yolo else 2
    warmup = args.warmup if args.warmup is not None else (0 if yolo else (30 if mode == "static" else 4))
    valid = np.full((H, W), 255, np.uint8)
    if args.hood > 0:
        valid[int(H * (1 - args.hood)):, :] = 0

    print(f"Input : {video}\nMode  : {mode} camera | detector: {det_name} ({args.weights}, imgsz {args.imgsz}) | hood mask: {args.hood:.0%}\nOutput: {args.out}\nGIF   : {args.gif}")
    sx, sy = W / float(W0), H / float(H0)
    if args.check:
        if not yolo:
            raise SystemExit("--check needs the YOLO detector (ultralytics).")
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(30, max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 0)))
        ok, raw = cap.read()
        if not ok:
            raise SystemExit("Could not read a frame for the check.")
        dets = yolo.detect(raw, sx, sy)
        vis = cv2.resize(raw, (W, H), interpolation=cv2.INTER_AREA)
        for (x, y, w, h) in dets:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if args.hood > 0:
            cv2.line(vis, (0, int(H * (1 - args.hood))), (W, int(H * (1 - args.hood))), (0, 0, 255), 2)
        path = str(outdir / f"{stem}_check.jpg")
        cv2.imwrite(path, vis)
        print(f"\nCHECK: YOLO found {len(dets)} vehicle(s) on this frame -> {path}")
        print("Green = detections, red line = hood cut-off. If boxes are missing, try --quality best "
              "or --conf 0.08; if the red line cuts through cars, lower --hood.")
        return
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    ego = EgoMotion()
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=True) \
        if (mode == "static" and not yolo) else None
    mdet = MovingDetector((H, W), args.motion_thr, valid) if (mode == "moving" and not yolo) else None
    preview, show_window = None, False
    if args.preview == "browser":
        preview = BrowserPreview(args.port)
        preview.start()
    elif args.preview == "window":
        show_window = True

    tracks, gif_frames = [], []
    total_shown = 0
    last_raw = []
    prev_gray, idx = None, 0
    rng = np.random.RandomState(7)
    colors = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw = frame
            frame = cv2.resize(raw, (W, H), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # camera model (identity for static cameras)
            if mode == "moving":
                Hm, h_ok, n_inl = ego.update(gray, valid)
                if mdet is not None:
                    mdet.H = Hm
                    mdet.update(prev_gray, gray, Hm, h_ok)
                    Hm = mdet.H  # refined camera model
            else:
                Hm, h_ok, n_inl = np.eye(3), True, 0
                fg_raw = bg.apply(frame) if bg is not None else None

            # propagate tracks with LK
            if prev_gray is not None:
                for t in tracks:
                    if not t.update_flow(prev_gray, gray, Hm):
                        t.missed += 1

            # periodic detection + IoU association
            if idx >= warmup and idx % args.detect_every == 0:
                if yolo:
                    boxes, mask = yolo.detect(raw, sx, sy), None
                    last_raw = boxes
                elif mode == "moving":
                    boxes, mask = mdet.boxes(args.min_area, gray.shape)
                else:
                    boxes, mask = detect_static(fg_raw, args.min_area, gray.shape)
                unmatched = list(range(len(boxes)))
                for t in tracks:
                    best, best_i = 0.3, None
                    for i in unmatched:
                        s = iou(tuple(t.box), boxes[i])
                        if s > best:
                            best, best_i = s, i
                    if best_i is not None:
                        unmatched.remove(best_i)
                        a = 0.7 if yolo else 0.5
                        t.box = (1 - a) * t.box + a * np.array(boxes[best_i], dtype=float)
                        t.missed = 0
                        t.unseen = 0
                        t.hits += 1
                        t.refresh_features(gray, mask)
                    else:
                        t.unseen += 1
                dup_thr = 0.5 if yolo else 0.1
                for i in unmatched:
                    if any(iou(tuple(t.box), boxes[i]) > dup_thr for t in tracks):
                        continue
                    tracks.append(Track(boxes[i], gray, mask, shrink=bool(yolo)))

            # prune
            def alive(t):
                cx, cy = t.center()
                w_, h_ = t.box[2], t.box[3]
                if w_ < 8 or h_ < 8 or w_ * h_ > 0.5 * W * H:
                    return False
                if yolo and t.unseen > 6:  # detector has not confirmed it for a while
                    return False
                lim = 10 if t.hits >= args.confirm else 4
                return t.missed <= lim and -50 < cx < W + 50 and -50 < cy < H + 50
            tracks = [t for t in tracks if alive(t)]

            # draw
            vis = frame.copy()
            shown = 0
            for t in tracks:
                if t.hits < args.confirm or (t.missed > 3 and not yolo):
                    continue
                shown += 1
                c = colors.setdefault(t.id, tuple(int(v) for v in rng.randint(60, 255, 3)))
                x, y, w, h = [int(v) for v in t.box]
                cv2.rectangle(vis, (x, y), (x + w, y + h), c, 2)
                cv2.putText(vis, f"ID {t.id} | {t.speed:.1f}px/f", (x, max(y - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
                if t.pts is not None:
                    for p in t.pts.reshape(-1, 2):
                        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)
                pts = np.array(t.trail, dtype=np.int32).reshape(-1, 1, 2)
                if len(pts) > 1:
                    cv2.polylines(vis, [pts], False, c, 2)
            if args.debug:
                for (x, y, w, h) in last_raw:
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.putText(vis, f"{det_name}/{mode} cam | frame {idx} | vehicles {shown}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            writer.write(vis)
            if idx % args.gif_step == 0 and len(gif_frames) < args.gif_max_frames and idx >= warmup:
                s = args.gif_width / W
                small = cv2.resize(vis, (args.gif_width, int(H * s)), interpolation=cv2.INTER_AREA)
                gif_frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            if preview is not None:
                preview.push(vis)
            elif show_window:
                try:
                    cv2.imshow("tracker", vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                except cv2.error:
                    print("Preview window unavailable, continuing without it.")
                    show_window = False

            prev_gray = gray
            idx += 1
            total_shown += shown
            if idx % 50 == 0:
                print(f"  processed {idx} frames...", end="\r")

    except KeyboardInterrupt:
        print("\nStopped by user - saving what was processed so far...")

    cap.release()
    writer.release()
    if preview is not None:
        preview.close()
    if show_window:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    if gif_frames:
        dur = int(1000 / fps * args.gif_step)
        gif_frames[0].save(args.gif, save_all=True, append_images=gif_frames[1:],
                           duration=dur, loop=0, optimize=True)
        print(f"\nSaved {args.gif} ({len(gif_frames)} frames)")
    print(f"Saved {args.out}")
    if settings_mode and gif_frames:
        try:
            subprocess.Popen(["xdg-open", args.gif], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    if yolo:
        print(f"YOLO raw detections over the run: {yolo.total_raw}")
    if total_shown == 0:
        print("\n!!! No vehicles were tracked in the whole video. Check: the 'detector:' line above says "
              "yolo; --hood is a fraction (0.4, not 40) when given on the command line; "
              "try --quality best or --conf 0.15.")


if __name__ == "__main__":
    main()