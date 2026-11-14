"""Local RoboGenius web app and guarded Dobot Magician serial bridge.

The robot protocol implementation follows DOBOT's Magician Communication
Protocol V1.1.5. The server binds to loopback only and never moves the arm on
connect; motion is accepted only for a validated, explicitly confirmed job.
"""

from __future__ import annotations

import json
import base64
import math
import os
import struct
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


ROOT = Path(__file__).resolve().parent
HTML_FILE = ROOT / "robogenius_dobot_magician_3d_digital_twin_studio.html"
BAUD_RATE = 115200
MAX_REACH_MM = 320.0
PAPER_MAX_MM = 140.0
ARTLINE_RUNNER = ROOT / "artline_runner.py"
ARTLINE_CHECKPOINT = Path(os.environ.get("ARTLINE_CHECKPOINT", str(ROOT / "models" / "ArtLine_650.pkl")))
ARTLINE_PYTHON = Path(os.environ.get("ARTLINE_PYTHON", str(ROOT / ".artline-venv" / "Scripts" / "python.exe")))
PORTRAIT_SKETCH_RUNNER = ROOT / "portrait_sketch_runner.py"
PORTRAIT_SKETCH_MODEL = Path(os.environ.get("PORTRAIT_SKETCH_MODEL", str(ROOT / "models" / "U2Net_Portrait_Sketch.tflite")))
APP_PYTHON = Path(os.environ.get("ROBOGENIUS_PYTHON", str(ROOT / ".venv" / "Scripts" / "python.exe")))


class DobotProtocolError(RuntimeError):
    pass


class DobotLink:
    """Small serial protocol client for read-only pose and queued PTP motion."""

    def __init__(self) -> None:
        self.port = None
        self.port_name: str | None = None
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.job_thread: threading.Thread | None = None
        self.job: dict[str, Any] = {"state": "idle", "progress": 0, "total": 0, "message": "Ready"}
        self.last_pose: dict[str, Any] | None = None

    @staticmethod
    def _frame(command_id: int, rw: int, queued: int, params: bytes = b"") -> bytes:
        control = (rw & 1) | ((queued & 1) << 1)
        payload = bytes((command_id & 0xFF, control)) + params
        checksum = (-sum(payload)) & 0xFF
        return b"\xAA\xAA" + bytes((len(payload),)) + payload + bytes((checksum,))

    def _read_exact(self, count: int, timeout: float = 2.0) -> bytes:
        if self.port is None:
            raise DobotProtocolError("Dobot is disconnected")
        deadline = time.monotonic() + timeout
        data = bytearray()
        while len(data) < count and time.monotonic() < deadline:
            part = self.port.read(count - len(data))
            if part:
                data.extend(part)
        if len(data) != count:
            raise DobotProtocolError("Timed out waiting for the Dobot response")
        return bytes(data)

    def _read_frame(self, timeout: float = 2.0) -> tuple[int, int, int, bytes]:
        if self.port is None:
            raise DobotProtocolError("Dobot is disconnected")
        deadline = time.monotonic() + timeout
        previous = 0
        while time.monotonic() < deadline:
            current = self._read_exact(1, max(0.05, deadline - time.monotonic()))[0]
            if previous == 0xAA and current == 0xAA:
                break
            previous = current
        else:
            raise DobotProtocolError("No Dobot protocol frame received")
        length = self._read_exact(1)[0]
        if length < 3 or length > 96:
            raise DobotProtocolError(f"Invalid Dobot frame length: {length}")
        payload = self._read_exact(length)
        checksum = self._read_exact(1)[0]
        if (sum(payload) + checksum) & 0xFF:
            raise DobotProtocolError("Dobot frame checksum failed")
        if len(payload) < 2:
            raise DobotProtocolError("Dobot frame header is incomplete")
        return payload[0], payload[1] & 1, (payload[1] >> 1) & 1, payload[2:]

    def _request(self, command_id: int, rw: int = 0, queued: int = 0, params: bytes = b"", timeout: float = 2.0) -> bytes:
        with self.lock:
            if self.port is None:
                raise DobotProtocolError("Dobot is disconnected")
            self.port.reset_input_buffer()
            self.port.write(self._frame(command_id, rw, queued, params))
            self.port.flush()
            response_id, response_rw, response_queued, response_params = self._read_frame(timeout)
            if response_id != command_id:
                raise DobotProtocolError(f"Unexpected response command {response_id}; expected {command_id}")
            if response_rw != rw:
                raise DobotProtocolError("Dobot response mode did not match the request")
            return response_params

    def _get_pose(self) -> dict[str, Any]:
        data = self._request(10, timeout=2.5)
        if len(data) != 32:
            raise DobotProtocolError(f"Unexpected pose response size: {len(data)}")
        x, y, z, r, j1, j2, j3, j4 = struct.unpack("<8f", data)
        self.last_pose = {
            "x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "r": round(r, 2),
            "joints": [round(j1, 2), round(j2, 2), round(j3, 2), round(j4, 2)],
        }
        return self.last_pose

    def connect(self, port_name: str) -> dict[str, Any]:
        if serial is None:
            raise DobotProtocolError("pyserial is missing. Run Run-RoboGenius.ps1 to install the app dependency.")
        if self.job_thread and self.job_thread.is_alive():
            raise DobotProtocolError("Wait for the current job to stop before reconnecting")
        self.disconnect()
        try:
            self.port = serial.Serial(port_name, BAUD_RATE, timeout=0.18, write_timeout=2)
            self.port_name = port_name
            time.sleep(0.15)
            self.port.reset_input_buffer()
            pose = self._get_pose()  # Read-only handshake. No motion command is issued.
            self.job = {"state": "idle", "progress": 0, "total": 0, "message": "Connected · pose read"}
            return pose
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        with self.lock:
            if self.port is not None:
                try:
                    self.port.close()
                finally:
                    self.port = None
                    self.port_name = None

    @staticmethod
    def _mapped_strokes(strokes: list[list[dict[str, float]]], settings: dict[str, Any]) -> list[list[tuple[float, float]]]:
        center_x = float(settings["centerX"])
        center_y = float(settings["centerY"])
        paper_size = float(settings["paperSize"])
        half = paper_size / 2.0
        if not 80 <= center_x <= 300 or not -120 <= center_y <= 120:
            raise DobotProtocolError("Paper center is outside the configured safe range")
        if not 20 <= paper_size <= PAPER_MAX_MM:
            raise DobotProtocolError(f"Paper size must be between 20 and {PAPER_MAX_MM:g} mm")
        if not -5 <= float(settings["penDownZ"]) <= 10 or not 2 <= float(settings["penUpZ"]) <= 40:
            raise DobotProtocolError("Pen heights are outside the allowed setup range")
        speed = float(settings["speed"])
        if not 5 <= speed <= 30:
            raise DobotProtocolError("Physical plotting speed must be between 5% and 30%")
        if len(strokes) > 150:
            raise DobotProtocolError("The SVG has too many paths; reduce detail before drawing")

        mapped: list[list[tuple[float, float]]] = []
        for stroke in strokes:
            if not isinstance(stroke, list) or len(stroke) < 2 or len(stroke) > 12000:
                raise DobotProtocolError("Each SVG path must contain 2 to 12000 points")
            path: list[tuple[float, float]] = []
            for point in stroke:
                nx, ny = float(point["x"]), float(point["y"])
                if not math.isfinite(nx) or not math.isfinite(ny) or abs(nx) > 0.83 or abs(ny) > 0.83:
                    raise DobotProtocolError("SVG path leaves the validated paper area")
                # Match the preview mapping: SVG top is the far edge (+X),
                # SVG left-to-right is robot -Y (the operator's right).
                x = center_x - ny * half
                y = center_y - nx * half
                if math.hypot(x, y) > MAX_REACH_MM:
                    raise DobotProtocolError("An SVG point exceeds the Magician's 320 mm reach")
                path.append((x, y))
            mapped.append(path)
        if not mapped:
            raise DobotProtocolError("There are no validated SVG paths to draw")
        return mapped

    def start_draw(self, payload: dict[str, Any]) -> None:
        if self.port is None:
            raise DobotProtocolError("Connect the Dobot before sending a drawing")
        if self.job_thread and self.job_thread.is_alive():
            raise DobotProtocolError("A physical drawing is already running")
        if payload.get("confirmed") is not True or payload.get("validated") is not True:
            raise DobotProtocolError("The drawing needs SVG validation and operator confirmation")
        if not all(payload.get("checks", {}).get(key) is True for key in ("workspace", "pen", "clearance", "operator")):
            raise DobotProtocolError("Complete all four physical setup checks before drawing")
        strokes = self._mapped_strokes(payload.get("strokes", []), payload.get("settings", {}))
        settings = payload["settings"]
        self.cancel_event.clear()
        self.job = {"state": "queued", "progress": 0, "total": sum(len(path) + 3 for path in strokes), "message": "Preparing the physical drawing"}
        self.job_thread = threading.Thread(target=self._draw_worker, args=(strokes, settings), daemon=True)
        self.job_thread.start()

    def _wait_for_index(self, target: int, timeout: float = 45.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.cancel_event.is_set():
                raise DobotProtocolError("Drawing stopped by operator")
            raw = self._request(246, timeout=2.0)
            if len(raw) != 8:
                raise DobotProtocolError("Invalid queue progress response")
            current = struct.unpack("<Q", raw)[0]
            if current >= target:
                return
            time.sleep(0.08)
        raise DobotProtocolError("The robot queue did not advance before timeout")

    def _draw_worker(self, strokes: list[list[tuple[float, float]]], settings: dict[str, Any]) -> None:
        try:
            pose = self._get_pose()
            pen_up = float(settings["penUpZ"])
            pen_down = float(settings["penDownZ"])
            speed = float(settings["speed"])
            current_r = pose["r"]
            with self.lock:
                # Conservative speed ratios and online queued PTP line motion.
                self._request(83, rw=1, queued=0, params=struct.pack("<2f", speed, speed))
                self._request(245, rw=1, queued=0)
                self._request(240, rw=1, queued=0)

            # First raise the pen at the robot's current XY before any lateral
            # travel, so a reconnect never starts with the tip dragging.
            commands: list[tuple[float, float, float]] = [
                (pose["x"], pose["y"], max(pen_up, pose["z"]))
            ]
            for path in strokes:
                commands.append((path[0][0], path[0][1], pen_up))
                commands.append((path[0][0], path[0][1], pen_down))
                commands.extend((x, y, pen_down) for x, y in path[1:])
                commands.append((path[-1][0], path[-1][1], pen_up))

            last_index = 0
            total = len(commands)
            self.job.update(state="drawing", total=total, message="Drawing on the Dobot")
            for index, (x, y, z) in enumerate(commands, start=1):
                if self.cancel_event.is_set():
                    raise DobotProtocolError("Drawing stopped by operator")
                params = struct.pack("<B4f", 2, x, y, z, current_r)  # MOVL_XYZ
                raw = self._request(84, rw=1, queued=1, params=params, timeout=3.0)
                if len(raw) != 8:
                    raise DobotProtocolError("The robot did not accept a queued movement")
                last_index = struct.unpack("<Q", raw)[0]
                self.job.update(progress=index)
                if index % 60 == 0:
                    self._wait_for_index(last_index - 25, timeout=60.0)

            if last_index:
                self._wait_for_index(last_index, timeout=max(60.0, total * 0.8))
            self._request(241, rw=1, queued=0)
            self.job.update(state="complete", progress=total, message="Drawing complete")
        except Exception as exc:
            try:
                with self.lock:
                    if self.port is not None:
                        self._request(242, rw=1, queued=0, timeout=1.5)
            except Exception:
                pass
            self.job.update(state="stopped" if self.cancel_event.is_set() else "error", message=str(exc))

    def stop_job(self) -> None:
        self.cancel_event.set()
        if self.port is not None:
            try:
                with self.lock:
                    self._request(242, rw=1, queued=0, timeout=1.5)
            except Exception:
                pass
        self.job.update(state="stopping", message="Stop requested · use the robot's physical stop if needed")

    def status(self, read_pose: bool = False) -> dict[str, Any]:
        pose = self.last_pose
        if read_pose and self.port is not None and not (self.job_thread and self.job_thread.is_alive()):
            try:
                pose = self._get_pose()
            except Exception as exc:
                self.job.update(state="error", message=f"Pose read failed: {exc}")
        ports = []
        if list_ports is not None:
            ports = [{"device": item.device, "description": item.description} for item in list_ports.comports()]
        return {"connected": self.port is not None, "port": self.port_name, "pose": pose, "ports": ports, "job": dict(self.job), "serviceReady": serial is not None}


LINK = DobotLink()


class RoboGeniusHandler(BaseHTTPRequestHandler):
    server_version = "RoboGeniusLocal/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _json(self, code: int, data: dict[str, Any]) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 12_000_000:
            raise ValueError("Invalid request size")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route == "/api/status":
            self._json(200, LINK.status(read_pose=False))
            return
        if route == "/api/artline/status":
            available = ARTLINE_PYTHON.is_file() and ARTLINE_CHECKPOINT.is_file()
            self._json(200, {"available": available, "model": "ArtLine" if available else None})
            return
        if route == "/api/portrait-sketch/status":
            available = APP_PYTHON.is_file() and PORTRAIT_SKETCH_MODEL.is_file()
            self._json(200, {"available": available, "model": "U²-Net Portrait Sketch" if available else None})
            return
        if route in ("/", "/index.html"):
            try:
                raw = HTML_FILE.read_bytes()
            except OSError:
                self.send_error(404, "Studio HTML file not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._body()
            route = self.path.split("?", 1)[0]
            if route == "/api/artline":
                image_data = str(body.get("image", ""))
                if not image_data.startswith("data:image/png;base64,"):
                    raise ValueError("ArtLine expects a PNG portrait image")
                if not (ARTLINE_PYTHON.is_file() and ARTLINE_CHECKPOINT.is_file() and ARTLINE_RUNNER.is_file()):
                    self._json(503, {"ok": False, "error": "ArtLine is not installed yet. Follow ARTLINE_SETUP.md to install its local model runtime."})
                    return
                encoded = image_data.split(",", 1)[1]
                raw_image = base64.b64decode(encoded, validate=True)
                if len(raw_image) > 8_000_000:
                    raise ValueError("Portrait image is too large for local inference")
                with tempfile.TemporaryDirectory(prefix="robogenius-artline-") as folder:
                    source = Path(folder) / "portrait.png"
                    output = Path(folder) / "line-art.png"
                    source.write_bytes(raw_image)
                    result = subprocess.run(
                        [str(ARTLINE_PYTHON), str(ARTLINE_RUNNER), str(ARTLINE_CHECKPOINT), str(source), str(output)],
                        capture_output=True, text=True, timeout=180, check=False,
                    )
                    if result.returncode != 0 or not output.is_file():
                        detail = (result.stderr or result.stdout or "ArtLine inference failed")[-1200:]
                        self._json(500, {"ok": False, "error": detail})
                        return
                    encoded_result = base64.b64encode(output.read_bytes()).decode("ascii")
                self._json(200, {"ok": True, "image": f"data:image/png;base64,{encoded_result}"})
            elif route == "/api/portrait-sketch":
                image_data = str(body.get("image", ""))
                if not image_data.startswith("data:image/png;base64,"):
                    raise ValueError("Portrait Sketch expects a PNG portrait image")
                if not (APP_PYTHON.is_file() and PORTRAIT_SKETCH_MODEL.is_file() and PORTRAIT_SKETCH_RUNNER.is_file()):
                    self._json(503, {"ok": False, "error": "Portrait Sketch is not installed. Follow PORTRAIT_SKETCH_SETUP.md to install its local model."})
                    return
                raw_image = base64.b64decode(image_data.split(",", 1)[1], validate=True)
                if len(raw_image) > 8_000_000:
                    raise ValueError("Portrait image is too large for local inference")
                with tempfile.TemporaryDirectory(prefix="robogenius-portrait-sketch-") as folder:
                    source = Path(folder) / "portrait.png"
                    output = Path(folder) / "sketch.png"
                    source.write_bytes(raw_image)
                    result = subprocess.run(
                        [str(APP_PYTHON), str(PORTRAIT_SKETCH_RUNNER), str(PORTRAIT_SKETCH_MODEL), str(source), str(output)],
                        capture_output=True, text=True, timeout=180, check=False,
                    )
                    if result.returncode != 0 or not output.is_file():
                        detail = (result.stderr or result.stdout or "Portrait Sketch inference failed")[-1200:]
                        self._json(500, {"ok": False, "error": detail})
                        return
                    encoded_result = base64.b64encode(output.read_bytes()).decode("ascii")
                self._json(200, {"ok": True, "image": f"data:image/png;base64,{encoded_result}"})
            elif route == "/api/connect":
                pose = LINK.connect(str(body.get("port", "COM6")))
                self._json(200, {"ok": True, "pose": pose, "port": LINK.port_name})
            elif route == "/api/disconnect":
                LINK.disconnect()
                self._json(200, {"ok": True})
            elif route == "/api/draw":
                LINK.start_draw(body)
                self._json(202, {"ok": True, "job": LINK.job})
            elif route == "/api/stop":
                LINK.stop_job()
                self._json(200, {"ok": True, "job": LINK.job})
            else:
                self.send_error(404)
        except (ValueError, KeyError, DobotProtocolError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._json(500, {"ok": False, "error": f"Local robot service error: {exc}"})


def main() -> None:
    port = int(os.environ.get("ROBOGENIUS_PORT", "8767"))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), RoboGeniusHandler)
    httpd.daemon_threads = True
    print(f"RoboGenius is ready at http://127.0.0.1:{port}/")
    print("The service only listens on this computer. Connecting reads pose; drawing requires confirmation in the app.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        LINK.stop_job()
        LINK.disconnect()
        httpd.server_close()


if __name__ == "__main__":
    main()
