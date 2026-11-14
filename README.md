# AIRA — RoboGenius Dobot Drawing Studio

AIRA is a local-first portrait-to-plotter project for the **Dobot Magician Classic**. It combines a Three.js digital twin with tools that turn a photo into editable drawing paths, let an operator review the result, and—when a Dobot is connected—send an explicitly validated drawing to the robot.

The project includes a main digital-twin studio and a separate AI portrait-to-SVG studio. Both run on your computer. Photos are sent only to a local loopback service for processing; model weights are downloaded from their upstream sources during setup or first use.

> **Project status:** experimental. Start with the digital twin and a disconnected robot. Validate paper size, pen position, workspace clearance, and the SVG before any physical job.

## What it does

- **Photo input:** upload a portrait or capture one from a camera.
- **Portrait preparation:** remove the background, then choose between ArtLine and the local U²-Net Portrait Sketch style.
- **Four-step portrait flow:** inspect the photo, the cleaned cutout, the generated AI sketch, and finally the plotter-ready SVG. The sketch and SVG are separate review steps.
- **SVG review:** convert the approved sketch into plotter strokes, adjust tracing detail, preview and download the SVG, and validate it before it enters the drawing workflow.
- **Dobot digital twin:** inspect the modeled robot and paper, adjust joints and pen state, and view the plot path in 3D.
- **Physical drawing:** connect to a Dobot Magician Classic over USB/serial when ready. Connecting does not start motion. A physical job requires a validated drawing, operator confirmation, and all four setup checks in the app.
- **Model export:** export the digital-twin robot as GLB for use in other 3D tools.
- **Standalone AI portrait studio:** use the additional `svg gen` app for face-aware portrait framing, AI line-art generation, adjustable stroke filtering, and SVG cleanup.

## Choose a studio

| Studio | Entry point | Local address | Best for |
|---|---|---|---|
| Digital Twin & Plotter Studio | `Run-RoboGenius.ps1` | `http://127.0.0.1:8767/` | Dobot view, photo workflow, ArtLine/U²-Net comparison, SVG validation, and guarded hardware control |
| AI Portrait-to-SVG Studio | `svg gen/server.py` | `http://127.0.0.1:8766/` | Portrait-focused local line art, face-aware crop, stroke tuning, and editing before plotting |

The second studio is a separate app with its own HTML page and local API. It does not replace the main digital twin.

## Run the main studio on Windows

### Requirements

- Windows 10 or 11
- Python 3.13 and the Python Launcher (`py`)
- Internet access for Python packages, browser libraries, and model downloads
- A Dobot Magician Classic and its USB/serial driver only if you intend to use physical drawing

No robot is needed to explore the digital twin, upload a photo, or prepare an SVG.

### Install and start

From PowerShell in the project folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\Setup-Portrait-Sketch.ps1
.\Run-RoboGenius.ps1
```

The portrait setup creates `.venv` if needed, installs the app dependencies, and downloads the approximately 176 MB U²-Net model into `models/`. The run script starts the local service. Open the address it prints, normally **http://127.0.0.1:8767/**.

The app loads Three.js, Tailwind, and MediaPipe components from CDNs, so the browser needs internet access when those assets are first loaded. Photo-processing requests stay on the local machine.

### Portrait styles

- **U²-Net Portrait Sketch:** included as the practical local comparison style. Run `Setup-Portrait-Sketch.ps1` to install its LiteRT runtime and download the model. It appears as **U²-Net** in the sketch model picker.
- **ArtLine:** optional legacy model. It uses a separate Python 3.8 environment and downloads a roughly 356 MB checkpoint. To install it, run:

  ```powershell
  powershell -ExecutionPolicy Bypass -File .\Setup-ArtLine.ps1
  ```

  Restart the main studio afterwards. If ArtLine is unavailable, the browser's built-in tracer remains available.

The image-to-SVG pipeline is photo-dependent. Use a well-lit, centered portrait and inspect the actual generated paths; neither model guarantees a clean result for every photo.

### Use the photo-to-drawing workflow

1. Open **AI Portrait Studio** and upload a photo or capture one.
2. Review the cleaned subject after background removal.
3. Wait for the **AI Sketch** step to finish. Inspect the generated image; switch between ArtLine and Portrait Sketch to compare styles. A progress indicator appears while the model is working.
4. Click **Trace sketch to SVG** only after the sketch looks right. Adjust edge sensitivity/detail and trace again if needed.
5. Review and download the vector paths. Select **Validate SVG** only after the paths and paper placement look right.
6. Use the digital twin to check robot and paper alignment. Connect the physical Dobot only when you are ready, then complete all four setup checks and explicitly confirm the job.

## Run the standalone AI portrait-to-SVG studio

This app has independent dependencies and a different image-to-SVG pipeline. Keep its environment separate from the main studio:

```powershell
py -3.12 -m venv .svg-gen-venv
.\.svg-gen-venv\Scripts\python.exe -m pip install --upgrade pip
.\.svg-gen-venv\Scripts\python.exe -m pip install -r ".\svg gen\requirements.txt"
.\.svg-gen-venv\Scripts\python.exe ".\svg gen\server.py"
```

Open **http://127.0.0.1:8766/**. On first launch, the server downloads the ONNX portrait and face-detection models. The `svg gen/README.md` file documents its API, quality settings, and legacy command-line converter.

To use another port:

```powershell
.\.svg-gen-venv\Scripts\python.exe ".\svg gen\server.py" --port 8770
```

Then open `http://127.0.0.1:8770/`.

## How the main app is structured

```text
Browser
  ├─ Three.js robot digital twin and SVG review UI
  ├─ MediaPipe portrait segmentation and built-in SVG tracing
  └─ Local HTTP service (127.0.0.1:8767)
       ├─ ArtLine runner (.artline-venv, optional)
       ├─ U²-Net Portrait Sketch runner (.venv)
       └─ Dobot Magician serial bridge (only after operator connects)
```

The standalone app under `svg gen/` has its own HTTP server and ONNX-based line-art engine. It runs on `127.0.0.1:8766` by default.

## Repository contents

| Path | Purpose |
|---|---|
| `robogenius_dobot_magician_3d_digital_twin_studio.html` | Main Three.js digital-twin and drawing UI |
| `dobot_service.py` | Loopback-only web service and guarded Dobot serial bridge |
| `portrait_sketch_runner.py` | Local LiteRT inference for the U²-Net portrait model |
| `artline_runner.py` | Optional ArtLine inference adapter |
| `Setup-Portrait-Sketch.ps1`, `Setup-ArtLine.ps1` | Model/runtime installers |
| `dobot_magician_classic_digital_twin.glb` | Robot model used by the project |
| `svg gen/` | Standalone AI portrait-to-SVG studio and processing engine |
| `RoboGenius_Dobot_AI_Portrait_Project_Dossier.pdf` | Project dossier |

Downloaded weights and local Python environments are intentionally excluded from Git. The setup scripts and standalone server download the required model files when needed.

## Configuration and troubleshooting

- **Port 8767 is busy:** set a different port before starting the main service, for example `$env:ROBOGENIUS_PORT = '8771'`. Open the address printed by the service. The standalone studio accepts `--port` as shown above.
- **Portrait Sketch is unavailable:** rerun `Setup-Portrait-Sketch.ps1`, confirm that the model download completed, then restart the main service.
- **ArtLine is unavailable:** it is optional; install it with `Setup-ArtLine.ps1`. The first setup downloads its older Python runtime, dependencies, and model weights.
- **Camera does not open:** use the local HTTP address, allow camera access in the browser, and check that another app is not using the camera.
- **No Dobot port appears:** check the USB cable and the manufacturer's serial driver, then reconnect the robot. The app should report a disconnected state until you explicitly connect it.
- **First run of the standalone studio is slow:** it may be downloading its ONNX models and initializing the CPU inference engine.

## Safety and privacy

- Both local services bind to loopback; they are intended for use on this computer.
- Connecting to the Dobot reads its pose and does not start motion.
- The service rejects drawing jobs unless the SVG is validated, the operator confirms the job, and all four physical setup checks are complete.
- Keep the work area clear, check the paper/pen setup, and remain by the robot during movement. Use the physical stop control if motion is unexpected.
- Photos are processed locally. Setup may download model assets from Hugging Face, Dropbox, and other upstream hosts; the photos themselves are not sent to those hosts.

## Models and licensing

- [U²-Net Portrait Sketch LiteRT](https://huggingface.co/litert-community/U2Net-Portrait-Sketch-LiteRT) — Apache-2.0 model card; downloaded by the setup script.
- [ArtLine](https://github.com/vijishmadhavan/ArtLine) — upstream project is MIT licensed; its checkpoint is downloaded separately.
- [Informative Drawings ONNX model](https://huggingface.co/rocca/informative-drawings-line-art-onnx) and [YuNet face detector](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) — downloaded by the standalone app when absent. Check upstream terms before redistribution or commercial use.
- Python packages and browser libraries retain their own licenses.

No top-level license has been selected for AIRA's original source code yet. Publishing this repository does not by itself grant a license to reuse that code.
