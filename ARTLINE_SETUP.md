# ArtLine for RoboGenius

ArtLine runs as a local image-to-line-art step. RoboGenius then traces that image into editable SVG paths for review and Dobot validation. Photos stay on this computer.

The upstream ArtLine checkpoint is about 356 MB and its published runtime uses Python 3.8, PyTorch 1.6, and FastAI 1.0.61. Install it once from PowerShell in this project folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\Setup-ArtLine.ps1
```

The installer places the isolated inference environment in `.artline-venv` and the checkpoint in `models`. It does not replace the Dobot service's Python environment. Restart `Run-RoboGenius.ps1` after setup.

The original ArtLine project is MIT licensed: https://github.com/vijishmadhavan/ArtLine
