# Local Portrait Sketch style

RoboGenius offers a second local drawing style, **Portrait Sketch**, based on the Apache-2.0 U²-Net portrait sketch model. Photos stay on this computer. The model runs on the same local Python environment as the Dobot service and does not require a GPU.

Install the runtime and download the roughly 176 MB model once:

```powershell
powershell -ExecutionPolicy Bypass -File .\Setup-Portrait-Sketch.ps1
```

The installer stores the model at `models\U2Net_Portrait_Sketch.tflite`. Restart `Run-RoboGenius.ps1` after setup, upload a photo, then choose **Portrait Sketch** in the clean cutout stage to compare it with ArtLine. The result still needs SVG review and explicit validation before it can be sent to the robot.

The local studio runs at `http://127.0.0.1:8767/` by default. Set `ROBOGENIUS_PORT` if that port is already in use.

Model card and license: https://huggingface.co/litert-community/U2Net-Portrait-Sketch-LiteRT

APDrawingGAN++ remains an option for face-only portrait drawings, but it is not bundled here: its published checkpoint archive is about 3.2 GB and its inference workflow requires aligned face landmarks and masks. The smaller U²-Net model is a practical local comparison for full portraits.
