# Local Portrait Sketch style

RoboGenius offers a second local drawing style, **Portrait Sketch**, based on the Apache-2.0 U²-Net portrait sketch model. Photos stay on this computer. The model runs on the same local Python environment as the Dobot service and does not require a GPU.

Install the runtime and download the roughly 176 MB model once:

```powershell
powershell -ExecutionPolicy Bypass -File .\Setup-Portrait-Sketch.ps1
```

The installer stores the model at `models\U2Net_Portrait_Sketch.tflite`. Restart `Run-RoboGenius.ps1` after setup, upload a photo, then choose **Portrait Sketch** in the AI Sketch stage to compare it with ArtLine. Review that generated image before tracing it to SVG. The SVG still needs explicit validation before it can be sent to the robot.

The local studio runs at `http://127.0.0.1:8767/` by default. Set `ROBOGENIUS_PORT` if that port is already in use.

Model card and license: https://huggingface.co/litert-community/U2Net-Portrait-Sketch-LiteRT

APDrawingGAN++ is linked from the sketch model picker for reference, but it is not runnable in this app yet. Its upstream instructions require separately downloaded checkpoints plus aligned face images, landmarks, and masks. U²-Net is the installed local model for portrait sketches; ArtLine is the other integrated option.
