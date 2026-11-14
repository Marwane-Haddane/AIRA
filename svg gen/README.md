# AI Portrait → Plotter SVG — RoboGenius / Dobot

This folder contains a standalone local AI portrait pipeline and studio variant.
Its HTML page is independent of the main digital-twin studio in the repository root.

It turns a photo into a detailed black-and-white line drawing (hair, eyes,
beard, clothing contours) and then into pen-plotter-ready polylines for the
3D Dobot digital twin.

Everything runs on your PC. **No Hugging Face API key is needed** and photos are
not uploaded to any cloud service.

## Model

- **Photo → line art:** [`rocca/informative-drawings-line-art-onnx`](https://huggingface.co/rocca/informative-drawings-line-art-onnx)
  (`informative-drawings.onnx`, ~16 MB, dynamic input size, CPU inference).
- Original demo / source: [`carolineec/informativedrawings`](https://huggingface.co/spaces/carolineec/informativedrawings).
- The model file is already in `models/informative-drawings.onnx`. `server.py`
  downloads it automatically if it is missing.
- Background removal uses `rembg` with `u2net_human_seg` (downloaded once to
  `%USERPROFILE%\.u2net`).
- Face-aware framing uses OpenCV's tiny YuNet detector
  (`models/face_detection_yunet_2023mar.onnx`).

## Install

Use Python 3.11–3.13. From the repository root, create an isolated environment:

```powershell
py -3.12 -m venv .svg-gen-venv
.\.svg-gen-venv\Scripts\python.exe -m pip install --upgrade pip
.\.svg-gen-venv\Scripts\python.exe -m pip install -r "svg gen\requirements.txt"
```

This app uses `.svg-gen-venv` so its image-processing dependencies stay separate from the main app.

## Run the studio

```powershell
.\.svg-gen-venv\Scripts\python.exe "svg gen\server.py"
```

Then open **http://127.0.0.1:8766/**.

The server:

- serves the studio HTML/JS/CSS from this folder;
- exposes `GET /api/health` for the UI status badge;
- exposes `POST /api/lineart` for photo → line-art → strokes.

If port 8766 is already used by an old `python -m http.server`, stop it first
(or run `server.py --port 8770` and open that port).

### Using the studio

1. Upload a photo (or snap a webcam frame).
2. The local AI runs automatically when the server is running. The first run
   loads the ONNX/rembg models, which takes a few seconds.
3. Choose **Fast / Balanced / Maximum** in the *Local AI Line Art* card and
   click **✨ Generate** to re-run at a different detail level.
4. Review the line-art preview in Stage 2 and the plotter strokes in Stage 3.
   Click any stroke to erase it; the **Plot SVG Now** button sends the result
   to the 3D Dobot simulation.
5. If the server is not running, the app falls back to the quick Canny/NMS
   tracer automatically.

## API

```http
POST /api/lineart?size=768&threshold=150&max_strokes=320&rdp_eps=0.9
Content-Type: image/jpeg
Body: raw JPEG/PNG bytes
```

Useful query parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `size` | `768` | Model input size (512–1024). Larger = more detail, slower. |
| `threshold` | `150` | Body/clothing ink threshold (100–230). |
| `texture_threshold` | `112` | Stricter threshold for knit/fabric texture. |
| `head_threshold` | `210` | Detailed face/hair/beard threshold. |
| `line_gamma` | `1.5` | Contrast boost for faint model strokes (0.8–3.0). |
| `min_length` | `18` | Minimum body stroke length in pixels. |
| `head_min_length` | `4` | Minimum face/hair stroke length. |
| `max_strokes` | `320` | Cap on returned polylines. |
| `rdp_eps` | `1.0` | Polyline simplification tolerance. |
| `smoothing` | `2` | Chaikin smoothing passes (0–3); higher = smoother pen paths. |

The response is JSON:

```json
{
  "ok": true,
  "engine": "rocca/informative-drawings-line-art-onnx",
  "strokes": [[[x, y], ...], ...],
  "stroke_count": 288,
  "point_count": 2203,
  "preview_png": "data:image/png;base64,...",
  "seconds": 8.1
}
```

Strokes use the studio's normalized paper coordinates: `x` right, `y` down,
clamped to `[-0.84, 0.84]`.

## Quality notes

- The pipeline crops to the detected face/head and upper chest, so the face
  fills the paper. Pan/zoom in Stage 1 before clicking **Generate** if needed.
- Face, hair and beard use a generous threshold; smooth clothing uses a normal
  threshold; knitted/high-frequency fabric needs much darker ink to survive.
  This is what keeps the shirt from turning into a noisy hatch.
- The engine fades and masks the crop border so the frame edge is not traced
  as a fake contour.
- `Fast` (640 px) is best for quick tests. `Balanced` (768 px) is the normal
  choice. `Maximum` (896 px) gives the smoothest, most detailed hair/beard and
  takes roughly 12–15 s on a typical CPU.
- The Stage 2 preview uses the same crop-edge mask as the vectoriser, so the
  fake frame line at the bottom of the photo is not shown or plotted.
- Review the SVG in Stage 3 before plotting. Pen-up travel is ordered with a
  nearest-neighbour pass; use the **MAX STROKES** cap in the quick Canny
  controls or the AI detail level to keep plot time reasonable.

## Legacy CLI

`portrait_to_svg.py` is the older OpenCV-only CLI. It still works, but the web
studio uses the AI engine above for detailed portraits.

```powershell
.\.svg-gen-venv\Scripts\python.exe "svg gen\portrait_to_svg.py" --input "photo.jpg" --output "portrait.svg" --detail medium
```

## Licensing

Check the licenses of the model repositories and `rembg` weights before
redistributing the models or generated artwork commercially. The web studio
and robot code are part of this project.
