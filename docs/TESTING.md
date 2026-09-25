# Testing

## PC unit tests

From `pc/`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-win.txt
pytest -q
```

Current tests cover:

- IoU correctness;
- stable ID under steady motion;
- ID survival across short detector dropouts;
- class-aware matching.

## Instant visual test without a model

```powershell
cd pc
.\run_demo.bat
```

The synthetic demo generates three moving targets, deliberately hides one target briefly, and renders the same tracker/HUD path used by the live app.

## Stress cases to try

1. Fast lateral camera pan.
2. Object exits/re-enters the frame.
3. Two same-class targets cross.
4. Low light / high ISO noise.
5. 4K input while detector remains 640px.
6. Webcam disconnect/reconnect.
7. Android portrait/landscape rotation.
8. Android thermal throttling after 10-20 minutes.

For prolonged Android testing, watch inference milliseconds rather than only the camera-preview frame rate; CameraX can keep Preview smooth while the analyzer intentionally drops frames.
