# Security / privacy design

SpectraTrack is designed to be easy to audit.

- No network client is used by either runtime client.
- No telemetry or analytics SDK is included.
- The Android app requests only `CAMERA`.
- No face recognition, biometric identification, cloud upload, background recording, or remote-control feature is implemented.
- Model files are not downloaded by the app.
- The PC Real-ESRGAN integration never downloads or updates a binary. It executes only the explicit path supplied with `--realesrgan`.
- `subprocess` is used only for that optional, user-selected Real-ESRGAN executable and with `shell=False`.

Before running third-party model weights or binaries, verify their source and SHA-256 where the publisher provides one.

## Important image-quality limitation

Enhancement and neural super-resolution cannot recover information that was never captured. Generative SR can create plausible-looking detail. Do not treat an AI-upscaled crop as forensic evidence of text, faces, number plates, or other fine detail.
