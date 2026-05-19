# Models

This project uses pretrained Torch `.t7` fast neural style transfer models that OpenCV can load with `cv2.dnn.readNetFromTorch`.

The included downloader fetches models from Justin Johnson's fast-neural-style model hosting:

```powershell
python scripts/download_models.py --model candy
python scripts/download_models.py --all
```

At least one `.t7` file must be present in this folder before running the CLI or Flask app.
