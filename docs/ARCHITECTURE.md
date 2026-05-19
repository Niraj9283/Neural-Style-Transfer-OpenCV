# Architecture Notes

## High-Level Flow

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant Flask as Flask Web App
    participant Engine as StyleTransferEngine
    participant OpenCV as OpenCV DNN
    participant Disk as Models and Outputs
    participant Docker as Docker Container

    User->>Browser: Open localhost:5000
    Browser->>Flask: Select image, model, width, strength
    Flask->>Engine: Send image and model path
    Engine->>Disk: Load .t7 style model
    Engine->>OpenCV: Create blob and run forward pass
    OpenCV-->>Engine: Return stylized tensor
    Engine->>Engine: Postprocess tensor to image
    Engine-->>Flask: Return stylized image and metrics
    Flask->>Disk: Save output image
    Flask-->>Browser: Display result
    Docker-->>Flask: Provides isolated runtime
```

## Important Design Decisions

- The core logic is in `src/nst_opencv/processor.py` so both CLI and Flask use the same implementation.
- The Flask app is used for the browser-based demonstration.
- Docker is used for deployment so the evaluator can run the same environment on any Docker-supported machine.
- The CLI is useful for reproducible demos and report screenshots.
- Models are stored outside source code in the `models` folder because model files are binary assets.
- The project uses pretrained models because training is outside the realistic scope of a CPU-only student demo.

## Data Flow

```text
Input image
  -> Flask upload or sample selection
  -> OpenCV BGR matrix
  -> resized BGR matrix
  -> DNN blob with mean subtraction
  -> style network output tensor
  -> postprocessed BGR image
  -> saved/displayed stylized output
```

## Deployment Flow

```mermaid
flowchart LR
    A["Project Files"] --> B["Docker Build"]
    B --> C["Python 3.11 Container"]
    C --> D["Install Requirements"]
    D --> E["Waitress Server"]
    E --> F["Flask App on Port 5000"]
    F --> G["Browser Demo"]
```

## Key Files

- `app.py`: Flask user interface and HTTP routes.
- `Dockerfile`: Container image definition.
- `docker-compose.yml`: One-command deployment configuration.
- `src/cli.py`: Command line entry point.
- `src/nst_opencv/processor.py`: Style transfer engine.
- `scripts/download_models.py`: Model downloader.
- `docs/PROJECT_REPORT.md`: Submission report.
- `docs/VIVA_QUESTIONS.md`: Viva preparation.
