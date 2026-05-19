# Neural Style Transfer with OpenCV

A final-year CSE style mini-project that applies artistic style transfer to images using OpenCV's DNN module and pretrained fast neural style transfer models. The project includes a Dockerized Flask web application for deployment and a CLI for quick demonstrations.

## Project Objective

The objective of this project is to transform a normal content image into an artistic image while preserving the main structure of the content image. The implementation uses pretrained feed-forward neural networks and OpenCV, so the output is generated in one inference pass instead of running slow iterative optimization.

## Main Features

- OpenCV DNN based neural style transfer.
- Dockerized Flask web app for deployment.
- Command line interface for quick project demos.
- Pretrained `.t7` model downloader.
- Sample input images and generated output folder.
- Project report, architecture notes, presentation script, and viva questions.

## Folder Structure

```text
Neural-Style-Transfer-OpenCV/
  app.py                         Flask web application
  Dockerfile                     Docker image definition
  docker-compose.yml             One-command container deployment
  run_docker.bat                 Windows helper for Docker demo
  run_demo.bat                   One-click CLI demo
  requirements.txt               Python dependencies
  templates/                     Flask HTML templates
  static/                        CSS for the web UI
  src/
    cli.py                       Command line runner
    nst_opencv/
      processor.py               Core OpenCV style-transfer logic
  scripts/
    download_models.py           Downloads pretrained .t7 style models
  assets/
    sample_content/              Sample images for demo
  models/                        Pretrained .t7 models
  outputs/                       Stylized results
  docs/                          Report, viva, and presentation material
```

## Run with Docker

Open PowerShell in this project folder and run:

```powershell
docker compose up --build
```

Then open:

```text
http://localhost:5000
```

On Windows, you can also run:

```powershell
.\run_docker.bat
```

Stop the container with:

```powershell
docker compose down
```

## Docker Image Commands

Build manually:

```powershell
docker build -t neural-style-transfer-opencv .
```

Run manually:

```powershell
docker run --rm -p 5000:5000 neural-style-transfer-opencv
```

## Local Python Setup

Docker is the preferred deployment method. For development without Docker:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`.

## Download More Models

The project already includes `models/candy.t7`. To download another model:

```powershell
python scripts\download_models.py --model mosaic
```

To download all configured styles:

```powershell
python scripts\download_models.py --all
```

## Run from Command Line

```powershell
python src\cli.py --image assets\sample_content\campus_scene.png --model models\candy.t7 --output outputs\campus_candy.jpg --width 700 --compare
```

Or run the Windows CLI demo script:

```powershell
.\run_demo.bat
```

## Methodology

1. Read the content image with OpenCV.
2. Resize the image while preserving aspect ratio.
3. Convert the image into a blob using VGG-style mean subtraction.
4. Load the pretrained Torch model with `cv2.dnn.readNetFromTorch`.
5. Run a forward pass through the neural network.
6. Add back mean pixel values and convert the result to a normal image.
7. Save or display the stylized result through Flask.
8. Deploy the app inside a Docker container.

## Why Docker

Docker packages the application code, dependencies, and runtime environment into a container. This makes the project easier to run on different machines because the evaluator only needs Docker installed instead of manually configuring Python packages.

## Why OpenCV

OpenCV provides a DNN inference module that can load pretrained networks and run them without needing a full deep learning training framework. This makes the project easier to present on normal laptops while still demonstrating a real computer vision and deep learning workflow.

## References

- Gatys, L. A., Ecker, A. S., and Bethge, M. "A Neural Algorithm of Artistic Style."
- Johnson, J., Alahi, A., and Fei-Fei, L. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution."
- Ulyanov, D., Vedaldi, A., and Lempitsky, V. "Instance Normalization: The Missing Ingredient for Fast Stylization."
- Pretrained Torch models: https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/
