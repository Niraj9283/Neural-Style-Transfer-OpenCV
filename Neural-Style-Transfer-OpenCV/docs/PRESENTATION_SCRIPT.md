# Presentation Script

## Slide 1: Title

Project title: Neural Style Transfer with OpenCV.

Introduce the project as a computer vision application that converts normal images into artistic images using deep learning.

## Slide 2: Motivation

Explain that normal image filters are fixed and simple, but neural networks can learn complex artistic textures, colors, and brush patterns.

## Slide 3: Problem Statement

The goal is to preserve the content of an input image while applying the visual style learned by a pretrained neural network.

## Slide 4: Background

Mention convolutional neural networks, feature extraction, neural style transfer, and fast feed-forward style transfer.

## Slide 5: Proposed System

Show that the project has an input image, preprocessing, OpenCV DNN inference, postprocessing, Flask display, and Docker deployment.

## Slide 6: Technology Used

Python, OpenCV, NumPy, Flask, Docker, Waitress, and pretrained Torch `.t7` models.

## Slide 7: Implementation

Explain that `cv2.dnn.readNetFromTorch` loads the model and `net.forward()` generates the stylized image. The Flask app sends the selected image and model to the same backend engine used by the CLI.

## Slide 8: Docker Deployment

Run:

```powershell
docker compose up --build
```

Then open:

```text
http://localhost:5000
```

Show the input image, selected model, output image, and inference time.

## Slide 9: CLI Demo

Run:

```powershell
python src\cli.py --image assets\sample_content\campus_scene.png --model models\candy.t7 --output outputs\campus_candy.jpg --width 700 --compare
```

Use this if the evaluator wants a direct command-line demonstration.

## Slide 10: Results and Limitations

Explain that the model produces good stylized results quickly, but each model supports one fixed style and very high resolution images require more processing time.

## Slide 11: Conclusion and Future Scope

Conclude that the project demonstrates practical deep learning inference with OpenCV and containerized deployment with Docker. Future enhancements can include video style transfer, GPU support, and custom model training.

## Short Viva Explanation

"This project implements fast neural style transfer using OpenCV. The content image is first resized and converted into a blob. A pretrained Torch model is loaded using OpenCV's DNN module. After a forward pass, the output tensor is converted back into an image and saved. The project includes a Dockerized Flask web app and a command-line interface for demonstration."
