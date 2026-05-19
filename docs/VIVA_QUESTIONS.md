# Viva Questions and Answers

## 1. What is Neural Style Transfer?

Neural Style Transfer is a technique that combines the content of one image with the style of another image or learned style model.

## 2. What is the main objective of this project?

The objective is to generate an artistic version of a content image using OpenCV and pretrained deep learning models.

## 3. Why did you use OpenCV?

OpenCV provides image processing functions and a DNN module that can run pretrained neural networks without needing a full training framework.

## 4. What is a `.t7` file?

A `.t7` file is a Torch model file. In this project it stores a pretrained style transfer network.

## 5. What does `cv2.dnn.readNetFromTorch` do?

It loads a Torch model into OpenCV's DNN module so the model can be used for inference.

## 6. What is the difference between training and inference?

Training learns model weights from data. Inference uses already learned weights to generate an output for new input.

## 7. Does this project train a model?

No. It uses pretrained models and focuses on deployment and inference through OpenCV.

## 8. Why are pretrained models used?

Training style transfer models requires large datasets, GPU resources, and more time. Pretrained models make the project practical for demonstration.

## 9. What is preprocessing?

Preprocessing prepares the input image for the neural network. In this project it includes resizing, blob creation, and mean subtraction.

## 10. What is a blob in OpenCV DNN?

A blob is a formatted input tensor used by OpenCV DNN. It stores image data in the shape expected by the network.

## 11. Why is mean subtraction used?

Mean subtraction normalizes image pixel values according to the preprocessing used when the model was trained.

## 12. What is postprocessing?

Postprocessing converts the neural network output tensor into a normal image by reshaping, adding mean values back, clipping pixel values, and saving the image.

## 13. What is style strength in this project?

Style strength blends the stylized output with the original resized image. A lower value preserves more original image color and structure.

## 14. Which file contains the main algorithm?

The main algorithm is in `src/nst_opencv/processor.py`.

## 15. Which file runs the command line demo?

The command line demo is run through `src/cli.py`.

## 16. Which file runs the web interface?

The web interface is run through `app.py` using Flask. For deployment, it is served inside Docker on port `5000`.

## 17. What are the limitations of this project?

Each model supports one fixed style, high resolution inference is slower, and the system does not train custom styles.

## 18. Can this project work with videos?

The current version is image-based. It can be extended to video by applying style transfer frame by frame.

## 19. Why can fast style transfer run quicker than original neural style transfer?

Fast style transfer uses a trained feed-forward network, so it needs only one forward pass instead of optimizing the image over many iterations.

## 20. What is the role of CNNs in style transfer?

CNNs extract visual features such as edges, textures, shapes, and patterns that help represent content and style.

## 21. What is OpenCV DNN target CPU?

It means the neural network inference runs on the CPU. This is useful because the project can run on common laptops.

## 22. How do you evaluate the output?

The output is evaluated visually by checking whether the content structure is preserved and the selected style is visible.

## 23. How can the project be improved?

It can be improved with video processing, GPU support, batch processing, custom model training, and better quality metrics.

## 24. What happens if the model file is missing?

The app shows an error and the user can run `python scripts/download_models.py --model candy` to download a model.

## 25. What did you learn from this project?

The project demonstrates OpenCV image handling, deep learning inference, model loading, preprocessing, postprocessing, Flask routing, and Docker deployment.

## 26. Why is Docker used in this project?

Docker packages the application, dependencies, and runtime environment together so the project can run consistently on different machines.

## 27. Which command is used to start the Docker deployment?

The command is `docker compose up --build`.

## 28. Which port does the Docker web app use?

The Flask web app runs on port `5000`, so the demo URL is `http://localhost:5000`.
