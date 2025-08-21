# Object_detection-code
It is an program that can detect object while placed near the camera


and OpenCV. It supports two modes:

Image Detection: Loads a static image and detects objects.

Webcam Detection: Captures live video from the webcam and detects objects in real time.

The program:

Loads the pre-trained model from TensorFlow Hub.

Converts input frames to tensors and runs inference.

Draws bounding boxes and labels for detected objects with confidence scores.

Handles errors for model loading, invalid image paths, and webcam access.

Uses argparse for command-line arguments instead of manual input.

This design ensures clean, maintainable, and professional-quality code with modular functions and proper error handling.
