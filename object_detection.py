"""
Simple Object Detection using TensorFlow Hub and OpenCV
-------------------------------------------------------
Detects objects in an image or webcam feed using SSD MobileNet V2.

Author: [Your Name]
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import argparse
import sys

# -------------------- Load Model --------------------
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

print("Loading model... Please wait.")
try:
    detector = hub.load(MODEL_URL)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# COCO label mapping (partial for demo)
LABELS = {
    1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Motorcycle', 5: 'Airplane',
    6: 'Bus', 7: 'Train', 8: 'Truck', 9: 'Boat', 10: 'Traffic Light'
}

def detect_objects(image):
    """Run object detection on an image and return results."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)
    img_tensor = tf.expand_dims(img_tensor, 0)

    results = detector(img_tensor)
    return {k: v.numpy() for k, v in results.items()}

def draw_boxes(image, boxes, classes, scores, threshold=0.5):
    """Draw bounding boxes and labels on the image."""
    h, w, _ = image.shape
    for i in range(len(scores)):
        if scores[i] > threshold:
            y_min, x_min, y_max, x_max = boxes[i]
            start_point = (int(x_min * w), int(y_min * h))
            end_point = (int(x_max * w), int(y_max * h))
            color = (0, 255, 0)

            # Get class label
            label = LABELS.get(classes[i], 'Unknown')
            label_text = f"{label} {int(scores[i]*100)}%"

            # Draw rectangle and label
            cv2.rectangle(image, start_point, end_point, color, 2)
            cv2.putText(image, label_text, (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def process_image(img_path):
    """Detect objects in a static image."""
    image = cv2.imread(img_path)
    if image is None:
        print("❌ Error: Could not load image. Check the file path.")
        return

    print("Detecting objects in image...")
    result = detect_objects(image)
    draw_boxes(image, result["detection_boxes"],
               result["detection_classes"].astype(int),
               result["detection_scores"])

    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_webcam():
    """Detect objects using webcam feed."""
    print("Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Unable to access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to read frame from webcam.")
            break

        result = detect_objects(frame)
        draw_boxes(frame, result["detection_boxes"],
                   result["detection_classes"].astype(int),
                   result["detection_scores"])

        cv2.imshow("Object Detection (Webcam)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------- Main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection using TensorFlow Hub and OpenCV")
    parser.add_argument("--mode", choices=["image", "webcam"], required=True,
                        help="Choose detection mode: 'image' or 'webcam'")
    parser.add_argument("--image_path", type=str, help="Path to image (required for image mode)")

    args = parser.parse_args()

    if args.mode == "image":
        if not args.image_path:
            print("❌ Error: --image_path is required for image mode.")
            sys.exit(1)
        process_image(args.image_path)
    else:
        process_webcam()
