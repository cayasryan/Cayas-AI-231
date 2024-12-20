import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# Load the YOLO models
detect_model_path = "best_models/YOLO-detect-v3.pt"
segment_model_path = "best_models/YOLO-segment-v3.pt"

# Set-up CUDA device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# use a specific GPU
os.environ["CUDA_VISIBLE_DEVICES"]="4"

# Load the YOLO models using ultralytics
detect_model = YOLO(detect_model_path, task = 'detect')
segment_model = YOLO(segment_model_path, task = 'segment')

class_names = [
  'bottled_soda',        # ID 1
  'cheese',              # ID 2
  'chocolate',           # ID 3
  'coffee',              # ID 4
  'condensed_milk',      # ID 5
  'cooking_oil',         # ID 6
  'corned_beef',         # ID 7
  'garlic',              # ID 8
  'instant_noodles',     # ID 9
  'ketchup',             # ID 10
  'lemon',               # ID 11
  'boxed_cream',         # ID 12
  'mayonnaise',          # ID 13
  'peanut_butter',       # ID 14
  'pasta',               # ID 15
  'pineapple_juice',     # ID 16
  'crackers',            # ID 17
  'canned_sardines',     # ID 18
  'bottled_shampoo',     # ID 19
  'soap',                # ID 20
  'soy_sauce',           # ID 21
  'toothpaste',          # ID 22
  'canned_tuna',         # ID 23
  'isopropyl_alcohol'    # ID 24
]

# Define the function to process the webcam input for detection
def detection(image, conf_threshold=0.3):
    if image is None or image.size == 0:
        return image, 0.0

    start_time = time.time()
    
    # Convert the image to RGB format
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = detect_model(frame_rgb, conf=conf_threshold)
    
    # Make the image writable
    image = image.copy()
    
    # Postprocess the outputs
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            if conf > conf_threshold:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_names[cls]}: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    return image, fps

# Define the function to process the webcam input for segmentation
def segmentation(image, conf_threshold=0.3):
    if image is None or image.size == 0:
        return image, 0.0

    start_time = time.time()
    
    # Convert the image to RGB format
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = segment_model(frame_rgb, conf=conf_threshold)
    
    # Make the image writable
    image = image.copy()
    
    # Postprocess the outputs
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            if conf > conf_threshold:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_names[cls]}: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        masks = result.masks
        if masks is not None:
            for mask in masks.data:
                mask = mask.cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                color = (0, 0, 255)  # blue color for the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, thickness=cv2.FILLED)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    return image, fps

# Define the function to process the image based on the selected mode
def process_image(image, conf_threshold, mode):
    if mode == "Detection":
        return detection(image, conf_threshold)
    else:
        return segmentation(image, conf_threshold)

css = """.my-group {max-width: 800px !important; max-height: 800px !important;}
         .my-column {display: flex !important; justify-content: center !important; align-items: center !important;}"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        Real-time Detection and Segmentation of Common Grocery Items
        </h1>
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            conf_threshold = gr.Slider(
                label="Confidence Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.30,
            )
            mode = gr.Radio(
                choices=["Detection", "Segmentation"],
                value="Detection",
                label="Mode"
            )
            fps_display = gr.Textbox(label="FPS", interactive=False)
        with gr.Column(scale=3):
            input_img = gr.Image(sources=["webcam"], type="numpy", streaming=True)

    input_img.stream(process_image, [input_img, conf_threshold, mode], [input_img, fps_display], stream_every=0.1)

if __name__ == "__main__":
    demo.launch()