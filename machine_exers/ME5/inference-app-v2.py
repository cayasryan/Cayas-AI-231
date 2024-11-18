import gradio as gr
import cv2
import numpy as np
import onnxruntime as ort
from gradio_webrtc import WebRTC
import time



# Load the YOLO models
detect_model_path = "YOLO-detect-v3/train1/weights/best.onnx"
segment_model_path = "YOLO-segment-v3/train1/weights/best.onnx"

# Load the YOLO models using onnxruntime
detect_session = ort.InferenceSession(detect_model_path, providers=['CPUExecutionProvider'])
segment_session = ort.InferenceSession(segment_model_path, providers=['CPUExecutionProvider'])

# Define the function to process the webcam input for detection
def detection(image, conf_threshold=0.3):
    start_time = time.time()
    
    # Preprocess the image
    input_size = (640, 640)  # Adjust based on your model's input size
    frame_resized = cv2.resize(image, input_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb / 255.0
    frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
    frame_input = np.expand_dims(frame_transposed, axis=0).astype(np.float32)

    # Run inference
    input_name = detect_session.get_inputs()[0].name
    output_name = detect_session.get_outputs()[0].name
    outputs = detect_session.run([output_name], {input_name: frame_input})[0]

    # Postprocess the outputs (this will depend on your model's output format)
    # For example, draw bounding boxes on the frame
    for output in outputs:
        x1, y1, x2, y2, conf, cls = output[:6]
        if conf > conf_threshold:  # Confidence threshold
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {int(cls)}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image

# Define the function to process the webcam input for segmentation
def segmentation(image, conf_threshold=0.3):
    start_time = time.time()
    
    # Preprocess the image
    input_size = (640, 640)  # Adjust based on your model's input size
    frame_resized = cv2.resize(image, input_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb / 255.0
    frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
    frame_input = np.expand_dims(frame_transposed, axis=0).astype(np.float32)

    # Run inference
    input_name = segment_session.get_inputs()[0].name
    output_name = segment_session.get_outputs()[0].name
    outputs = segment_session.run([output_name], {input_name: frame_input})[0]

    # Postprocess the outputs (this will depend on your model's output format)
    # For example, draw segmentation masks on the frame
    for output in outputs:
        x1, y1, x2, y2, conf, cls = output[:6]
        if conf > conf_threshold:  # Confidence threshold
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {int(cls)}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image

# Define the function to process the image based on the selected mode
def process_image(image, conf_threshold, mode):
    if mode == "Detection":
        return detection(image, conf_threshold)
    else:
        return segmentation(image, conf_threshold)

css = """.my-group {max-width: 600px !important; max-height: 600px !important;}
         .my-column {display: flex !important; justify-content: center !important; align-items: center !important;}"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        Object Detection and Segmentation
        </h1>
        """
    )
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            image = WebRTC(label="Stream")
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

        image.stream(
            fn=process_image, inputs=[image, conf_threshold, mode], outputs=[image], time_limit=10
        )

if __name__ == "__main__":
    demo.launch()