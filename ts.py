import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSortTracker
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from slowfast.models import build_model
from slowfast.config.defaults import get_cfg

# Initialize YOLO model
yolo_model = YOLO('yolov8n.pt')  # Use YOLOv8 model

# Initialize DeepSORT tracker
tracker = DeepSortTracker(max_age=30)

# Load SlowFast model configuration and weights
cfg = get_cfg()
cfg.merge_from_file("PATH_TO_SLOWFAST_CONFIG.yaml")
cfg.NUM_GPUS = 1
model = build_model(cfg)
model.eval()
model.load_state_dict(torch.load("PATH_TO_SLOWFAST_WEIGHTS.pth"))

# Preprocessing for SlowFast input
transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

# Function to predict actions using SlowFast
def predict_action(frames, model):
    with torch.no_grad():
        inputs = torch.stack([transform(frame) for frame in frames]).unsqueeze(0)  # Batch size of 1
        inputs = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        preds = model(inputs)
        action = torch.argmax(preds).item()
        return action

# Open video source (camera or file)
cap = cv2.VideoCapture(0)

# Buffer for SlowFast input frames
frame_buffer = []
BUFFER_SIZE = 32  # SlowFast typically uses 32-frame input

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = yolo_model(frame)
    detections = []

    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        if int(cls) == 0:  # Class 0 is 'person'
            detections.append((int(x1), int(y1), int(x2), int(y2), conf))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        track_id = track.track_id
        l, t, r, b = track.to_tlbr()  # bounding box

        # Extract person crop
        person_crop = frame[int(t):int(b), int(l):int(r)]
        person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        # Add to buffer
        frame_buffer.append(person_crop)
        if len(frame_buffer) > BUFFER_SIZE:
            frame_buffer.pop(0)

        # Predict action if buffer is full
        if len(frame_buffer) == BUFFER_SIZE:
            action_id = predict_action(frame_buffer, model)
            action = f"Action {action_id}"  # Map action_id to actual action names if available
        else:
            action = "Analyzing..."

        # Draw bounding box and label
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id} {action}", (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('YOLO + SlowFast Human Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
