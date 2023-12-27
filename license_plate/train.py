from ultralytics import YOLO
import torch

# Load a model
model = YOLO('yolov8n.yaml').load('best_yolov8n.pt')  # build from YAML and transfer weights
DEVICE = "0" if torch.cuda.is_available() else "cpu"


# Use the model
model.train(data="plate_dataset_small/data.yaml", epochs=30, device=DEVICE, optimizer='Adam', pretrained=True, batch=-1)  # train the model

success = model.export(format="onnx")  # export the model to ONNX format