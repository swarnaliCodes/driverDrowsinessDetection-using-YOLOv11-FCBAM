from ultralytics import YOLO
import torch

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO("configs/yolov11s_fcbam.yaml")

    model.train(
        data="data/data.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        device=device
    )

if __name__ == "__main__":
    train()
