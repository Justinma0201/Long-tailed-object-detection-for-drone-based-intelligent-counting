from ultralytics import YOLO

model = YOLO("yolo11m.yaml")  
weight = [1, 10, 7, 5]
train_results = model.train(
    data="./yolo_dataset/mydata.yaml",
    epochs=180,
    imgsz=1920,
    batch=1,
    device=0,
    project="./YOLO_weights",
    name="yolom",
    exist_ok=True,
    flipud=0.5,
    pretrained = False
)

metrics = model.val()
print(metrics)
