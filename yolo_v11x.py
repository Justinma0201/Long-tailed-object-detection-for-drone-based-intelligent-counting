from ultralytics import YOLO

model = YOLO("yolo11x.yaml")  
weight = [1, 10, 7, 5]
train_results = model.train(
    data="./yolo_dataset/mydata.yaml",
    epochs=300,
    imgsz=1280,
    batch=2,
    device=0,
    project="./YOLO_weights",
    name="yolox",
    exist_ok=True,
    flipud=0.5,
    pretrained = False
)

metrics = model.val()
print(metrics)
