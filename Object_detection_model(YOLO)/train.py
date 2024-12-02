from ultralytics import YOLO

if __name__ == '__main__':
    model_yaml = r"E:\PycharmProjects\yolov8-train\ultralytics-main\yolov8n.yaml"
    data_yaml = r"E:\PycharmProjects\yolov8-train\ultralytics-main\data.yaml"
    pre_model = r"E:\PycharmProjects\yolov8-train\yolov8n.pt"
    model = YOLO(model_yaml, task='detect').load(pre_model)
    # build from YAML and transfer weights
    # Train the model
    results = model.train(data=data_yaml, epochs=100, imgsz=640, batch=4, workers=2)