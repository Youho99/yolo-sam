
from ultralytics import settings, YOLO
import logging
import mlflow
import yaml
import os

if __name__ == '__main__':

    settings.update({"mlflow": True})

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

    print("Start...")

    # Load training configuration from YAML file
    with open('train.yaml', 'r') as file:
        data = yaml.safe_load(file)

    # Set MLflow experiment and tracking URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = data['project']
    os.environ["MLFLOW_TRACKING_URI"] = data['MLFLOW_TRACKING_URI']

    # Load a pre-trained YOLO model
    model = YOLO(data['model'])

    print("Training...")

    # Train the model
    model.train(
        data=data['data'],
        epochs=data['epochs'],
        patience=data['patience'],
        batch=data['batch'],
        imgsz=data['imgsz'],
        save=data['save'],
        cache=data['cache'],
        device=data['device'],
        project=data['project'],
        name=data['name'],
        pretrained=data['pretrained'],
        optimizer=data['optimizer'],
        seed=data['seed'],
        deterministic=data['deterministic'],
        lr0=data['lr0'],
        verbose=data['verbose'],
        val=data['val']
    )

    print("Finished !")