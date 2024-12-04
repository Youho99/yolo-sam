import zipfile
import json
import os
import cv2
import torch
import numpy as np
from copy import deepcopy
import shutil

from sam2.sam2_image_predictor import SAM2ImagePredictor


def initialize_model(selected_model):
    return SAM2ImagePredictor.from_pretrained(selected_model)


def sam_bbox(bbox, image, model):
    xyxy = bbox.cpu().numpy()  # Convertir en numpy car SAM attend un tableau numpy

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.set_image(image)
        masks, _, _ = model.predict(point_coords=None, point_labels=None, box=xyxy, multimask_output=False)

    # Calcul des nouvelles coordonnées de la bbox à partir des masques
    rows, cols = np.where(masks[0])
    x_min = np.min(cols)
    y_min = np.min(rows)
    x_max = np.max(cols)
    y_max = np.max(rows)

    # Nouvelle bbox au format xyxy
    new_bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]

    # Créer un tensor pour la nouvelle bbox sur le même device
    new_bbox_tensor = torch.tensor(new_bbox, device=bbox.device).unsqueeze(0)

    return new_bbox_tensor


def process_annotations(annotation_file, path_dir_images, model):
    with open(annotation_file, "r") as f:
        json_annotation = json.load(f)

    images_dict = {image["id"]: image for image in json_annotation["images"]}
    json_annotation_modified = deepcopy(json_annotation)

    for annotation in json_annotation_modified["annotations"]:
        if "sam_bbox" in annotation and annotation["sam_bbox"] == 1:
            continue
        
        image_id = annotation["image_id"]
        image_name = images_dict.get(image_id)["file_name"]
        image = cv2.cvtColor(cv2.imread(os.path.join(path_dir_images, image_name)), cv2.COLOR_BGR2RGB)
        
        bbox = annotation["bbox"]
        new_bbox, new_area = sam_bbox(bbox, image, model)
        annotation["bbox"] = new_bbox
        annotation["area"] = new_area
        annotation["sam_bbox"] = 1

    return json_annotation_modified


def bbox_to_xyxy(bbox):
    return np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
