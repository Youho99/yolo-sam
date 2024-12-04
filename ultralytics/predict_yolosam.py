from ultralytics import YOLO
import ultralytics.yolosam.utils as yolosam
from PIL import Image
import torch

model = YOLO("yolov8n.pt")

sam_model = yolosam.initialize_model("facebook/sam2-hiera-tiny")

image_path = "ultralytics/assets/bus.jpg" 
image = Image.open(image_path).convert("RGB") 

results = model(image_path)

new_data = []

# Process each detected box
for i, box in enumerate(results[0].boxes):
    # Get the new bounding box using sam_bbox
    new_bbox_tensor = yolosam.sam_bbox(box.xyxy, image, sam_model).to('cuda')  # Ensure tensor is on GPU
    
    # Ensure that conf and cls are correctly shaped
    conf_tensor = results[0].boxes.conf[i].unsqueeze(0).unsqueeze(1)  # Make sure it's 2D
    cls_tensor = results[0].boxes.cls[i].unsqueeze(0).unsqueeze(1)    # Make sure it's 2D

    # Convert the new bounding box tensor to be 2D
    if new_bbox_tensor.dim() == 1:
        new_bbox_tensor = new_bbox_tensor.unsqueeze(0)  # Make it 2D if necessary

    # Create a new tensor combining the new bbox with the original confidence and class
    new_data_tensor = torch.cat((new_bbox_tensor, conf_tensor, cls_tensor), dim=1)
    
    new_data.append(new_data_tensor)

# Concatenate all new data tensors
new_datas_tensor = torch.cat(new_data, dim=0)  # Use dim=0 to concatenate vertically

# Retrieve the original image shape
orig_shape = results[0].orig_shape

# Recreate the Boxes object with the new boxes and the original shape
new_boxes_object = results[0].boxes.__class__(new_datas_tensor, orig_shape)

# Replace the old instance with the new one
results[0].boxes = new_boxes_object
