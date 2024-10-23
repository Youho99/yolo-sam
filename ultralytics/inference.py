from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from ultralytics.engine.results import Results
import cv2
import torch


def get_object_features(feat_list, idxs):
    # feat_list would contain feature maps in grid format (N, C, H, W), where each (H,W) is an anchor location.
    # We permute and then flatten the grid so that each grid and its feature vectors
    # correspond with the indexes of the prediction. We also downsample the vector to the smallest one (64).
    obj_feats = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, 64, x.shape[1] // 64).mean(dim=-1) for x in feat_list], dim=0)
    return obj_feats[idxs]


# Combined function
def get_result_with_features(img):
    # Run inference
    img = cv2.imread(img)
    prepped = model.predictor.preprocess([img])
    result = model.predictor.inference(prepped)

    # This would return the NMS output in xywh format and the idxs of the predictions that were retained.
    output, idxs = non_max_suppression(result[-1][0], in_place=False)

    # Get features of every detected objected in the final output.
    obj_feats = get_object_features(result[:3], idxs[0].tolist())

    # Also turn the original inference output into results
    output[0][:, :4] = scale_boxes(prepped.shape[2:], output[0][:, :4], img.shape)
    result = Results(img, path="", names=model.predictor.model.names, boxes=output[0])
    result.feats = obj_feats

    return result






model = YOLO('yolov8n.pt')

_ = model("ultralytics/assets/bus.jpg", save=False, embed=[15, 18, 21, 22])

result_with_feat = get_result_with_features("ultralytics/assets/bus.jpg")

# You can now easily access the box along with the features of that particular box
for box, feat in zip(result_with_feat.boxes.xyxy, result_with_feat.feats):
  # Use box or feat
  print(box)
  print()
  print(feat)
  print()
  print()