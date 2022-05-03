import torch

def intersection_over_union(boxes_preds, boxes_labels):
    # boxes_preds has shape (N, 4). N is the number of boxes.
    # boxes_labels has shape (N, 4)
    x1_box1 = boxes_preds[:, 0:1]
    y1_box1 = boxes_preds[:, 1:2]
    x2_box1 = boxes_preds[:, 2:3]
    y2_box1 = boxes_preds[:, 3:4]

    x1_box2 = boxes_labels[:, 0:1]
    y1_box2 = boxes_labels[:, 1:2]
    x2_box2 = boxes_labels[:, 2:3]
    y2_box2 = boxes_labels[:, 3:4]

    x1 = torch.max(x1_box1, x1_box2)
    y1 = torch.max(y1_box1, y1_box2)
    x2 = torch.min(x2_box1, x2_box2)
    y2 = torch.min(y2_box1, y2_box2)

    # clamp is for edge cases where they do not intersect,
    # one of the terms below will be negative if the two
    # boxes do not intersect.
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    area_box1 = abs((x2_box1 - x1_box1) * (y2_box1 - y1_box1))
    area_box2 = abs((x2_box2 - x1_box1) * (y2_box1 - y1_box2))

    union = area_box1 + area_box2 - intersection + 1e-6

    return intersection/union