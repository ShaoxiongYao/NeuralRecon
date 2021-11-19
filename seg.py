import glob
from numpy.lib.twodim_base import mask_indices
import torch 
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import cv2

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


if __name__ == "__main__":

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    img_dir_path = 'demo_data/labDynamic/images'
    img_path_lst = glob.glob(img_dir_path + "/*")
    img_path_lst = sorted(img_path_lst)

    for img_path in img_path_lst:
        img = cv2.imread(img_path)
        cv2.imshow("input", img)

        img_tensor = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        with torch.no_grad():
            outputs = model([img_tensor])

        label_names = [COCO_INSTANCE_CATEGORY_NAMES[idx] for idx in outputs[0]['labels']]

        person_idx = (outputs[0]['labels'] == 1).nonzero()

        if person_idx.shape[0] != 0:
            person_idx = person_idx.item()

            mask_np = outputs[0]['masks'].cpu().numpy()
            mask_np = mask_np[person_idx, 0, :, :]

            mask_binary = (mask_np>0.3).nonzero()

            rgb = np.zeros_like(img).astype(np.uint8)

            rgb[mask_binary[0], mask_binary[1], 0] = 255

            cv2.imshow("seg", rgb)
            cv2.waitKey(1)

