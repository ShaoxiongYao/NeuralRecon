import cv2
import numpy as np
import torch
import glob
# import pre-trained models
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn,
                                          fasterrcnn_resnet50_fpn)

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

if __name__ == '__main__':
    # (1) initialize model
    # model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, 
    #                                           box_nms_thresh=0.2, 
    #                                           box_score_thresh=0.7)
    model = fasterrcnn_resnet50_fpn(pretrained=True, 
                                    box_nms_thresh=0.2, 
                                    box_score_thresh=0.7)
    model.eval()

    img_dir_path = 'demo_data/labDynamic/images'
    img_path_lst = glob.glob(img_dir_path + "/*")
    img_path_lst = sorted(img_path_lst)

    for img_path in img_path_lst:
        print("image:", img_path)
        img = cv2.imread(img_path)

        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # (2) convert np to torch tensor, .float() -> compatible with model
        img_tensor = torch.tensor(im_rgb/255.0).permute(2, 0, 1).float()

        with torch.no_grad():
            # (3) model forward, input should be list of images
            detections = model([img_tensor])
        bboxes = detections[0]['boxes'] 

        # Note: visualization image must be contiguous
        vis_img = np.ascontiguousarray(img, dtype=np.uint8)
        # or: vis_img = img.copy()

        # (4) add detected bbox and lables
        for idx in range(bboxes.shape[0]):
            print(f"bbox {idx}: {bboxes[idx]}")

            # plot bounding box
            xmin, ymin, xmax, ymax = [int(coord) for coord in bboxes[idx]] 
            vis_img = cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            # plot class label
            class_idx = detections[0]['labels'][idx]
            cv2.putText(vis_img, COCO_INSTANCE_CATEGORY_NAMES[class_idx], (xmin, ymin-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow("Object detection", vis_img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
