import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from IPython import display
import sys

# from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
# import torch
# from huggingface_hub import hf_hub_download
# from PIL import Image

import detectron2
import numpy as np
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
sys.path.append('../Mask2Former')
from mask2former import add_maskformer2_config

### Preliminaries ###

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    from sam2.build_sam import build_sam2_camera_predictor

# # Semantic model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
# # we are going to do whole inference, so no resizing of the image
# processor = SegformerImageProcessor(do_resize=False)
# model = SegformerForSemanticSegmentation.from_pretrained(model_name)
# model.to(device)
    

### Helper Functions ###

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_bbox(bbox, ax, marker_size=200):
    tl, br = bbox[0], bbox[1]
    w, h = (br - tl)[0], (br - tl)[1]
    x, y = tl[0], tl[1]
    print(x, y, w, h)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=None, edgecolor="blue", linewidth=2))
    
    
def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]


def run_semantic_model(model, im):
    outputs = model(im)
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1]).get_image()
    # v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    # instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    # v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    # semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
    # print("Panoptic segmentation (top), instance segmentation (middle), semantic segmentation (bottom)")
    # plt.imshow(panoptic_result)
    # plt.show()
    
    panoptics = outputs['panoptic_seg'][1]
    out_mask = outputs['panoptic_seg'][0]
    
    classes = {2: {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
               3: {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
               4: {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"}}
    
    out = out_mask.cpu().numpy()
    
    labels = {}
    pr = []
    for p in panoptics:
        if p['isthing'] == True: # car or motorcycle
            if p['area'] > 400: # filter out small things
                seg = np.where(out == p['id'])
                y_pom = abs(np.min(seg[0]) - np.max(seg[0]))//6
                x_pom = abs(np.min(seg[1]) - np.max(seg[1]))//6
                y = (np.min(seg[0]) + np.max(seg[0])) // 2
                x = (np.min(seg[1]) + np.max(seg[1])) // 2
                if p['category_id'] in labels.keys():
                    labels[p['category_id']].append([[x+x_pom, y-y_pom], [x-x_pom, y+y_pom]])
                else:
                    labels[p['category_id']] = [[x+x_pom, y-y_pom], [x-x_pom, y+y_pom]]
                pr.append([[x+x_pom, y-y_pom], [x-x_pom, y+y_pom]])
    #             plt.imshow(out)
    #             plt.plot(x+x_pom, y-y_pom, marker='v', color='red')
    #             plt.plot(x-x_pom, y+y_pom, marker='v', color='red')
    
    # plt.show()
    
    return pr
                


if __name__=='__main__':
    repo_id = "hf-internal-testing/fixtures_ade20k"
    
    
    sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"



    cap = cv2.VideoCapture("test.mp4")
    ret, frame = cap.read()
    width, height = frame.shape[:2][::-1]
    
    im = frame
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("../Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    model = DefaultPredictor(cfg)
    
    pr = run_semantic_model(model, im)
    
                
    # print('h')
                
    # find category_id - 1 match to id in https://raw.githubusercontent.com/cocodataset/panopticapi/refs/heads/master/panoptic_coco_categories.json
                
    # image = Image.fromarray(frame)
    # pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # with torch.no_grad():
    #     outputs = model(pixel_values)
    #     logits = outputs.logits
        
    # predicted_segmentation_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    # predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
    
    # color_seg = np.zeros((predicted_segmentation_map.shape[0],
    #                   predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

    # palette = np.array(ade_palette())
    # for label, color in enumerate(palette):
    #     if label == 20:
    #         color_seg[predicted_segmentation_map == label, :] = color
    # # Convert to BGR
    # color_seg = color_seg[..., ::-1]

    # # Show image + mask
    # img = np.array(image) * 0.5 + color_seg * 0.5
    # img = img.astype(np.uint8)

    # plt.figure(figsize=(15, 10))
    # plt.imshow(img)
    # plt.show()
    
    prompts = np.array(pr, dtype=np.float32) # list of points we are tracking

    predictors = [[build_sam2_camera_predictor(model_cfg, sam2_checkpoint), prompts, 0, 0]]


    predictors[0][0].load_first_frame(frame)
    if_init = True

    using_point = True # if True, we use point prompt
    # using_box = False # if True, we use point prompt
    # using_mask= False  # if True, we use mask prompt

    ann_frame_idx = 0  # the frame index we interact with
    # ann_obj_id = (1)
    # ann_obj_id2 = (2)

    # using point prompt
    labels = np.array([1], dtype=np.int32)
    # bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(12, 8))
    # plt.title(f"frame {ann_frame_idx}")
    # plt.imshow(frame)

    # if using_point:
    for idx, points in enumerate(prompts):
        for j in range(points.shape[1]):
            _, out_obj_ids, out_mask_logits = predictors[0][0].add_new_prompt(
                frame_idx=ann_frame_idx,
                obj_id=idx,
                points=[points[j]],
                labels=labels,
            )
        # show_points(point, labels, plt.gca())

    # elif using_box:
    #     _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
    #         frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
    #     )
    #     show_bbox(bbox, plt.gca())

    # elif using_mask:
    #     mask_img_path="masks/aquarium/aquarium_mask.png"
    #     mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
    #     mask = mask / 255

    #     _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
    #         frame_idx=ann_frame_idx, obj_id=ann_obj_id, mask=mask
    #     )

    # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    # show_mask((out_mask_logits[1] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[1])

    # plt.show()

    vis_gap = 1

    while True:
        ret, frame = cap.read()
        ann_frame_idx += 1
        if not ret:
            break
        width, height = frame.shape[:2][::-1]
        
        if ann_frame_idx == 20:
            pr = run_semantic_model(model, frame)
            
            new_predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
            new_predictor.load_first_frame(frame)
            prompts = np.array(pr, dtype=np.float32)
            predictors.append([new_predictor, prompts, 20, 0])
            for idx, points in enumerate(prompts):
                for j in range(points.shape[1]):
                    _, out_obj_ids, out_mask_logits = new_predictor.add_new_prompt(
                        frame_idx=ann_frame_idx-20,
                        obj_id=idx,
                        points=[points[j]],
                        labels=labels,
                    )
            # show_points(np.array([[300, 250]], dtype=np.float32), labels, plt.gca())
            
        if ann_frame_idx % vis_gap == 0:
            print(f"frame {ann_frame_idx}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display.clear_output(wait=True)
            plt.figure(figsize=(12, 8))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(frame)

        for idx, (predictor, prompts, starting_frame_idx, frames_no_match) in enumerate(predictors):
            tracks = 0
            out_obj_ids, out_mask_logits = predictor.track(frame)

            if ann_frame_idx % vis_gap == 0:
                for i in range(len(prompts)):
                    if ((out_mask_logits[i] > 0.0).cpu().numpy() == False).all():
                        tracks += 1
                    show_mask(
                        (out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[i]
                    )
            
            # if none of the object being tracked in the predictor were found in this frame, increment the counter for number of frames without matches
            if tracks == len(prompts):
                predictors[idx][3] += 1

            plt.savefig('sam2_out/' + str(ann_frame_idx) + '.png') 
            
        for idx in range(len(predictors)):
            predictor, prompts, starting_frame_idx, frames_no_match = predictors[idx]
            if frames_no_match >= 3:
                predictors.pop(idx)
                print("Popped a predictor from the list.")
                break

    cap.release()

    os.system('ffmpeg -f image2 -i sam2_out/%d.png sam2_out.mp4')