import os
import json
from pycocotools import mask as maskUtils
from tqdm import tqdm
from collections import defaultdict

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
np.random.seed(3)

# load the SAM 2 model and predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)


# 读取 GT.json 文件
with open('val/coco_ins_seg_val.json', 'r') as f:
    gt_data = json.load(f)

# 获取图像列表和注释列表
images = gt_data['images']
annotations = gt_data['annotations']
categories = gt_data['categories']
# Create a mapping from image_id to annotations for faster lookup
annotations_by_image = defaultdict(list)
for ann in annotations:
    annotations_by_image[ann['image_id']].append(ann)
predictions = []


for image_info in tqdm(images):
    image_id = image_info['id']
    image_file = image_info['file_name']
    image_height = image_info['height']
    image_width = image_info['width']
    
    # 加载图像
    image_path = os.path.join('val', image_file)
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # 获取该图像的所有真实框（如果需要）
    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
    if not image_annotations:
        continue  # Skip images with no annotations
    image_boxes = [ann['bbox'] for ann in image_annotations]
    
    # # 将COCO格式的bbox转换为 [x1, y1, x2, y2]
    image_boxes_xyxy = []
    for bbox in image_boxes:
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
        image_boxes_xyxy.append([x1, y1, x2, y2])

    image_boxes_xyxy = np.array(image_boxes_xyxy)
    
    # 使用预测器进行预测
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=image_boxes_xyxy,
        multimask_output=False
    )
    # print(scores)
    # print(image_boxes_xyxy)
       
    # batch_size, num_masks_per_input, H, W = masks.shape
    if masks.ndim == 4:
        batch_size, num_masks_per_input, H, W = masks.shape
    elif masks.ndim == 3:
        num_masks_per_input, H, W = masks.shape
        batch_size = 1  # 因为每个输入只有一个掩码
    else:
        raise ValueError(f"Unexpected masks shape: {masks.shape}")
    

    for batch_idx in range(batch_size):
        if batch_size != 1:
            mask = masks[batch_idx, 0, :, :]
            score = scores[batch_idx, 0]
        else:
            mask = masks[0, :, :]
            score = scores[0]

        # 将掩码转换为 Fortran order
        mask = np.asfortranarray(mask.astype(np.uint8))

        # 转换为 RLE
        rle = maskUtils.encode(mask)

        # 如果 'counts' 是 bytes 类型，则解码为字符串
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')

        # 计算面积和边界框
        area = int(maskUtils.area(rle))
        bbox = maskUtils.toBbox(rle).tolist()  # [x, y, w, h]

        # 获取类别ID，如果有真实标注则使用真实标注的类别，否则默认类别ID为1
        if batch_idx < len(image_annotations):
            category_id = image_annotations[batch_idx]['category_id']
        else:
            category_id = 1  # 默认类别ID

        # 构建预测结果的注释
        prediction = {
            'image_id': image_id,
            'category_id': category_id,
            'segmentation': rle,
            'score': float(score),
            'bbox': bbox
        }

        predictions.append(prediction)
        


with open('bdd100k_val.json', 'w') as f:
    json.dump(predictions, f)
    