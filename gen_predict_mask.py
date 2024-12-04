import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import cv2
import os

# 加载 Ground Truth JSON 文件
try:
    cocoGt = COCO('/home/lch/Downloads/bdd100k_sam2_test/coco_ins_seg_val.json')
    print("Ground Truth JSON file loaded successfully!")
except Exception as e:
    print(f"Error loading Ground Truth JSON file: {e}")

# 加载预测结果 JSON 文件
try:
    cocoDt = cocoGt.loadRes('/home/lch/Downloads/bdd100k_sam2_test/large/bdd100k_val.json')
    print("Result JSON file loaded successfully!")
except Exception as e:
    print(f"Error loading Result JSON file: {e}")

# 创建保存结果的文件夹
output_dir = '/home/lch/Downloads/bdd100k_sam2_test/predict_vis_result/'
os.makedirs(output_dir, exist_ok=True)

# 遍历所有图片，并保存带有 mask 的图片
for image_id in cocoGt.getImgIds():
    # 获取图像信息
    image_data = cocoGt.loadImgs(image_id)[0]
    image_path = os.path.join('/home/lch/Downloads/bdd100k_sam2_test/bdd100k_ins_seg_labels_trainval/bdd100k/val', image_data['file_name'])  # 更新成你的图像路径
    image = io.imread(image_path)

    # 获取预测结果标注
    dt_annIds = cocoDt.getAnnIds(imgIds=image_id)
    dt_anns = cocoDt.loadAnns(dt_annIds)

    # 为每个预测生成 mask 并叠加在图像上
    for ann in dt_anns:
        segmentation = ann['segmentation']
        if isinstance(segmentation, dict) and 'counts' in segmentation:
            mask = cocoDt.annToMask(ann)
        elif isinstance(segmentation, list):
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            for poly in segmentation:
                pts = np.array(poly).reshape(-1, 2)
                pts = np.int32([pts])
                cv2.fillPoly(mask, pts, 1)
        
        # 生成随机颜色并叠加在图像上
        color_mask = [random.randint(0, 255) for _ in range(3)]
        for c in range(3):  # 对每个通道应用颜色
            image[:, :, c] = np.where(mask == 1, 
                                      (0.7 * color_mask[c] + 0.3 * image[:, :, c]).astype(np.uint8), 
                                      image[:, :, c])

    # 保存处理后的图像
    output_path = os.path.join(output_dir, f"masked_{image_data['file_name']}")
    io.imsave(output_path, image)
    print(f"Saved masked image: {output_path}")
