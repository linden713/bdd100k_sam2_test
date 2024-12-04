from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

def calculate_iou_per_category(ground_truth_path, prediction_path):
    # 加载 ground truth 和预测结果
    coco_gt = COCO(ground_truth_path)
    coco_pred = coco_gt.loadRes(prediction_path)
    
    # 创建 COCOeval 对象
    coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
    
    # 运行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # 提取每个类别的 IoU
    category_iou = {}
    num_iou_thresholds = len(coco_eval.params.iouThrs)
    
    for category_id, category_info in coco_gt.cats.items():
        category_name = category_info['name']
        
        # 提取该类别的IoU
        precisions = coco_eval.eval['precision']
        # precision 的维度是 [10, #类别, #IoU阈值, #面积范围, #最大检测数]
        
        category_index = coco_eval.params.catIds.index(category_id)
        
        # 计算每个 IoU 阈值的平均 IoU
        ious = []
        for iou_threshold_index in range(num_iou_thresholds):
            precision_at_threshold = precisions[iou_threshold_index, category_index, :, 0, 2]  # 选择面积范围为所有，检测数不受限
            valid_precisions = precision_at_threshold[precision_at_threshold > -1]  # 过滤掉无效值 (-1)
            if len(valid_precisions) > 0:
                # 计算该 IoU 阈值下的 IoU 平均值
                mean_iou_at_threshold = np.mean(valid_precisions)
                ious.append(mean_iou_at_threshold)
        
        # 计算所有 IoU 阈值的平均值
        if len(ious) > 0:
            average_iou = np.mean(ious)
        else:
            average_iou = 0.0
        
        category_iou[category_name] = average_iou
    
    return category_iou

# 示例：使用文件路径来调用函数
ground_truth_json = '/home/lch/Downloads/bdd100k_sam2_test/coco_ins_seg_val.json'
prediction_json = '/home/lch/Downloads/bdd100k_sam2_test/small/bdd100k_val.json'

iou_per_category = calculate_iou_per_category(ground_truth_json, prediction_json)
print("每个类别的 IoU：")
for category, iou in iou_per_category.items():
    print(f"{category}: {iou:.4f}")



