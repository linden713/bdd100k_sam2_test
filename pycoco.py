from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 加载真实值和预测值（真实情况下预测值应该来自模型）
cocoGt = COCO('/home/lch/paper/coco_ins_seg_val.json')
cocoDt = cocoGt.loadRes('/home/lch/paper/tiny/bdd100k_val.json')

# 进行评价
cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# 打印评估结果
for metric, value in zip(cocoEval.stats, cocoEval.stats):
    print(f"{metric}: {value:.3f}")
# 访问并打印 IoU 值
# ious = cocoEval.ious  # 获取所有的 IoU 值
# for (img_id, cat_id), ious_matrix in ious.items():
#     print(f"Image ID: {img_id}, Category ID: {cat_id}")
#     print("IoU Matrix between Ground Truths and Detections:")
#     print(ious_matrix)

