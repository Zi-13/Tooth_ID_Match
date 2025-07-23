import os
import json
import cv2
import numpy as np

contour_dir = 'templates/contours'  # BulidTheLab.py生成的json目录
image_dir = 'images'                # 原图目录
mask_dir = 'masks'                  # 生成的mask保存目录
os.makedirs(mask_dir, exist_ok=True)

for json_file in os.listdir(contour_dir):
    if not json_file.endswith('.json'):
        continue
    with open(os.path.join(contour_dir, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_name = data.get('image_path')
    if img_name:
        if os.path.isabs(img_name) and os.path.exists(img_name):
            img_path = img_name
        else:
            img_path = os.path.join(image_dir, os.path.basename(img_name))
    else:
        img_path = os.path.join(image_dir, os.path.splitext(json_file)[0] + '.png')
    if not os.path.exists(img_path):
        print(f"原图不存在: {img_path}")
        continue
    img = cv2.imread(img_path)
    if img is None:
        print(f"图片读取失败: {img_path}")
        continue
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for contour in data['contours']:
        points = np.array(contour['points'], dtype=np.int32)
        print('原始points:', points.shape, points[:5])  # 打印前5个点
        # 检查points shape
        if points.ndim == 3:
            points = points.reshape(-1, 2)
            print('reshape后points:', points.shape)
        # 检查点是否在图片范围内
        if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= h):
            print('警告：有点超出图片范围！')
        cv2.fillPoly(mask, [points], 1)
    print('mask像素值统计:', np.unique(mask, return_counts=True))
    mask_path = os.path.join(mask_dir, os.path.splitext(json_file)[0] + '_mask.png')
    cv2.imwrite(mask_path, mask * 255)  # 乘255方便肉眼查看
    print(f"生成mask: {mask_path}")

