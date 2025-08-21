import os
import json
import random
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

def get_neighbor_color(img, polygon):
    """获取多边形邻域平均颜色"""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    dilated = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    border = cv2.bitwise_xor(dilated, mask)
    color = img[border == 255]
    if len(color) == 0:
        return (127, 127, 127)
    return tuple(np.mean(color, axis=0).astype(np.uint8).tolist())

def shape_to_polygon(shape):
    """将 polygon 或 rectangle 转换为标准 polygon 坐标"""
    if shape["shape_type"] == "polygon":
        return shape["points"]
    elif shape["shape_type"] == "rectangle":
        (x1, y1), (x2, y2) = shape["points"]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    else:
        return None

def shape_top_y(shape):
    """获取 shape 的最小 y 值"""
    poly = shape_to_polygon(shape)
    if poly:
        return min([p[1] for p in poly])
    else:
        return float('inf')

def calculate_y_threshold(shapes, exclude_row_num=3):
    """计算前 N 行标签的 y 阈值"""
    tops = [shape_top_y(s) for s in shapes if s["shape_type"] in ("polygon", "rectangle")]
    if len(tops) < exclude_row_num:
        return 0
    return sorted(tops)[exclude_row_num - 1]

def get_dominant_color(img):
    """获取整幅图中最常见的颜色（背景色）"""
    reshaped = img.reshape(-1, 3)
    reshaped = reshaped[np.all(reshaped != [0, 0, 0], axis=1)]  # 可选：排除纯黑背景
    colors, counts = np.unique(reshaped, axis=0, return_counts=True)
    dominant_color = colors[np.argmax(counts)]
    return tuple(int(c) for c in dominant_color)

def mask_labelme_dataset(
    input_dir,
    output_dir,
    samples_per_image=5,
    exclude_top_rows=3,
    mask_color=None,
    max_mask_ratio=0.5
):
    os.makedirs(output_dir, exist_ok=True)
    json_paths = sorted(glob(os.path.join(input_dir, "*.json")))

    for json_path in tqdm(json_paths, desc="Processing"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        img_path = os.path.join(input_dir, data['imagePath'])
        if not os.path.exists(img_path) or img_path.endswith("vis.jpg"):
            continue

        base_name = os.path.splitext(os.path.basename(json_path))[0]
        img = cv2.imread(img_path)
        shapes = data['shapes']
        y_threshold = calculate_y_threshold(shapes, exclude_row_num=exclude_top_rows)
        valid_shapes = [
            s for s in shapes if s["shape_type"] in ("polygon", "rectangle")
        ]
        total_polygons = len(valid_shapes)

        dominant_color = get_dominant_color(img)

        for idx in range(samples_per_image):
            img_copy = img.copy()
            json_copy = dict(data)
            json_copy['shapes'] = []
            # 筛选参与遮挡的候选标签
            mask_candidates = [
                s for s in valid_shapes if shape_top_y(s) > y_threshold
            ]
            num_to_mask = min(len(mask_candidates), int(total_polygons * max_mask_ratio))
            masked_shapes = random.sample(mask_candidates, num_to_mask) if num_to_mask > 0 else []
            for shape in shapes:
                if shape["shape_type"] not in ("polygon", "rectangle"):
                    json_copy['shapes'].append(shape)
                    continue

                if shape in masked_shapes:
                    poly = np.array(shape_to_polygon(shape), dtype=np.int32)
                    # fill = mask_color if mask_color else get_neighbor_color(img_copy, poly)
                    fill = mask_color if mask_color else dominant_color
                    cv2.fillPoly(img_copy, [poly], fill)
                    continue  # 从标注中移除
                else:
                    json_copy['shapes'].append(shape)
            img_out_name = f"{base_name}_masked_{idx}.jpg"
            json_out_name = f"{base_name}_masked_{idx}.json"
            cv2.imwrite(os.path.join(output_dir, img_out_name), img_copy)
            json_copy['imagePath'] = img_out_name
            json_copy['imageData'] = None
            with open(os.path.join(output_dir, json_out_name), 'w', encoding='utf-8') as f:
                json.dump(json_copy, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    mask_labelme_dataset(
        input_dir=r"/ssdata/user/gangwei.li/Code/table-tools/datasets/zhaoshang/det",
        output_dir=r"/ssdata/user/gangwei.li/Code/table-tools/datasets/zhaoshang/det-augmented",
        samples_per_image=5,
        exclude_top_rows=5,
        mask_color=None,
        max_mask_ratio=0.1
    )



