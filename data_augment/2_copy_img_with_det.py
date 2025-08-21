import os
import json
import shutil

src_dir = "/ssdata/user/gangwei.li/Code/table-tools/datasets/zhaoshang/line"                # 原图和json目录
dst_dir = "/ssdata/user/gangwei.li/Code/table-tools/datasets/zhaoshang/line-augmented"       # 新生成的增强样本保存目录
os.makedirs(dst_dir, exist_ok=True)

for file in os.listdir(src_dir):
    if file.endswith(".jpg"):
        base_name = os.path.splitext(file)[0]
        json_name = base_name + ".json"
        img_path = os.path.join(src_dir, file)
        json_path = os.path.join(src_dir, json_name)
        if not os.path.exists(json_path):
            print(f"[Warning] JSON not found for: {file}")
            continue

        for i in range(5):
            new_base = f"{base_name}_masked_{i}"
            # 复制并重命名图片
            new_img_name = new_base + ".jpg"
            shutil.copy(img_path, os.path.join(dst_dir, new_img_name))
            # 加载 JSON，更新 imagePath 字段
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["imagePath"] = new_img_name
            data["imageData"] = None
            # 保存 JSON
            new_json_name = new_base + ".json"
            with open(os.path.join(dst_dir, new_json_name), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

