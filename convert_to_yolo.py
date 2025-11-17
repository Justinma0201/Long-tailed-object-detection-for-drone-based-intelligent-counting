import os
import cv2
import random
import yaml
import shutil

input_file = "./ground_truth.txt"
img_dir = "./TAICA_CVPDL_2025_HW2/CVPDL_hw2/CVPDL_hw2/train"
output_base = "./yolo_dataset"


img_ext = ".png"

train_ratio = 0.8

class_names = ["car", "hov", "person", "motorcycle"]

train_img_dir = os.path.join(output_base, "images/train")
val_img_dir = os.path.join(output_base, "images/val")
train_label_dir = os.path.join(output_base, "labels/train")
val_label_dir = os.path.join(output_base, "labels/val")

for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
    os.makedirs(d, exist_ok=True)

with open(input_file, "r") as f:
    lines = f.readlines()

annotations = {}
for line in lines:
    parts = line.strip().split(",")
    if len(parts) != 6:
        print(f"忽略格式不符的行: {line.strip()}")
        continue
    
    img_id, file_class_id, x, y, w, h = map(float, parts)
    img_id = int(img_id)
    
    if img_id not in annotations:
        annotations[img_id] = []
    
    annotations[img_id].append((file_class_id, x, y, w, h))

all_ids = list(annotations.keys())
random.shuffle(all_ids)
split_idx = int(len(all_ids) * train_ratio)
train_ids = all_ids[:split_idx]
val_ids = all_ids[split_idx:]

def process_split(ids, img_dest, label_dest):
    for img_id in ids:
        
        img_name = f"img{img_id:04d}{img_ext}"
        img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"圖片不存在: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"無法讀取圖片: {img_path}")
            continue
        img_h, img_w = img.shape[:2]

        dest_img_path = os.path.join(img_dest, img_name)
        shutil.copy(img_path, dest_img_path)

        label_lines = []
        if img_id not in annotations:
            print(f"圖片 {img_id} 沒有找到任何有效標註。")
            continue
            
        for bbox in annotations[img_id]:
            label_class_id, x, y, w, h = bbox
            
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            
            label_lines.append(f"{int(label_class_id)} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        label_name = f"img{img_id:04d}.txt"
        label_path = os.path.join(label_dest, label_name)
        
        with open(label_path, "w") as f:
            f.writelines(label_lines)

process_split(train_ids, train_img_dir, train_label_dir)
process_split(val_ids, val_img_dir, val_label_dir)

yaml_dict = {
    "path": output_base,
    "train": "images/train",
    "val": "images/val",
    "nc": len(class_names),
    "names": class_names
}

yaml_path = os.path.join(output_base, "mydata.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(yaml_dict, f, sort_keys=False, default_flow_style=False)

print(f"轉換完成！")
print(f"- 訓練集: {len(train_ids)} 張圖片")
print(f"- 驗證集: {len(val_ids)} 張圖片")
print(f"- 類別總數 (nc): {len(class_names)}")
print(f"- 類別名稱 (names): {class_names}")
print(f"- YAML: {yaml_path}")