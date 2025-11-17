from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import numpy as np
import os, csv, cv2
import time 


SCALES_FOR_1280 = [1024, 1280, 1440]
SCALES_FOR_1920 = [1440, 1920, 2400]

MODEL_CONFIGS = [
    ("./YOLO_weights/yolox/weights/best.pt", SCALES_FOR_1280), # 11x @ 1280
    ("./YOLO_weights/yolom/weights/best.pt",           SCALES_FOR_1920), # 11m @ 1920
    ("./YOLO_weights/yolol/weights/best.pt",           SCALES_FOR_1920)  # 11l @ 1920
]
# --------------------

TEST_IMAGES = "./TAICA_CVPDL_2025_HW2/CVPDL_hw2/CVPDL_hw2/test"
OUTPUT_CSV = "./TAICA_CVPDL_2025_HW2/result/predictions_ENSEMBLE.csv"
OUTPUT_IMG_DIR = "./TAICA_CVPDL_2025_HW2/result/exp_ENSEMBLE/vis_images" 

CONFIDENCE_THRESHOLD = 0    
IOU_THRESHOLD = 0.45        
SKIP_BOX_THRESHOLD = 0.05   

print(f"載入 {len(MODEL_CONFIGS)} 個模型...")
models = [YOLO(config[0]) for config in MODEL_CONFIGS]
# --------------------
print(f"{len(models)} 個模型載入完成。")

np.random.seed(42)
color_map = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(100)] 

def predict_with_ms_tta_all_flips(model, img, w, h, skip_box_thr, scales):
    all_boxes_norm = []
    all_scores = []
    all_labels = []

    for tta_mode in range(4):
        current_img = img.copy() 
        if tta_mode == 1: current_img = cv2.flip(img, 1) 
        elif tta_mode == 2: current_img = cv2.flip(img, 0)
        elif tta_mode == 3: current_img = cv2.flip(img, -1)

        for scale in scales:
            res = model.predict(source=current_img, 
                                conf=skip_box_thr, 
                                iou=0.99, 
                                imgsz=scale,
                                verbose=False)[0]
            
            boxes = res.boxes.xyxy.cpu().numpy().copy()
            scores = res.boxes.conf.cpu().numpy()
            labels = res.boxes.cls.cpu().numpy()

            if boxes.shape[0] == 0:
                continue

            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h
            
            if tta_mode == 1: boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
            elif tta_mode == 2: boxes[:, [1, 3]] = 1.0 - boxes[:, [3, 1]]
            elif tta_mode == 3:
                boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
                boxes[:, [1, 3]] = 1.0 - boxes[:, [3, 1]]

            all_boxes_norm.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
    
    return all_boxes_norm, all_scores, all_labels

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

print(f"\nStarting Heterogeneous Ensemble MS-TTA (Models: {len(models)})...")

submission = []
image_files = sorted([os.path.join(TEST_IMAGES, f) for f in os.listdir(TEST_IMAGES) if f.endswith(('.jpg', '.png'))])


for i, image_path in enumerate(image_files):
    start_time = time.time() 
    
    img = cv2.imread(image_path)
    if img is None: 
        print(f"圖片讀取失敗: {image_path}")
        continue
        
    h, w = img.shape[:2]
    base_id = os.path.splitext(os.path.basename(image_path))[0]
    image_id = int(base_id[3:])

    all_models_boxes_lists = []
    all_models_scores_lists = []
    all_models_labels_lists = []

    for model_idx in range(len(models)):
        
        model = models[model_idx]
        model_specific_scales = MODEL_CONFIGS[model_idx][1] 
        
        boxes_list, scores_list, labels_list = predict_with_ms_tta_all_flips(
            model, img, w, h, SKIP_BOX_THRESHOLD, model_specific_scales
        )
        
        all_models_boxes_lists.extend(boxes_list)
        all_models_scores_lists.extend(scores_list)
        all_models_labels_lists.extend(labels_list)

    if not all_models_boxes_lists:
        print(f"圖片 {image_id} 未偵測到任何物體。")
        submission.append([image_id, ""])
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, f"{base_id}_EMPTY.jpg"), img)
        continue

    final_boxes, final_scores, final_labels = weighted_boxes_fusion(
        all_models_boxes_lists, all_models_scores_lists, all_models_labels_lists,
        iou_thr=IOU_THRESHOLD, skip_box_thr=SKIP_BOX_THRESHOLD
    )
    
    pred_strings = []
    
    for (x1n, y1n, x2n, y2n), score, cls in zip(final_boxes, final_scores, final_labels):
        if score < CONFIDENCE_THRESHOLD:
            continue

        x1_f, y1_f, x2_f, y2_f = x1n * w, y1n * h, x2n * w, y2n * h
        w_f, h_f = x2_f - x1_f, y2_f - y1_f
        
        if w_f <= 0 or h_f <= 0: continue 

        x1_i, y1_i, x2_i, y2_i = map(int, [x1_f, y1_f, x2_f, y2_f])
        color = color_map[int(cls) % len(color_map)]

        cv2.rectangle(img, (x1_i, y1_i), (x2_i, y2_i), color, 2)
        label = f"{int(cls)}:{score:.2f}"
        cv2.putText(img, label, (x1_i, max(y1_i - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        pred_strings.extend([
            f"{score:.6f}", f"{x1_f:.2f}", f"{y1_f:.2f}", f"{w_f:.2f}", f"{h_f:.2f}", str(int(cls))
        ])

    out_path = os.path.join(OUTPUT_IMG_DIR, f"{base_id}_ENSEMBLE.jpg")
    cv2.imwrite(out_path, img)

    submission.append([image_id, " ".join(pred_strings)])
    
    end_time = time.time()
    print(f"已處理 {image_id} / {len(image_files)}")

submission.sort(key=lambda x: x[0])
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image_ID", "PredictionString"])
    writer.writerows(submission)

print(f"\nEnsemble 融合結果已儲存：{OUTPUT_CSV}")
print(f"圖像輸出位置：{OUTPUT_IMG_DIR}")