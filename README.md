# Long-tailed-object-detection-for-drone-based-intelligent-counting

## 安裝與準備（以下指令皆假設當前資料夾為 `HW2_114064558/`）

### 必要條件

環境：

  * **Python 3.11**
  * **CUDA 12.1**
  * **Torch 2.2.0+cu121**
  * **Torchvision 0.17.0+cu121**
  * **Numpy 1.26.4**
  * **ultralytics 8.3.206**
  * **ensemble-boxes 1.0.9**
  * **opencv-python 4.12.0.88**

**安裝依賴套件：**

```bash
pip install -r ./code_114064558/requirements.txt
```
-----
##  預期檔案結構

```bash
hw1_114064558
     |-------- report_114064558.pdf
     |-------- code_114064558  
          |-------- src
                    |-------- convert_to_yolo.py
                    |-------- yolo_v11x.py
                    |-------- yolo_v11l.py
                    |-------- yolo_v11m.py
                    |-------- test_all.py
                    |-------- prepare_gt.py
          |-------- readme.md
          |-------- requirements.txt
     |-------- TAICA_CVPDL_2025_HW2 (下載完Dataset後)
          |-------- CVPDL_hw2
               |-------- CVPDL_hw2
                    |-------- test/
                    |-------- train/
```
## Dataset 下載
```bash
kaggle competitions download -c taica-cvpdl-2025-hw-2
```
-----
##  執行流程（以下指令皆假設當前資料夾為 `HW2_114064558/`）
**若路徑出現錯誤請再換成絕對路徑執行**

### 1\. 生成 Ground Truth 檔 (`prepare_gt.py`)
將原始 train 資料夾中的 ground truth 檔案合併成一個 ground_truth.txt 輸出

```bash
python ./code_114064558/src/prepare_gt.py
```

### 2\. 資料格式轉換 (`convert_to_yolo.py`)

將原始資料集轉換為 Yolo 模型所需的標註格式。

```bash
python ./code_114064558/src/convert_to_yolo.py
```
> 轉換完後得切分好的 train 和 val 資料集以及 mydata.yaml

### 3\. 模型訓練
**調整訓練參數**

找到以下程式碼
```bash
/.conda/envs/cvpdl3.11/lib/python3.11/site-packages/ultralytics/utils/loss.py
```
將class v8DetectionLossc 換成以下程式碼 (更改不同類別的權重)
```bash
class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        self.class_weights = torch.tensor([1.0, 10.0, 7.0, 5.0], device=device)
        print(f"[類別權重] 使用權重: {self.class_weights}")

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss with class weights
        cls_loss = self.bce(pred_scores, target_scores.to(dtype))  # (batch, anchors, num_classes)
        weights = self.class_weights.view(1, 1, -1)  # (1, 1, 4)
        cls_loss = cls_loss * weights
        loss[1] = cls_loss.sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)
```

**使用轉換後的資料集開始訓練 YoloV11 模型。**

step 1: 使用yolo v11x訓練，image_size = 1280，batch_size = 4
```bash
python ./code_114064558/src/yolo_v11x.py
```
step 2: 使用yolo v11l訓練，image_size = 1920，batch_size = 2
```bash
python ./code_114064558/src/yolo_v11l.py
```
step 3: 使用yolo v11m訓練，image_size = 1920，batch_size = 2
```bash
python ./code_114064558/src/yolo_v11m.py
```

訓練完成後，最佳的模型權重會儲存至 `./YOLO_weights/yolo11x(l/m)/weights/best.pt`。

### 3\. 模型測試與驗證 (`test_all.py`)

載入訓練好的權重，對測試集進行最終的效能評估和結果可視化。
> 預測完成，得 **predictions_ENSEMBLE.csv** 和 **output_image**

```bash
python ./code_114064558/src/test_all.py
```

-----

## 預期輸出與結果

資料處理結果如：train, test 資料集 和 mydata.yaml 會儲存至 `./yolo_dataset`

訓練後最佳權重會儲存至 `./YOLO_weights/yolo11x(l/m)/weights/best.pt`

預測結果如：**predictions_ENSEMBLE.csv** ＆ **output_image** 會儲存至 `./TAICA_CVPDL_2025_HW2/result/predictions_ENSEMBLE.csv` 及 `TAICA_CVPDL_2025_HW2/result/exp_ENSEMBLE/vis_images`

| 輸出結果 | 說明 |
| :--- | :---
| **predictions_ENSEMBLE.csv** | 符合kaggle競賽規定繳交格式的預測結果。 |
| **images** | 每張照片圈完Bounding Box的結果。 |
|
