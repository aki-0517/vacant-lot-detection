# Land Vacancy Detection Documentation

このドキュメントは、Claude Code にプロジェクトの全体像と詳細な操作手順を伝えるための統合マニュアルです。  
以下の構成で、データ準備から前処理、モデル学習、推論、提出ファイル作成までのすべてをカバーします。

---

## 1. プロジェクト概要

- **プロジェクト名**: Land Vacancy Detection  
- **目的**: 航空・衛星画像から「空き地」を  
  1. **物体検出**（Bounding Box）  
  2. **セマンティックセグメンテーション**（ピクセル単位）  
  の両面で検出し、提出用 JSON ファイルを自動生成する。

---

## 2. ディレクトリ構成

```plaintext
land_vacancy_detection/
├── data/
│   ├── raw/  
│   │   ├── train_bbox_images.zip
│   │   ├── train_bbox_annotations.json
│   │   ├── train_segmentation_images.zip
│   │   ├── train_segmentation_annotations.json
│   │   ├── evaluation_bbox_images.zip
│   │   ├── evaluation_segmentation_images.zip
│   │   └── sample_submission.zip
│   ├── processed/
│   │   ├── bbox/
│   │   └── segmentation/
│   └── submissions/
├── configs/
│   └── config.yaml
├── docs/
│   └── README.md           ← このファイル
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_visualization.ipynb
├── src/
│   ├── data_pipeline/
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── unet.py
│   │   └── build_model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── outputs/
│   ├── logs/
│   └── checkpoints/
├── requirements.txt
├── setup.py
└── README.md
````

---

## 3. データパイプライン

### 3.1 Raw データ構造

* **train\_bbox\_images.zip**：600枚の物体検出用 RGB 画像
* **train\_bbox\_annotations.json**：物体検出用のアノテーション
* **train\_segmentation\_images.zip**：2653枚のセグメンテーション用 RGB 画像
* **train\_segmentation\_annotations.json**：セグメンテーション用アノテーション
* **evaluation\_bbox\_images.zip**：400枚の評価用物体検出画像
* **evaluation\_segmentation\_images.zip**：2500枚の評価用セグメン画像
* **sample\_submission.zip**：提出サンプル（bbox.json + segmentation.json）

### 3.2 前処理手順

1. **ZIP 展開**
2. **画像変換**: TIFF → PNG or JPEG、リサイズ
3. **マスク生成**: セグメンテーション用ポリゴン → バイナリマスク
4. 出力先:

   * `data/processed/bbox/`
   * `data/processed/segmentation/`

```bash
python src/data_pipeline/preprocess.py \
  --input_dir data/raw/ \
  --output_dir data/processed/ \
  --mode both
```

---

## 4. モデル学習

### 4.1 設定ファイル（configs/config.yaml）

```yaml
data:
  processed_dir: data/processed/
model:
  type: unet
  backbone: resnet34
training:
  epochs: 50
  batch_size: 8
  lr: 1e-4
  weight_decay: 1e-5
logging:
  tensorboard_logdir: outputs/logs/
checkpoint:
  dir: outputs/checkpoints/
inference:
  score_threshold: 0.5
  nms_iou_threshold: 0.3
```

### 4.2 学習実行

```bash
python src/train.py --config configs/config.yaml
```

* **ログ**: TensorBoard で確認 → `tensorboard --logdir outputs/logs/`
* **チェックポイント**: `outputs/checkpoints/unet_epochXX.pth`

---

## 5. 評価

```bash
python src/evaluate.py --config configs/config.yaml
```

* **物体検出**: mAP (mean Average Precision)
* **セグメンテーション**: IoU (Intersection over Union)

---

## 6. 推論 & 提出ファイル生成

### 6.1 推論実行

```bash
python src/inference.py \
  --config configs/config.yaml \
  --model_path outputs/checkpoints/unet_epoch50.pth \
  --output_dir data/submissions/
```

* 出力:

  * `data/submissions/bbox.json`
  * `data/submissions/segmentation.json`

### 6.2 提出準備

```bash
cd data/submissions/
zip submission.zip bbox.json segmentation.json
```

* 生成された `submission.zip` をコンペティションページにアップロード

---

## 7. ユーティリティ & 補足

* **utils.py**: JSON 操作、可視化関数、汎用ログ出力
* **notebooks/**:

  * `01_data_inspection.ipynb`：データ構造確認
  * `02_preprocessing.ipynb`：前処理結果の可視化
  * `03_visualization.ipynb`：推論結果のサンプル表示


