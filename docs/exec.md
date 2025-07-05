# 実行手順書 - 空き地検出コンペティション

このドキュメントは、データ前処理から最終的なコンペティション提出ファイル生成までの完全な実行手順を説明します。

---

## 前提条件

### 1. 環境設定

```bash
# 依存関係のインストール
pip install -r requirements.txt

# プロジェクトのインストール（開発モード）
pip install -e .
```

### 2. データファイルの確認

以下のファイルが `data/raw/` に配置されていることを確認：

- `train_bbox_images.zip` (600枚, 256.4 MB)
- `train_bbox_annotations.json` (734.4 KB)
- `train_segmentation_images.zip` (2653枚, 23.9 MB)
- `train_segmentation_annotations.json` (2.1 MB)
- `evaluation_bbox_images.zip` (400枚, 170.8 MB)
- `evaluation_segmentation_images.zip` (2500枚, 24.3 MB)
- `sample_submission.zip` (83.8 KB)

---

## ステップ 1: データ前処理

### 1.1 前処理実行

```bash
cd /Users/user/Desktop/vacant-lot-detection

# 全データの前処理（物体検出 + セグメンテーション）
python src/data_pipeline/preprocess.py \
  --config configs/config.yaml \
  --mode both
```

### 1.2 前処理結果確認

```bash
# 処理された画像の確認
ls -la data/processed/bbox/images/
ls -la data/processed/segmentation/images/
ls -la data/processed/segmentation/masks/

# 評価用画像の確認
ls -la data/processed/bbox/eval_images/
ls -la data/processed/segmentation/eval_images/
```

**期待される出力:**
- `data/processed/bbox/images/`: 600枚の学習用画像
- `data/processed/segmentation/images/`: 2653枚の学習用画像
- `data/processed/segmentation/masks/`: 2653枚のマスク画像
- `data/processed/bbox/eval_images/`: 400枚の評価用画像
- `data/processed/segmentation/eval_images/`: 2500枚の評価用画像

---

## ステップ 2: モデル学習

### 2.1 学習実行

```bash
# セグメンテーションモデルの学習
python src/train.py --config configs/config.yaml

# 特定のエポック数から再開する場合
python src/train.py --config configs/config.yaml --resume outputs/checkpoints/last.pth
```

### 2.2 学習進捗監視

```bash
# TensorBoardでの監視
tensorboard --logdir outputs/logs/
```

**期待される出力:**
- `outputs/checkpoints/best.pth`: 最良モデル
- `outputs/checkpoints/last.pth`: 最新モデル
- `outputs/logs/`: TensorBoardログ

---

## ステップ 3: モデル評価

### 3.1 評価実行

```bash
# 最良モデルの評価
python src/evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pth \
  --output_dir outputs/evaluation

# 複数閾値での評価
python src/evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pth \
  --output_dir outputs/evaluation \
  --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
```

### 3.2 評価結果確認

```bash
# 評価結果の確認
cat outputs/evaluation/evaluation_summary.json
```

**期待される出力:**
- IoU、Dice、Precision、Recall、F1スコア
- 最適閾値の推奨値
- メトリクス分布のグラフ

---

## ステップ 4: 推論・提出ファイル生成

### 4.1 推論実行

```bash
# 提出ファイル生成
python src/inference.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pth \
  --output_dir data/submissions/ \
  --visualize \
  --num_vis 20
```

### 4.2 提出ファイル確認

```bash
# 生成されたファイルの確認
ls -la data/submissions/

# ファイル内容の確認
head -20 data/submissions/bbox.json
head -20 data/submissions/segmentation.json

# ファイルサイズの確認
du -h data/submissions/*.json
```

**期待される出力:**
- `data/submissions/bbox.json`: 物体検出結果
- `data/submissions/segmentation.json`: セグメンテーション結果
- `data/submissions/visualizations/`: 予測結果の可視化

---

## ステップ 5: 最終提出準備

### 5.1 提出ZIPファイル作成

```bash
cd data/submissions/

# 提出用ZIPファイル作成
zip submission.zip bbox.json segmentation.json

# ZIPファイル内容確認
unzip -l submission.zip

# ファイルサイズ確認
ls -lh submission.zip
```

### 5.2 最終チェック

```bash
# JSONファイルの形式確認
python -m json.tool data/submissions/bbox.json > /dev/null && echo "bbox.json: Valid JSON"
python -m json.tool data/submissions/segmentation.json > /dev/null && echo "segmentation.json: Valid JSON"

# 予測数の確認
echo "Bbox predictions: $(jq length data/submissions/bbox.json)"
echo "Segmentation predictions: $(jq length data/submissions/segmentation.json)"
```

---

## トラブルシューティング

### よくあるエラーと対処法

#### 1. CUDA out of memory

```bash
# バッチサイズを減らす
# configs/config.yaml の batch_size を 8 → 4 → 2 に変更
```

#### 2. Missing dependencies

```bash
# 依存関係の再インストール
pip install -r requirements.txt --upgrade
```

#### 3. データファイルが見つからない

```bash
# データファイルの場所確認
find data/ -name "*.zip" -o -name "*.json"
```

#### 4. 学習が収束しない

```bash
# 学習率を下げる
# configs/config.yaml の lr を 1e-4 → 1e-5 に変更
```

---

## 実行時間の目安

- **前処理**: 5-10分
- **学習（50エポック）**: 2-4時間（GPU使用時）
- **評価**: 10-20分
- **推論**: 15-30分

---

## 最終提出物

1. **submission.zip** - コンペティション提出用ファイル
   - `bbox.json`: 物体検出結果（400画像分）
   - `segmentation.json`: セグメンテーション結果（2500画像分）

2. **評価レポート** - `outputs/evaluation/evaluation_summary.json`

3. **可視化結果** - `data/submissions/visualizations/`

---

## 注意事項

- GPUメモリが不足する場合は、`configs/config.yaml`でバッチサイズを調整
- 学習時間を短縮したい場合は、エポック数を減らす
- より高精度を求める場合は、データ拡張やモデルアーキテクチャを調整
- 提出前に必ずJSONファイルの形式を確認