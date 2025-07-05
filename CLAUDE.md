# Claude Code Memory for Land Vacancy Detection

## Project Overview
- **Name**: Land Vacancy Detection
- **Goal**: 航空・衛星画像から空き地（未使用または十分に活用されていない土地）を物体検出（バウンディングボックス）およびセグメンテーションで検出し、提出用JSONを自動生成する。

## Directory Structure
- `data/raw/`：ZIP/JSONの生データ
- `data/processed/`：前処理後の画像・マスクデータ
- `data/submissions/`：最終提出用 `bbox.json` & `segmentation.json`
- `src/`
  - `data_pipeline/`: データ読み込み・前処理・Dataset定義
  - `models/`: U-Net などのモデル定義
  - `train.py`: 学習スクリプト
  - `inference.py`: 推論＆提出ファイル生成
  - `evaluate.py`: IoU, mAP などの評価指標
  - `utils.py`: 共通ユーティリティ
- `configs/config.yaml`: ハイパーパラメータ & パス設定
- `notebooks/`: EDA・可視化用ノートブック
- `outputs/`: TensorBoardログ・チェックポイント

## Workflow
1. **Preprocess**  
   `src/data_pipeline/preprocess.py` を実行し、`data/raw/`→`data/processed/` に画像・マスクを展開・生成する。  
2. **Train**  
   `python src/train.py --config configs/config.yaml`  
3. **Evaluate**  
   `python src/evaluate.py --config configs/config.yaml`  
4. **Infer & Submit**  
   `python src/inference.py --config configs/config.yaml --output data/submissions/`  
   → `bbox.json` と `segmentation.json` が生成される  
5. **ZIP & Upload**  
   `submission.zip` にまとめて提出

## Dependencies
```yaml
# requirements.txt
torch>=2.0
torchvision
opencv-python
pyyaml
numpy
Pillow
tqdm
scikit-learn
````

## Notes

* データはすべて `configs/config.yaml` のパスに従って読み書きする
* 学習済みモデルは `outputs/checkpoints/` に自動保存
* 推論結果はスコア閾値やNMS（検出）・マスク後処理（セグメン）を `config.yaml` で調整可能
