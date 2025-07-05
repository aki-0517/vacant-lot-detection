# 空き地検出コンペティション（Land Vacancy Detection）

Solafuneが主催する「空き地検出（Land Vacancy Detection）」コンペティションの概要をまとめたドキュメントです。

---

## 1. コンペ概要

* **タイトル**: 空き地検出（Land Vacancy Detection）

### 1.1 コンペの目的

航空写真やその他の地理空間データを用い、未使用または十分に活用されていない土地（空き地）を検出・セグメンテーションする高精度モデルを開発することで、都市計画、不動産投資、持続可能な土地管理に貢献します。


---

## 4. データ概要（Dataset）

### 4.1 使用データ

以下の国内自治体提供の航空画像データを改変して使用（各CC BY 4.0ライセンス）。

* 練馬区 (日本)
* 半田市 (日本)
* 長岡市 (日本)
* 静岡市 (日本)
* 掛川市 (日本)

**ファイル形式**: RGB TIFF, 3バンド (赤=1, 緑=2, 青=3)

### 4.2 アノテーション情報

#### 物体検出タスク (Object Detection)

* **形式**: JSON
* **学習用**: ポリゴン情報をバウンディングボックス形式で保存

#### セグメンテーションタスク (Segmentation)

* **形式**: JSON
* **学習用**: ポリゴン情報をマスク形式で保存

---

## 5. データファイル一覧

| ファイル名                                 | 種類   | 数量   | 容量       | 説明                                   |
| ------------------------------------- | ---- | ---- | -------- | ------------------------------------ |
| train\_bbox\_images.zip               | zip  | 600  | 256.4 MB | 物体検出トレーニング用RGB画像                     |
| train\_bbox\_annotations.json         | json | 1    | 734.4 KB | 物体検出アノテーション                          |
| train\_segmentation\_images.zip       | zip  | 2653 | 23.9 MB  | セグメンテーショントレーニング用RGB画像                |
| train\_segmentation\_annotations.json | json | 1    | 2.1 MB   | セグメンテーションアノテーション                     |
| evaluation\_bbox\_images.zip          | zip  | 400  | 170.8 MB | 物体検出評価用RGB画像                         |
| evaluation\_segmentation\_images.zip  | zip  | 2500 | 24.3 MB  | セグメンテーション評価用RGB画像                    |
| sample\_submission.zip                | zip  | 2    | 83.8 KB  | 提出サンプル（bbox.json, segmentation.json） |

---

## 6. 提出形式（Submission）

* 提出物は以下2ファイルを含むZIP:

  1. `bbox.json` (Object Detection結果)
  2. `segmentation.json` (Segmentation結果)
* フォーマットはトレーニングデータおよびサンプル提出ファイルを参照のこと。

---

## 7. Solafune-Tools

Solafune-Toolsリポジトリでは、内部地理データ作成・管理ツールをOSSで提供中。Planetary ComputerのSTACカタログからSentinel-2画像をダウンロードし、クラウドレスモザイクを生成する機能などを含みます。今後も機能追加予定。

* **GitHub**: [https://github.com/Solafune/Solafune-Tools](https://github.com/Solafune/Solafune-Tools)

### 貢献手順

1. リポジトリをフォーク
2. `solafune_tools/community_tools`配下にツール追加
3. プルリクエストを送信
4. レビュー後統合

詳細はリポジトリのREADMEを参照してください。

