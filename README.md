# U-KAN: 医用画像セグメンテーションと生成のための強力なバックボーン

:pushpin: これは **U-KAN: 医用画像セグメンテーションと生成のための強力なバックボーン** の公式 PyTorch 実装です。

[[`プロジェクトページ`](https://yes-u-kan.github.io/)] [[`arXiv`](https://arxiv.org/abs/2406.02918)] [[`BibTeX`](#citation)]

<p align="center">
  <img src="./assets/logo_1.png" alt="" width="120" height="120">
</p>

> [**U-KAN: 医用画像セグメンテーションと生成のための強力なバックボーン**](https://arxiv.org/abs/2406.02918)<br>
> [Chenxin Li](https://xggnet.github.io/)<sup>1\*</sup>, [Xinyu Liu](https://xinyuliu-jeffrey.github.io/)<sup>1\*</sup>, [Wuyang Li](https://wymancv.github.io/wuyang.github.io/)<sup>1\*</sup>, [Cheng Wang](https://scholar.google.com/citations?user=AM7gvyUAAAAJ&hl=en)<sup>1\*</sup>, [Hengyu Liu](https://liuhengyu321.github.io/)<sup>1</sup>, [Yifan Liu](https://yifliu3.github.io/)<sup>1</sup>, [Chen Zhen](https://franciszchen.github.io/)<sup>2</sup>, [Yixuan Yuan](https://www.ee.cuhk.edu.hk/~yxyuan/people/people.htm)<sup>1✉</sup><br> <sup>1</sup>香港中文大学, <sup>2</sup>香港人工知能・ロボティクスセンター

私たちは Kolmogorov–Arnold Network（KAN）がビジョンタスクのバックボーンを改善する潜在能力を探求しました。既存の U-Net パイプラインを調査・改良・再設計し、トークン化された中間表現に専用の KAN レイヤーを組み込んだ **U-KAN** を考案しました。厳密な医用画像セグメンテーションのベンチマークにより、U-KAN は計算コストを抑えつつもより高い精度を示すことが確認されました。さらに、拡散モデルにおける U-Net のノイズ予測器の代替として U-KAN を利用し、生成タスク向けモデル構造のバックボーンとしての有効性も示しました。これらの取り組みにより、U-KAN によって医用画像セグメンテーションと生成のための強力なバックボーンを構築できるという新たな展望が得られました。

<div align="center">
    <img width="100%" alt="UKAN overview" src="assets/framework-1.jpg"/>
</div>

## 📰ニュース

**[注意]** ランダムシードは評価指標にとって重要であり、報告された結果は 2981, 6142, 1187 の3つのシードで3回実行した平均です（rolling-UNet に従う）。多くの問題はこれに関連していると考えられます。

**[2024.10]** U-KAN は AAAI-25 に採択されました。

**[2024.6]** Seg_UKAN にいくつかの修正を行い、再現性が向上しました。以前のコードは train.py と archs.py を差し替えるだけで更新可能です。

**[2024.6]** モデルチェックポイントと学習ログを公開しました！

**[2024.6]** コードと論文を公開しました！

## 💡主な特徴
- 新たな KAN を導入し、既存の U-Net パイプラインをより **高精度・高効率・高い解釈性** に改良。
- Segmentation U-KAN は **トークン化された KAN ブロック** を用いて既存の畳み込み設計と互換性を確保。
- Diffusion U-KAN は **改良されたノイズ予測器** として生成タスクや幅広いビジョン設定での可能性を実証。

## 🛠セットアップ

```bash
git clone https://github.com/CUHK-AIM-Group/U-KAN.git
cd U-KAN
conda create -n ukan python=3.10
conda activate ukan
cd Seg_UKAN && pip install -r requirements.txt

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

**Tips A**: 本フレームワークは `pytorch=1.13.0` と `CUDA 11.6` でテストしています。他バージョンでも動作する可能性はありますが保証はしません。

## 📚 データ準備

- **BUSI**: [こちら](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **GLAS**: [こちら](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest)
- **CVC-ClinicDB**: [こちら](https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?e=3&dl=0)

すべての [前処理済みデータセット](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155206760_link_cuhk_edu_hk/ErDlT-t0WoBNlKhBlbYfReYB-iviSCmkNRb1GqZ90oYjJA?e=hrPNWD) も提供しています。  
ダウンロードして `data` ディレクトリに配置すればすぐ使えます。


ディレクトリ構造例:
```
Seg_UKAN
├── inputs
│   ├── busi
│     ├── images
│           ├── malignant (1).png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── malignant (1)_mask.png
|           ├── ...
│   ├── GLAS
│     ├── images
│           ├── 0.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── 0.png
|           ├── ...
│   ├── CVC-ClinicDB
│     ├── images
│           ├── 0.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── 0.png
|           ├── ...
```


## 🔖 セグメンテーション U-KAN の評価

事前学習済みモデルから直接 U-KAN を評価することができます。  
以下は [セグメンテーションモデル動物園](#セグメンテーションモデル動物園) にある **事前学習済みモデル** を使用する簡単な例です。

1. 事前学習済みの重みをダウンロードし、```{args.output_dir}/{args.name}/model.pth``` に配置してください。  
2. 次のスクリプトを実行します。

```bash
cd Seg_UKAN
python val.py --name ${dataset}_UKAN --output_dir [YOUR_OUTPUT_DIR]
```

評価時には `best_model.pth` ファイル（最良IoUでのモデル）が自動的に読み込まれます。

## ⏳ セグメンテーション U-KAN の学習

単一の GPU で簡単に学習することができます。
```--dataset```と  ```--input_size```.を指定してください。

```bash
cd Seg_UKAN
python3 train.py --arch UKAN --dataset {dataset} --input_w {input_size} --input_h {input_size} --name {dataset}_UKAN --data_dir [YOUR_DATA_DIR]
```

例: BUSI データセットを 256×256 の解像度で単一 GPU で学習する場合（inputs ディレクトリ内のデータを使用）:

```bash
cd Seg_UKAN
python3 train.py --arch UKAN --dataset busi --input_w 256 --input_h 256 --name busi_UKAN --data_dir ./inputs
```

詳細については Seg_UKAN/scripts.sh を参照してください。
なお、GLAS データセットは 512×512 の解像度で、他のデータセット（256×256）とは異なります。

**[Quick Update]**
論文中の実験結果を完全に再現するには、シード値 2981, 6142, 1187 を使用してください。
比較したすべての手法は同じシード設定で評価されています。

## 🎪 セグメンテーションモデル動物園

すべての事前学習済みモデルの [チェックポイント](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155206760_link_cuhk_edu_hk/Ej6yZBSIrU5Ds9q-gQdhXqwBbpov5_MaWF483uZHm2lccA?e=rmlHMo) を提供しています。  
以下は公開されている性能とチェックポイントの概要です。  
単一実行の結果と、論文で報告された平均結果が異なる点にご注意ください。

| メソッド  | データセット | IoU   | F1    | チェックポイント                                                                                                                              |
| --------- | ------------ | ----- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Seg U-KAN | BUSI         | 65.26 | 78.75 | [リンク](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155206760_link_cuhk_edu_hk/EjktWkXytkZEgN3EzN2sJKIBfHCeEnJnCnazC68pWCy7kQ?e=4JBLIc) |
| Seg U-KAN | GLAS         | 87.51 | 93.33 | [リンク](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155206760_link_cuhk_edu_hk/EunQ9KRf6n1AqCJ40FWZF-QB25GMOoF7hoIwU15fefqFbw?e=m7kXwe) |
| Seg U-KAN | CVC-ClinicDB | 85.61 | 92.19 | [リンク](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155206760_link_cuhk_edu_hk/Ekhb3PEmwZZMumSG69wPRRQBymYIi0PFNuLJcVNmmK1fjA?e=5XzVSi) |

パラメータ `--no_kan` は、KAN レイヤーを MLP レイヤーに置き換えたベースラインモデルを意味します。  
性能にばらつきが見られる場合もありますが、複数回実行した平均結果でより明確な傾向がわかります。

| メソッド             | レイヤータイプ | IoU   | F1    | チェックポイント                                                                                                                              |
| -------------------- | -------------- | ----- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Seg U-KAN (--no_kan) | MLP Layer      | 63.49 | 77.07 | [リンク](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155206760_link_cuhk_edu_hk/EmEH_qokqIFNtP59yU7vY_4Bq4Yc424zuYufwaJuiAGKiw?e=IJ3clx) |
| Seg U-KAN            | KAN Layer      | 65.26 | 78.75 | [リンク](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155206760_link_cuhk_edu_hk/EjktWkXytkZEgN3EzN2sJKIBfHCeEnJnCnazC68pWCy7kQ?e=4JBLIc) |

## 🎇 Diffusion U-KAN による医用画像生成

詳しくは [Diffusion_UKAN](./Diffusion_UKAN/README.md) を参照してください。

## 🛒 TODO リスト

- [X] Seg U-KAN のコード公開  
- [X] Diffusion U-KAN のコード公開  
- [X] 事前学習済みチェックポイント公開

## 🎈 謝辞

以下のプロジェクトに深く感謝します。

- [CKAN](https://github.com/AntonioTepsich/Convolutional-KANs)

## 📜 引用

```bibtex
@article{li2024ukan,
  title={U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation},
  author={Li, Chenxin and Liu, Xinyu and Li, Wuyang and Wang, Cheng and Liu, Hengyu and Yuan, Yixuan},
  journal={arXiv preprint arXiv:2406.02918},
  year={2024}
}