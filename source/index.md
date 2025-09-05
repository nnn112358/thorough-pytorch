# やさしく理解する PyTorch
PyTorch は深層学習によるデータサイエンス研究で重要なツールです。柔軟性・可読性・性能に優れ、近年は学術界で最も広く使われる深層学習フレームワークの一つになっています。

PyTorch の学習には、理論の理解と手を動かす実践の両方が欠かせません。私たちはチーム学習の形式で、入門から実装まで段階的に習得できるよう「やさしく理解する PyTorch」コースを用意しました。これにより、各自の深層学習アルゴリズムの実装まで到達することを目指します。

私たちの目標は、チーム学習を通じて PyTorch の基礎から応用までを体系的に身につけ、実践で操作に慣れ、プロジェクト演習でプログラミング力を鍛え、PyTorch による深層学習の基本的な流れを理解し、実問題を解く力を高めることです。

受講の前提は、Python によるプログラミングができること、ニューラルネットワークを含む機械学習の基礎を理解していること、そして積極的に手を動かして学べることです。

本シリーズは全3部構成です。現在は第1部・第2部を公開済みで、今後は実務により近いケーススタディを中心に「後編」を継続的に更新していきます。

```{toctree}
:maxdepth: 2
:caption: 目次
第0章/index
第1章/index
第2章/index
第3章/index
第4章/index
第5章/index
第6章/index
第7章/index
第8章/index
第9章/index
第10章/index
```

## メンバー
| メンバー | 紹介 | 個人ページ |
| --------------- | --------------------------------------------------- | -------------------------------------------------- |
| 牛志康 | DataWhale メンバー，西安電子科技大学 学部生 | [[知乎](https://www.zhihu.com/people/obeah-82)][[個人サイト](https://nofish-528.github.io/)] |
| 李嘉骐 | DataWhale メンバー，清華大学 大学院生 | [[知乎](https://www.zhihu.com/people/li-jia-qi-16-9/posts)] |
| 刘洋 | DataWhale メンバー，中国科学院 数学とシステム科学研究所 大学院生 | [[知乎](https://www.zhihu.com/people/ming-ren-19-34/asks)] |
| 陈安东 | DataWhale メンバー，中央民族大学 大学院生 | [[個人サイト](https://andongblue.github.io/chenandong.github.io/)] |

公開済み内容の貢献状況：

李嘉骐：第3章・第4章・第5章・第6章・第7章・第8章・内容統合

牛志康：第1章・第3章・第6章・第7章・第8章・第9章・第10章・ドキュメント整備

刘洋：第2章・第3章

陈安东：第2章・第3章・第7章

## 4. コース構成と使い方

一部の章はライブ配信のアーカイブ（順次更新）をご覧ください：https://www.bilibili.com/video/BV1L44y1472Z

- コース構成：
  本コースは3段階で構成します。PyTorch の基礎、PyTorch の発展的な操作、PyTorch によるケーススタディ。

- 使い方：
  本コースの教材は本リポジトリに Markdown あるいは Jupyter Notebook 形式で保存しています。理解を深めるために読み込むことに加え、何よりも手を動かして練習することが重要です。

- チーム学習の進め方：
  第1部：第1章〜第4章（学習目安：10日間）

  第2部：第5章〜第8章（学習目安：11日間）
  
## 5. コントリビュートについて

本プロジェクトは `Forking` ワークフローを採用しています。詳細は [Atlassian のドキュメント](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow) を参照してください。

おおまかな手順：

1. GitHub で本リポジトリを Fork する
1. Fork 先の個人リポジトリを Clone する
1. `upstream` のリモートを設定し、`push` を無効化する
1. ブランチで開発する。コースのブランチ名は `lecture{#NO}`（例：`lecture07`）とし、該当する章ディレクトリに対応させる
1. PR 前に元のリポジトリと同期し、その後 PR を作成する

コマンド例：

```shell
# fork
# clone
git clone git@github.com:USERNAME/thorough-pytorch.git
# set upstream
git remote add upstream git@github.com:datawhalechina/thorough-pytorch.git
# disable upstream push
git remote set-url --push upstream DISABLE
# verify
git remote -v
# some sample output:
# origin	git@github.com:NoFish-528/thorough-pytorch.git (fetch)
# origin	git@github.com:NoFish-528/thorough-pytorch.git (push)
# upstream	git@github.com:datawhalechina/thorough-pytorch.git (fetch)
# upstream	DISABLE (push)
# do your work
git checkout -b lecture07
# edit and commit and push your changes
git push -u origin lecture07
# keep your fork up to date
## fetch upstream main and merge with forked main branch
git fetch upstream
git checkout main
git merge upstream/main
## rebase brach and force push
git checkout lecture07
git rebase main
git push -f
```

### コミットメッセージ

コミットメッセージは次の形式にしてください：`<type>: <short summary>`

```
<type>: <short summary>
  │            │
  │            └─⫸ Summary in present tense. Not capitalized. No period at the end.
  │
  └─⫸ Commit Type: lecture{#NO}|others
```

`others` は本コース以外の変更（例：この `README.md` の変更、`.gitignore` の調整など）を指します。

## 6. 更新予定
| 項目 | 更新日 | 内容 |
| :---- | :---- | :----: |
| Visdom 可視化 |  | `Visdom` の使い方 |
| Apex |  | Apex の概要と使用方法 |
| モデルデプロイ |  | Flask による PyTorch モデルのデプロイ |
| TorchScript |  | TorchScript |
| 並列学習 |  | 並列学習 |
| 事前学習モデル - torchhub | 2022.4.16 | torchhub の概要と使い方 |
| 物体検出 - SSD |  | SSD の概要と実装 |
| 物体検出 - RCNN 系列 |  | Fast-RCNN & Mask-RCNN |
| 物体検出 - DETR |  | DETR の実装 |
| 画像分類 - GoogLeNet | 2022.5.11 | GoogLeNet の紹介と実装 |
| 画像分類 - MobileNet 系列 | 2022年4月 | MobileNet 系列の紹介と実装 |
| 画像分類 - GhostNet | 2022年4月 | GhostNet コード解説 |
| 生成対向ネットワーク - 手書き数字生成実践 | 2022.5.25 | 生成と可視化 |
| 生成対向ネットワーク - DCGAN |  |  |
| スタイル変換 - StyleGAN |  |  |
| 生成モデル - VAE |  |  |
| 画像セグメンテーション Deeplab 系列 |  | Deeplab 系列コード解説 |
| 自然言語処理 LSTM |  | LSTM 感情分析の実践 |
| 自然言語処理 Transformer |  |  |
| 自然言語処理 BERT |  |  |
| 動画 |  | 未定 |
| 音声 |  | 未定 |
| カスタム CUDA 拡張と演算子 |  |  |
