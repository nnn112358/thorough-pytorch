

# 深入浅出PyTorch

> \[!IMPORTANT]
> [オンライン閲覧アドレス](https://datawhalechina.github.io/thorough-pytorch/) | [対応ビデオチュートリアル](https://www.bilibili.com/video/BV1L44y1472Z) | [智海（国家級AI教育プラットフォーム）](https://aiplusx.momodel.cn/classroom/class/664bf5db24cff38ad7d2a20e?activeKey=intro)
>
> ご注意：オンラインドキュメントの更新はリポジトリの更新より遅れます。最新情報は`source`フォルダ内のMarkdownファイルをご覧ください。

---

## 一、プロジェクトの初志

PyTorchは深層学習を利用したデータサイエンス研究の重要なツールであり、柔軟性・可読性・性能の面で大きな利点を持ち、近年では学術界で深層学習アルゴリズムを実装する際に最も広く使われるフレームワークとなっています。

PyTorchの学習は、理論的な蓄積と実践的なトレーニングの両面を必要とする「両手に花」の特徴を持っています。そこで私たちは《深入浅出PyTorch》というコースを開発しました。チーム学習の形式をとり、入門から熟練までPyTorchを使いこなし、自ら深層学習アルゴリズムを実装できるようになることを目指しています。

私たちのビジョンは：チーム学習を通じて、PyTorchの基本知識を基礎から段階的に学び、実際に手を動かして操作への熟練度を高めることです。同時にプロジェクト実践を通じてプログラミング能力を十分に鍛え、PyTorchを使った深層学習の基本的な流れを習得し、実際の問題解決能力を高めることを目指しています。

学習の前提条件としては、Pythonプログラミングができること、ニューラルネットワークを含む機械学習アルゴリズムを理解していること、そして積極的に手を動かす姿勢が求められます。

《深入浅出PyTorch》はシリーズとして全3部構成です。すでに本シリーズの第1部と第2部は公開済みであり、今後も《深入浅出PyTorch（下）》を更新し、実践的な応用により近い事例を提供していく予定です。

---

## 二、内容概要

* **第0章：前提知識（選択学習）**

  * 人工知能の簡単な歴史
  * 関連する評価指標
  * よく使うライブラリの学習
  * Jupyter関連の操作
* **第1章：PyTorchの紹介とインストール**

  * PyTorchの概要
  * PyTorchのインストール
  * PyTorch関連リソースの紹介
* **第2章：PyTorch基礎知識**

  * テンソルとその演算
  * 自動微分の概要
  * 並列計算、CUDAとcuDNNの紹介
* **第3章：PyTorchの主要モジュール**

  * 思考：深層学習の一連の流れを実現するために必要な要素は何か
  * 基本的な設定
  * データの読み込み
  * モデルの構築
  * 損失関数
  * 最適化手法
  * 学習と評価
  * 可視化
* **第4章：PyTorch基礎実戦**

  * 基礎実戦 —— Fashion-MNISTファッション分類
  * 基礎実戦 —— 果物・野菜分類（notebook）
* **第5章：PyTorchモデル定義**

  * モデル定義の方法
  * モジュールブロックを活用して複雑なネットワークを迅速に構築
  * モデルの修正
  * モデルの保存と読み込み
* **第6章：PyTorch応用トレーニングテクニック**

  * カスタム損失関数
  * 学習率の動的調整
  * モデルの微調整 - torchvision
  * モデルの微調整 - timm
  * 半精度トレーニング
  * データ拡張
  * ハイパーパラメータの変更と保存
  * PyTorchモデル定義と応用トレーニングの実践
* **第7章：PyTorch可視化**

  * ネットワーク構造の可視化
  * CNN畳み込み層の可視化
  * TensorBoardを使った学習過程の可視化
  * wandbを使った学習過程の可視化
  * SwanLabを使った学習過程の可視化
* **第8章：PyTorchエコシステムの紹介**

  * 概要
  * 画像 —— torchvision
  * 動画 —— PyTorchVideo
  * テキスト —— torchtext
  * 音声 —— torchaudio
* **第9章：モデルデプロイ**

  * ONNXを使ったデプロイと推論
* **第10章：よくあるネットワークコードの解読（進行中）**

  * コンピュータビジョン

    * 画像分類

      * ResNetコード解読
      * Swin Transformerコード解読
      * Vision Transformerコード解読
      * RNNコード解読
      * LSTMコード解読と実戦
    * 物体検出

      * YOLOシリーズ解読（MMYOLOと共同）
    * 画像分割
  * 自然言語処理

    * RNNコード解読
  * 音声処理
  * 動画処理
  * その他

---

## 三、メンバー紹介

| メンバー | 個人紹介                                | 個人ページ                                                                                |
| ---- | ----------------------------------- | ------------------------------------------------------------------------------------ |
| 牛志康  | DataWhaleメンバー、西安電子科技大学学部生           | [知乎](https://www.zhihu.com/people/obeah-82) / [個人ページ](https://nofish-528.github.io/) |
| 李嘉骐  | DataWhaleメンバー、清華大学大学院生              | [知乎](https://www.zhihu.com/people/li-jia-qi-16-9/posts)                              |
| 刘洋   | DataWhaleメンバー、中国科学院数学・システム科学研究所大学院生 | [知乎](https://www.zhihu.com/people/ming-ren-19-34/asks)                               |
| 陈安东  | DataWhaleメンバー、ハルビン工業大学大学院生          | [個人ページ](https://andongblue.github.io/chenandong.github.io/)                          |

**チュートリアルへの貢献状況（公開済み内容）：**

* 李嘉骐：第3章、第4章、第5章、第6章、第7章、第8章、内容統合
* 牛志康：第1章、第3章、第6章、第7章、第8章、第9章、第10章、ドキュメントデプロイ
* 刘洋：第2章、第3章
* 陈安东：第2章、第3章、第7章

---

## 四、コース編成と対応ビデオ

<details>

一部の章はBilibiliにてライブ講義の録画を視聴可能です（随時更新）：
[https://www.bilibili.com/video/BV1L44y1472Z](https://www.bilibili.com/video/BV1L44y1472Z)

* **コース編成：**
  深入浅出PyTorchは3段階に分かれています：

  1. PyTorch深層学習基礎
  2. PyTorch応用操作
  3. PyTorchケーススタディ

* **使用方法：**
  コース内容はMarkdownまたはJupyter Notebook形式でリポジトリ内に保存されています。理解を深めるために繰り返し読み返すことも大切ですが、最も重要なのは **手を動かして練習すること** です。

* **チーム学習スケジュール：**

  * 第1部：第1章〜第4章（学習期間：10日）
  * 第2部：第5章〜第8章（学習期間：11日）

</details>

---

## 五、貢献方法

<details>

本プロジェクトは`Forking`ワークフローを採用しています。詳細は[Atlassianのドキュメント](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow)をご覧ください。大まかな手順は以下の通りです：

1. GitHubで本リポジトリをFork
2. ForkしたリポジトリをClone
3. `upstream`を設定し、`push`を無効化
4. ブランチを作成して開発（ブランチ名は`lecture{#NO}`、#NOは2桁で指定、例：`lecture07`）
5. PR送信前に元リポジトリと同期し、その後PRを作成

**コマンド例：**

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
# 出力例:
# origin	git@github.com:NoFish-528/thorough-pytorch.git (fetch)
# origin	git@github.com:NoFish-528/thorough-pytorch.git (push)
# upstream	git@github.com:datawhalechina/thorough-pytorch.git (fetch)
# upstream	DISABLE (push)
# 作業開始
git checkout -b lecture07
# 編集してコミットし、push
git push -u origin lecture07
# フォークを最新化
git fetch upstream
git checkout main
git merge upstream/main
# リベースしてforce push
git checkout lecture07
git rebase main
git push -f
```

### Commitメッセージ

コミットメッセージは以下の形式を使用します：
`<type>: <short summary>`

例：

```
<type>: <short summary>
  │            │
  │            └─⫸ 要約は現在形で。先頭は大文字にしない。末尾にピリオドを付けない。
  │
  └─⫸ コミットタイプ: [docs #NO]:others
```

`others`はコース内容に関係しない変更（例：`README.md`の修正や`.gitignore`の調整など）を指します。

</details>

---

## 六、更新予定

<details>

| 内容                   | 更新予定日 | 説明                       |
| -------------------- | ----- | ------------------------ |
| apex                 |       | apexの紹介と使用方法             |
| モデルデプロイ              |       | Flaskを使ったPyTorchモデルのデプロイ |
| TorchScript          |       | TorchScript              |
| 並列学習                 |       | 並列学習                     |
| モデル事前学習 - torchhub   |       | torchhubの紹介と使用方法         |
| 物体検出 - SSD           |       | SSDの紹介と実装                |
| 物体検出 - RCNNシリーズ      |       | Fast-RCNN & Mask-RCNN    |
| 物体検出 - DETR          |       | DETRの実装                  |
| 画像分類 - GoogLeNet     |       | GoogLeNetの紹介と実装          |
| 画像分類 - MobileNetシリーズ |       | MobileNetシリーズの紹介と実装      |
| 画像分類 - GhostNet      |       | GhostNetコード解読            |
| GAN - 手書き数字生成実戦      |       | 数字を生成して可視化               |
| GAN - DCGAN          |       |                          |
| スタイル転送 - StyleGAN    |       |                          |
| 生成ネットワーク - VAE       |       |                          |
| 画像分割 - Deeplabシリーズ   |       | Deeplabシリーズコード解読         |
| NLP - LSTM           |       | LSTMによる感情分析実戦            |
| NLP - Transformer    |       |                          |
| NLP - BERT           |       |                          |
| 動画                   |       | 未定                       |
| 音声                   |       | 未定                       |
| 自作CUDA拡張・演算子         |       |                          |

</details>

---

## 七、謝辞とフィードバック

* DataWhaleメンバーの **叶前坤 @[PureBuckwheat](https://github.com/PureBuckwheat)** と **胡锐锋 @[Relph1119](https://github.com/Relph1119)** に文書の校正でご尽力いただきました。


Made with [contrib.rocks](https://contrib.rocks).

---

---

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知識共有ライセンス" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />
本作品は <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">クリエイティブ・コモンズ 表示 - 非営利 - 継承 4.0 国際ライセンス</a> に基づいて公開されています。


