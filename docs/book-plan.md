# 64ページ構成案

## 前提

- B5・64ページ
- 表紙、あとがき、奥付を含む
- Python経験者向け
- PyTorchを使用
- Transformerの主要部分は自作
- コードを含む技術同人誌
- 篠澤広とプロデューサーが、LLM自作の「ままならなさ」を観察する

## ページ配分

| ページ | 内容 | 物語・技術上の役割 |
|---:|---|---|
| 1 | 表紙 | タイトル、篠澤広、プロデューサー |
| 2 | 扉・注意書き | 対象読者、前提知識、本書の範囲 |
| 3–4 | 目次・読み方 | 章構成、コードの参照方法 |
| 5–8 | プロローグ「ままならないものを作る」 | 壊れた生成結果から始め、LLM自作の動機を示す |
| 9–14 | 第1章「まずデータを読む」 | テキスト、分割、バッチ、入力と教師 |
| 15–20 | 第2章「言葉を数字にする」 | 文字単位Tokenizer、語彙、encode/decode |
| 21–24 | 第3章前半「言葉をベクトルにする」 | Embedding、位置情報、テンソル形状 |
| 25–31 | 第3章中盤「Attentionを作る」 | Query、Key、Value、softmax、causal mask |
| 32–37 | 第3章後半「Transformer Blockを作る」 | Multi-Head、MLP、Residual、LayerNorm |
| 38–42 | 第4章「学習させる」 | Cross Entropy、optimizer、学習ループ |
| 43–47 | 第5章「数字の上では成功する」 | lossは下がったが、生成結果は不完全という偽の成功 |
| 48–53 | 第6章「文章を生成する」 | greedy、temperature、top-k、生成例 |
| 54–58 | 第7章「ままならなさを観察する」 | データ量、モデルサイズ、学習率、過学習 |
| 59–60 | エピローグ | 完璧な成功ではなく、失敗の理由を理解して終える |
| 61–62 | 付録 | 実行方法、設定値、コード一覧 |
| 63 | あとがき | 制作後記 |
| 64 | 奥付 | クレジット、リポジトリ情報 |

## 第3章の内訳

Transformerは本書の中心なので、17ページを割り当てる。

| ページ | 内容 |
|---:|---|
| 21–22 | token IDからEmbeddingへ |
| 23–24 | 位置情報と `(B, T, C)` |
| 25–26 | Query、Key、Valueの直感 |
| 27–28 | Attention scoreとsoftmax |
| 29 | causal mask |
| 30–31 | Self-AttentionのPyTorch実装 |
| 32–33 | Multi-Head Attention |
| 34 | Feed Forward Network |
| 35 | Residual connection |
| 36 | LayerNormとTransformer Block |
| 37 | logitsと次トークン予測 |

数式、テンソル形状、コードを対応させ、同じ概念を別々の場所で重複説明しない。

## 誌面における分量

| 要素 | 目安 |
|---|---:|
| コード | 14–18ページ相当 |
| 数式・図 | 8–10ページ相当 |
| 会話・本文 | 25–30ページ相当 |
| 実行結果・ログ | 5–7ページ相当 |

完成コードをそのまま誌面へ貼るのではなく、理解や物語の転換に必要な10〜25行程度を掲載する。補助関数、設定、完全な実装はリポジトリに置く。

## 本文から外す内容

64ページの焦点を保つため、初版では次を扱わない。

- 分散学習
- Mixture of Experts
- RLHF
- Instruction tuning
- RAG
- CUDAカーネル最適化
- 推論サーバー
- 本格的なBPE実装
- 最新モデルの比較レビュー

## 実装上の到達点

```text
生テキスト
  ↓
文字単位Tokenizer
  ↓
Dataset / Batch
  ↓
Token Embedding + Position Embedding
  ↓
Causal Self-Attention
  ↓
Feed Forward
  ↓
Residual + LayerNorm
  ↓
Language Model Head
  ↓
Cross Entropy
  ↓
学習
  ↓
テキスト生成
```
