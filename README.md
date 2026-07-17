# hiro-llm

篠澤広とプロデューサーが、小さな言語モデルをPyTorchで自作する技術同人誌プロジェクトです。

完成済みのLLMを便利に使うのではなく、データの読み込み、トークン化、Transformer、学習、文章生成までを自分たちの手で組み立てます。

本書の中心にあるのは、篠澤広が好む「ままならなさ」です。コードが動いても文章にならない、lossが下がっても理解したようには見えない、モデルを大きくしても別の問題が増える——そうした失敗を避けるのではなく、観察しながらLLMの仕組みを理解していきます。

## 企画概要

- 形式: B5・64ページ（表紙・奥付を含む）
- 対象読者: Python経験者
- 実装: PyTorchを使用し、Transformer部分を自作
- 構成: 会話、技術解説、数式、コード、実行結果
- 到達点: 小さなdecoder-only Transformerを学習し、文章を生成する
- 完全な実装コードはリポジトリに収録し、誌面では理解に必要な部分を抜粋する

## 実装するもの

- テキストの読み込みとデータ分割
- 文字単位Tokenizer
- Token Embeddingと位置情報
- Causal Self-Attention
- Multi-Head Attention
- Feed Forward Network
- Residual connectionとLayerNorm
- Transformer Block
- Language Model Head
- 学習・評価ループ
- テキスト生成

PyTorchのテンソル演算、autograd、`nn.Linear`、`nn.Embedding`、`nn.LayerNorm`、optimizerは利用します。`nn.Transformer`や既存のAttention層には依存せず、Transformerの主要部分を実装します。

## 本書の方針

技術を順番に説明するだけでなく、各段階で生じる失敗を物語の進行に組み込みます。

```text
データを読む
  ↓
トークン化する
  ↓
Transformerを組む
  ↓
学習する
  ↓
生成する
  ↓
思い通りにならない理由を調べる
```

最終的な目標は、完璧に話すモデルを作ることではありません。

> 思い通りに話すものを作りたかった。  
> でも、思い通りにならない理由の方が、ずっと面白かった。

## ドキュメント

- [64ページ構成案](docs/book-plan.md)
- [同人誌としての演出・編集方針](docs/editorial-review.md)

## ステータス

現在は企画・構成設計段階です。実装、原稿、図版、組版は今後追加します。

## 注意

本リポジトリは非公式の二次創作・技術解説プロジェクトです。原作および関連する企業・団体とは関係ありません。

参考資料として購入した書籍やPDFそのものは、このリポジトリには収録しません。
