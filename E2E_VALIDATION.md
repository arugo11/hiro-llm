# Full E2E validation

実施日: 2026-07-18  
環境: NVIDIA A100 80GB PCIe、PyTorch 2.6.0、CUDA 12.4

## Result

同梱sampleデータを使用し、4つの学習stageを設定どおり合計17,000 step実行しました。
これは機能と安定性の検証であり、汎化性能の評価ではありません。

| Stage | Steps | Final loss | Hub repository |
|---|---:|---:|---|
| LLM pretraining | 10,000 | 1.49e-08 | [argo11/hiro-llm-pretrain](https://huggingface.co/argo11/hiro-llm-pretrain) |
| Instruction tuning | 1,000 | 1.24e-04 | [argo11/hiro-llm-instruction](https://huggingface.co/argo11/hiro-llm-instruction) |
| Vision pretraining | 5,000 | 0.498 | [argo11/hiro-llm-vision-pretrain](https://huggingface.co/argo11/hiro-llm-vision-pretrain) |
| Vision instruction tuning | 1,000 | 0.0250 | [argo11/hiro-llm-vision-instruction](https://huggingface.co/argo11/hiro-llm-vision-instruction) |

各repositoryはpublicで、10 checkpointずつ、合計40 checkpointを保持しています。全最終
checkpointはformat version 1として読み戻し、期待するtaskとstepを確認しました。

## Generation checks

- Pretraining: `Language models learn a probability distribution over sequences of tokens...`
- Instruction tuning: causal maskがfuture tokensへのattentionを防ぐ旨を回答
- Vision pretraining: sample画像に赤と青がある旨を回答
- Vision instruction tuning: sample画像に赤と青の領域がある旨を回答

## Issue found and fixed

最初のVision pretraining起動時、text stageのrelative-position tableが256、Vision設定が512で
checkpoint shape mismatchになりました。Visionのsequence長512は維持し、relative-position
上限を256へ揃えて、長距離を既存実装のclampで扱うよう修正しました。text checkpointから
Vision用Language Modelへstate dictを完全にロードする回帰テストを追加しています。

## Cleanup

検証後にlocalの `models/checkpoints/*.pt` を削除し、4つの最終checkpointをHubから再取得して
再検証しました。その後local checkpointを再度削除しており、Hub上の成果物だけを保持しています。
