# hiro-llm

`hiro-llm` は、Decoder-only Language ModelとCLIPベースのVision Language Modelを
学習・評価するための独立したPythonプロジェクトです。Notebookや特定の教材repositoryを
実行時に必要とせず、データ準備、学習、checkpoint、推論をCLIから再現できます。

## 主な機能

- causal self-attentionとrelative position biasを使うLanguage Model
- token列による事前学習
- prompt部分をlossから除外するInstruction Tuning
- 凍結した画像encoderと学習可能なprojectorによるVision Pretraining
- Language Modelも更新するVision Instruction Tuning
- local、HTTP、Hugging Face Hubからの汎用データ取得
- version付きcheckpoint、学習再開、Hugging Face Hubへの自動upload

## セットアップ

Python 3.12を使用します。通常のLLM学習にはCUDA対応のPyTorch環境を推奨します。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Vision機能と開発ツールも導入する場合は次を使います。

```bash
python -m pip install -e ".[vision,dev]"
```

`uv` を使う場合は `uv sync --extra vision --extra dev` でも構築できます。

## ディレクトリ

- `configs/`: 再現可能なYAML設定
- `data/raw/`: 取得した入力データ
- `data/processed/`: token化・padding済み配列
- `models/checkpoints/`: version付き学習checkpoint
- `src/hiro_llm/data/`: データ取得と前処理
- `src/hiro_llm/model/`: Language ModelとVision Language Model
- `src/hiro_llm/training/`: scheduler、checkpoint、Trainer
- `tests/`: CPUで動く小型テスト

## 設定

YAMLは `model`、`training`、`data`、`vision`、`hub`、`runtime` に分かれています。
未知のキーや不正な値は起動時にエラーになります。値はCLIから上書きできます。

```bash
hiro-llm train --config configs/pretrain.yaml \
  --set training.batch_size=4 \
  --set runtime.device=cuda
```

`local`、`http`、`huggingface` の3種類のdata sourceを利用できます。Hugging Faceの
checkpoint参照は `hf://owner/repository/path/to/checkpoint.pt` と記述します。

## データ準備

同梱の小さなデータは前処理確認用です。

```bash
hiro-llm data prepare --config configs/pretrain.yaml
hiro-llm data prepare --config configs/instruction_tuning.yaml
hiro-llm data prepare --config configs/vision_pretrain.yaml
hiro-llm data prepare --config configs/vision_instruction_tuning.yaml
```

事前学習はUTF-8 textまたは一次元 `.npy` token列を入力にします。Instruction Tuningは
JSONLを読み、`prompt_field` と `response_field` でフィールド名を指定します。Vision用
JSONLはさらに `image_field` を持ち、相対パスは `image_root` を基準に解決されます。

## 学習

学習checkpointはローカル保存後、必ずHugging Face Hubへuploadされます。設定内の
`hub.repo_id` を自分のrepositoryへ変更し、開始前にtokenを設定してください。

```bash
export HF_TOKEN=hf_xxx
hiro-llm train --config configs/pretrain.yaml
hiro-llm train --config configs/instruction_tuning.yaml
hiro-llm train --config configs/vision_pretrain.yaml
hiro-llm train --config configs/vision_instruction_tuning.yaml
```

別stageの重みから開始するときは `training.init_from`、同じrunをoptimizer stateごと
再開するときは `training.resume_from` を使います。Vision PretrainingのLanguage Modelは
`vision.language_model_checkpoint` で指定します。Hub認証またはrepo IDが不足している場合、
学習は最初のstepより前に停止します。

## 生成

```bash
hiro-llm generate \
  --config configs/instruction_tuning.yaml \
  --checkpoint models/checkpoints/checkpoint-00001000.pt \
  --prompt "Explain causal attention."

hiro-llm generate \
  --config configs/vision_instruction_tuning.yaml \
  --checkpoint hf://USER/MODEL/checkpoint-00001000.pt \
  --image data/raw/sample.ppm \
  --prompt "<user>What is shown?<assistant>"
```

## Shakespeare実験と評価

Shakespeare本文、TinyStories、SmolTalkの設定を順に実行できます。
Hugging FaceとW&Bへログインしたうえで、各設定の`hub.repo_id`を自分のrepositoryへ変更してください。

```bash
hiro-llm data prepare --config configs/shakespeare_pretrain.yaml
hiro-llm train --config configs/shakespeare_pretrain.yaml
hiro-llm data prepare --config configs/tinystories_adaptation.yaml
hiro-llm train --config configs/tinystories_adaptation.yaml
hiro-llm data prepare --config configs/smoltalk_sft.yaml
hiro-llm train --config configs/smoltalk_sft.yaml
```

checkpointの評価は、設定をcheckpointから復元するため、次のように実行します。

```bash
hiro-llm evaluate \
  --checkpoint models/checkpoints/checkpoint-00010000.pt \
  --benchmark all \
  --set runtime.device=cuda
```

W&B entityはログイン済みユーザーから解決されます。
通信が失敗した場合は再試行後に`wandb_offline/`へJSONLを保存し、ネットワーク復旧後に`wandb sync wandb_offline/`で同期できます。

## Docker

```bash
docker build -t hiro-llm .
docker run --rm hiro-llm --help
```

GPU利用時はホストに対応するNVIDIA Container ToolkitとCUDA対応PyTorch image構成が
別途必要です。標準DockerfileはCLIとCPU実行を確認するための最小構成です。

## 開発

```bash
pytest
ruff check .
python -m build
```

## 謝辞

このプロジェクトの教育的な着想には EveryonesLLM の公開教材を参考にしました。
実装、API、設定、checkpoint形式、データ処理は本プロジェクト向けに独立して構成しています。

## License

MIT License
