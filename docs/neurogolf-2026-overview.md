# The 2026 NeuroGolf Championship 大会概要

取得元: Kaggle MCP (`neurogolf-2026`)  
大会URL: https://www.kaggle.com/competitions/neurogolf-2026  
取得日: 2026-06-10

## 基本情報

- 大会名: The 2026 NeuroGolf Championship
- 主催: The Neurosynthetic Research Institute
- カテゴリ: Research
- 概要: ARC-AGIの画像変換タスクを解く、できるだけ小さいニューラルネットワークを設計する
- 評価指標: NeuroGolf Metric
- 賞金総額: $50,000
- 最大チームサイズ: 5人
- Kaggle上の参加チーム数: 1,775チーム

## 目的

この大会では、ARC-AGI public training v1 benchmark suite に含まれる400個の画像変換タスクに対して、各タスクを正しく再現するONNX形式のニューラルネットワークを提出する。

重要なのは「解けること」だけではなく、ネットワークをできるだけ小さくすること。各タスクの変換を、少ないパラメータ数・小さいメモリフットプリントで実装することが目標になる。

## タスクの内容

各タスクは、入力グリッドと出力グリッドのペアによって暗黙的に変換ルールが示される。例として、回転、切り抜き、拡大などの変換があり得る。

各 `taskXXX.json` には次の3つのフィールドが含まれる。

- `train`: ARC-AGI-1由来の学習用 input/output ペア
- `test`: ARC-AGI-1由来のテスト用 input/output ペア
- `arc-gen`: ARC-GEN-100K dataset 由来の追加 input/output ペア

各ペアは次の2フィールドを持つ。

- `input`: 入力グリッド
- `output`: 出力グリッド

グリッドは0から9までの整数からなる長方形行列で、サイズは最小1x1、最大30x30。ネットワークへ渡される前に、入力グリッドは `[BATCH_DIM=1, CHANNELS=10, HEIGHT=30, WIDTH=30]` のテンソルへ変換される。各色ピクセルはone-hot channel encoding、元画像の外側にあるクリア領域はzero-hot channel encodingになる。

## データ構成

Kaggle MCPで確認したデータファイル概要:

- 合計ファイル数: 401
- JSONファイル: 400個、合計 97,125,539 bytes
- Pythonファイル: 1個、21,599 bytes

主なファイル:

- `task001.json` から `task400.json`: 各ARC-AGIタスク
- `neurogolf_utils/neurogolf_utils.py`: NeuroGolf用ユーティリティ

## 評価方法

400タスクそれぞれについて、提出されたネットワークが機能的に正しい場合にスコアが与えられる。タスクごとのスコアは次の式。

```text
max(1, 25 - ln(cost))
```

`cost` は次の合計。

- ネットワークの総パラメータ数
- ネットワークの総メモリフットプリント(byte)

機能的な正しさは、元のARC-AGIベンチマークと小規模な非公開ベンチマークで検証される。過学習を防ぐため、公開例だけでなく非公開データでも正解する必要がある。正解判定は完全一致で、出力グリッドの全セルが期待値と一致する必要がある。

## 提出形式

提出ファイル名は `submission.zip`。中には各タスクにつき最大1つのONNXファイルを含める。

```text
task001.onnx
task002.onnx
...
task400.onnx
```

すべてのタスクを提出する必要はなく、各タスクにつき高効率なONNXネットワークを用意する形式。

## 制約

- 各ONNXネットワーク内のテンソルとパラメータは、静的に定義されたshapeを持つ必要がある
- 禁止ONNX operation:
  - `Loop`
  - `Scan`
  - `NonZero`
  - `Unique`
  - `Script`
  - `Function`
- 各ONNXファイルサイズ上限: 1.44MB
- これらの制約は公式network validatorで自動チェックされる

## 賞金

賞金総額は $50,000。

- 1位: $12,000
- 2位: $10,000
- 3位: $10,000
- Top Student Team: $8,000
- Longest Leader: $10,000

Top Student Teamは、大学生または大学院生がチームメンバーの50%以上を占めるチームが対象。Longest Leaderは、2026-05-06 00:00 UTCから2026-07-15 23:59 UTCまでの間、リーダーボード1位を最も長く保持したチームに与えられる。

## 日程

すべての締切は、特記がない限り該当日の 23:59 UTC。

- 2026-04-15: 開始日
- 2026-07-08: 参加登録締切
- 2026-07-08: チームマージ締切
- 2026-07-15: 最終提出締切

## ライセンス・利用条件

- Winner License Type: Open Source - Apache 2.0
- Data Access and Use: Competition Use and Commercial - Apache 2.0
- 外部データや外部ツールの利用は可能だが、参加者全員が合理的にアクセス可能で、コスト面でも合理的である必要がある
- 受賞者は、提出物生成に使ったソフトウェアコード、学習コード、推論コード、再現に必要な環境説明などの提出を求められる可能性がある

## 関連情報

この大会は IJCAI-ECAI 2026 Competitions Track の一部。上位提出チームは、ドイツ・ブレーメンで開催されるIJCAI-ECAI 2026の特別セッションでの発表に招待される可能性がある。

参考リンク:

- ARC-AGI: https://arcprize.org/arc-agi
- ARC-AGI GitHub: https://github.com/fchollet/ARC-AGI
- ONNX: https://onnx.ai/
