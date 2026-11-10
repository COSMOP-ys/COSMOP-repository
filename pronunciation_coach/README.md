# pronunciation_coach

英語の発音を **音素レベルで採点** し、結果を **Claude が日本語でコーチング** する最小プロトタイプ。

```
音声 ─▶ wav2vec2(音素認識) ─┐
                            ├─▶ アライメント & スコア ─▶ Claude が助言
英文 ─▶ espeak-ng(期待音素) ─┘
```

## セットアップ
```bash
sudo apt-get install espeak-ng libsndfile1 libportaudio2   # macOS: brew install espeak-ng
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
```
初回実行時に Hugging Face から音素認識モデル(約1.2GB)をダウンロードします。

## 使い方(Web UI)
```bash
python app.py   # → http://127.0.0.1:7860
```
ブラウザのマイクで録音(または音声ファイルをアップロード)して「採点する」を押すと、
スコア・音素ごとの色分け結果・Claude のアドバイスが表示されます。
`ANTHROPIC_API_KEY` が未設定でもスコア表示までは使えます。

### 会話練習タブ
シチュエーションとレベルを選んで「会話を始める」→ 録音して返事 → Claude が英語で返答します。
右側に、毎ターンの **文法・表現の添削(日本語)** と **発音スコア** が出ます。
- 音声は Whisper(`openai/whisper-small.en`、環境変数 `WHISPER_MODEL` で変更可)で文字起こしします。
- 発音スコアは「文字起こし結果」を期待文として採点します。Whisper は文脈で補正して正しい単語を
  出しがちなので、発音練習タブ(正解文を指定)より甘めに出ます。
- 会話モードには `ANTHROPIC_API_KEY` が必須です。返答の読み上げ(TTS)はありません。

## 使い方(CLI)
```bash
python cli.py "I think the right answer is light." --record 5
python cli.py "I think the right answer is light." --wav my.wav
python cli.py "..." --wav my.wav --no-claude    # スコアのみ(API キー不要)
```
使用モデルは環境変数 `CLAUDE_MODEL` で変更できます(既定: `claude-opus-5-5`)。

## テスト(依存ライブラリ不要)
```bash
python -m unittest discover -s tests
```

## 限界
- 採点は「認識された音素 vs 期待音素」の比較です。認識モデルの誤りがそのままスコアに出ます。
- 音素の長さ・アクセント・イントネーションは採点しません。
- 厳密な採点が必要なら、Azure Pronunciation Assessment など商用 API への差し替えを検討してください
  (`coach/phonemes.py` と `coach/scoring.py` を置き換えるだけで済む構成です)。
