# COSMOP Repository — Ys DEMA 日次レポート運用規約

このリポジトリは仮想通貨戦略 "Ys DEMA SSL1+SSL2+reach_rate v9.0" の
日次レポート Routine の成果物を保管する。以下は全セッション/Routine 共通の規約。

## 保存規約 (正式な保管先 = Git)

- 全レポートは `ys_daily_report/` 配下に保存する。
- 命名: `YYYY-MM-DD.md` (本文) と `YYYY-MM-DD-dashboard.html` (HTML ダッシュボード)。
- **Git が正式な保管先**。Google Drive 保存は任意 (MCP 承認が下りた環境のみ)。
  Drive が使えない場合でも欠落扱いにせず、必ず Git に commit & push する。
- 各 run の最後に作業ブランチへ commit し、`git push -u origin <branch>` する。
- **同日再実行で既存ファイルを無条件上書きしない**。差分を確認し、内容を更新する
  場合は commit メッセージに `update` と理由を明記。価値ある旧版を消さない。

## 各 md に必須: PREDICTION_DATA ブロック

翌々日の Step 4 (予測検証) が参照するため、本文末尾に必ず付与する:

```
<!-- PREDICTION_DATA -->
date: YYYY-MM-DD
btc_close_at_report: <int>
predicted_range_low: <int>
predicted_range_high: <int>
bias: <Long|Neutral|Short  (+/- 修飾可)>
volatility: <低|中|高  (範囲表記可)>
confidence: <低|中|高>
<!-- /PREDICTION_DATA -->
```

dashboard 側の予測帯オーバーレイも、この値と同一にする。

## 指標定義の固定 (run 間のブレ防止)

- ヘッダーの PF バッジは **直近30日実績** を表示する。フル期間バックテスト値
  (PF 2.05 等) と混同しない。参考値を出す場合はラベルに `参考(02-05)` と明記。
- PF 色分け: PF<1 赤 / 1–1.5 橙 / ≥1.5 緑。
- マクロは「前日 US close」、Crypto は「レポート時点」で統一し、各値に時点を明記。

## データ確信度

- 取引所/指数 API が遮断される環境では値は web 検索ベースの近似。確信度を明示し、
  ATR14 / BB が実計算でなく推定の場合はその旨を必ず記載する。
- dashboard のキャンドルは閲覧ブラウザから Binance API を直接取得 (失敗時は近似補間に
  フォールバック) し、データ出所をチャート下に表示する。

## パラメータ最適化

- 本 Routine では最適化を実施しない (週次 PC ローカルで 1 年データ別途実行)。
- Step 9-3 の推奨列は N<20 のとき過剰最適化懸念大 → グレーアウト/非表示とし、
  具体数値で変更を促さない。`N>=20 かつ 期待PF > 現状PF*1.3` のときのみ「変更推奨」。

## 免責

- 全成果物は分析支援のみ・投資助言ではない旨を明記する。
