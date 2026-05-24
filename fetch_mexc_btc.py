import requests, pandas as pd, time
from datetime import datetime

SYMBOL = "BTC_USDT"  # MEXC形式
INTERVAL = "Min15"   # MEXC API形式
START = "2025-05-22"; END = "2026-05-22"
OUTPUT = "MEXC_BTCUSDT_15m_1year.csv"

# MEXC contract API
def fetch():
    all_data = []
    start_ms = int(datetime.strptime(START, "%Y-%m-%d").timestamp())
    end_ms = int(datetime.strptime(END, "%Y-%m-%d").timestamp())
    cur = start_ms
    while cur < end_ms:
        url = f"https://contract.mexc.com/api/v1/contract/kline/{SYMBOL}"
        r = requests.get(url, params={
            'interval': INTERVAL, 'start': cur, 'end': min(cur + 1000 * 900, end_ms)
        }, timeout=15).json()
        if r.get('data'):
            d = r['data']
            for i in range(len(d['time'])):
                all_data.append([
                    d['time'][i], d['open'][i], d['high'][i],
                    d['low'][i], d['close'][i], d['vol'][i]
                ])
            cur = d['time'][-1] + 900
        else:
            break
        time.sleep(0.2)
        print(f"  累計 {len(all_data):,} 本")
    return all_data

data = fetch()
df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
df['open_time_utc'] = pd.to_datetime(df['ts'], unit='s')
df.to_csv(OUTPUT, index=False)
print(f"完了: {OUTPUT}")
