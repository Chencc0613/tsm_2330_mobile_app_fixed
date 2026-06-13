# TSM / 2330 Mobile Buy System

這是從原本 Colab / Notebook 版本改成的手機版 Streamlit App。

## 最佳用法

先用 Streamlit 做成網頁 App：

- iPhone / Android 都可以用瀏覽器打開
- 可以加入主畫面，看起來像 App
- 不需要上架 App Store
- 保留原本 Python / yfinance / FRED 邏輯
- 只做 BUY / HOLD 與試算，不會自動下單

## 本機執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

手機跟電腦在同一個 Wi-Fi 時，可以用 Streamlit 顯示的 Network URL 在手機打開。

## 部署到雲端

最簡單做法：

1. 把這整個資料夾上傳到 GitHub repository
2. 到 Streamlit Community Cloud 建立 App
3. 指定 `app.py`
4. 部署完成後，用手機 Safari / Chrome 打開網址
5. iPhone：分享 → 加入主畫面
6. Android：Chrome 選單 → 加到主畫面

## 使用注意

- `TSM ADR / USD` 模式：帳戶金額請填 USD
- `2330.TW 台股 / TWD` 模式：帳戶金額請填 TWD
- App 裡面的 `est_buy_usd` 等內部變數名稱沿用原系統，但畫面會依模式顯示 USD 或 TWD
- yfinance / FRED 偶爾會抓不到資料，重新整理或稍後再算即可
- 這不是自動交易系統，也不是投資建議，只是你的規則系統手機化


## 2026-06-14 fix

Removed the hard dependency on `pandas_datareader` during import. FRED data is optional; if unavailable, the app still runs with yfinance market-proxy macro data.
