# TSM / 2330 Mobile Streamlit App — v6

這版只修配色與對比度：

- 移除亮紅色主按鈕，改成深綠。
- 強制修正標題、卡片、metric、tabs 在 Streamlit Cloud 上變白看不清的問題。
- 保留原本程式邏輯，沒有改 BUY/HOLD 計算。

## Deploy

1. 把 ZIP 解壓縮。
2. 將解壓縮後的所有檔案上傳/覆蓋到 GitHub repo 根目錄。
3. Streamlit Cloud main file path 使用：`app.py`。

## Files

- `app.py`：手機版 Streamlit 介面
- `tsm_core.py`：投資系統核心
- `requirements.txt`：部署套件
- `.streamlit/config.toml`：簡化配色

此 App 只提供 BUY / HOLD 與部位試算，不會自動下單。
