# TSM / 2330 Mobile Streamlit App — v5

這版移除了 Colab / IPython import 的硬依賴，避免 Streamlit Cloud 出現 `ModuleNotFoundError: No module named 'IPython'`。

## Deploy

1. 把 ZIP 解壓縮。
2. 將解壓縮後的所有檔案上傳/覆蓋到 GitHub repo 根目錄。
3. Streamlit Cloud main file path 使用：`app.py`。

## Files

- `app.py`：手機版 Streamlit 介面
- `tsm_core.py`：投資系統核心
- `requirements.txt`：部署套件
- `.streamlit/config.toml`：簡化配色

## Note

此 App 只提供 BUY / HOLD 與部位試算，不會自動下單。
