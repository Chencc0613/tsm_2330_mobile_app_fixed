import math
from datetime import date, datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import tsm_core as core


st.set_page_config(
    page_title="TSM / 2330 Mobile Buy System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #f7f7f4;
        --card: #ffffff;
        --ink: #141414;
        --muted: #686868;
        --line: #dedbd2;
        --accent: #2f5d50;
        --warn: #a66a00;
        --danger: #9f2f2f;
    }
    .stApp { background: var(--bg); }
    h1, h2, h3 { letter-spacing: -0.03em; }
    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--line);
        padding: 14px 14px 10px 14px;
        border-radius: 18px;
        box-shadow: 0 1px 8px rgba(0,0,0,0.04);
    }
    .decision-card {
        background: #161616;
        color: white;
        border-radius: 22px;
        padding: 18px 18px;
        margin: 6px 0 16px 0;
        border: 1px solid #2d2d2d;
    }
    .decision-title { font-size: 12px; color: #b9b9b9; font-weight: 800; letter-spacing: 0.08em; }
    .decision-main { font-size: clamp(32px, 8vw, 56px); font-weight: 950; line-height: 1.0; margin-top: 8px; }
    .pill { display:inline-block; padding: 6px 10px; border-radius: 999px; font-size: 13px; font-weight: 800; margin-right: 6px; }
    .pill-buy { background: #2f5d50; color: white; }
    .pill-hold { background: #606060; color: white; }
    .pill-warn { background: #a66a00; color: white; }
    .small-muted { color: #686868; font-size: 13px; line-height: 1.5; }
    .block-note {
        background: #ffffff;
        border-left: 5px solid #2f5d50;
        border-radius: 14px;
        border-top: 1px solid #dedbd2;
        border-right: 1px solid #dedbd2;
        border-bottom: 1px solid #dedbd2;
        padding: 12px 14px;
        margin: 10px 0;
    }
    div[data-testid="stSidebar"] { background: #efeee9; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Formatting helpers
# -----------------------------

def fnum(x: Any, digits: int = 2) -> str:
    try:
        if x is None or pd.isna(x):
            return "NA"
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "NA"


def fpct(x: Any, digits: int = 2) -> str:
    try:
        if x is None or pd.isna(x):
            return "NA"
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return "NA"


def fmoney(x: Any, currency: str, digits: int = 0) -> str:
    try:
        if x is None or pd.isna(x):
            return "NA"
        symbol = "$" if currency == "USD" else "NT$"
        return f"{symbol}{float(x):,.{digits}f}"
    except Exception:
        return "NA"


def safe_float(x: Any, fallback: float = np.nan) -> float:
    try:
        if x is None:
            return fallback
        v = float(x)
        return v if np.isfinite(v) else fallback
    except Exception:
        return fallback


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.tail(10).reset_index().astype(str).to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tail(10).astype(str).to_dict()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if pd.isna(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


# -----------------------------
# Core adapter
# -----------------------------

def apply_config(cfg: Dict[str, Any]) -> None:
    core.CORE_TICKER = cfg["ticker"]
    core.EQUITY_USD = cfg["equity"]
    core.FREE_CASH_USD = cfg["free_cash"]
    core.HAIRCUT = cfg["haircut"]
    core.TSM_SHARES = cfg["shares"]
    core.TSM_AVG_COST = cfg["avg_cost"]
    core.DEFAULT_START = cfg["start"]
    core.DEFAULT_END = cfg["end"]
    core.ALLOW_FRACTIONAL = cfg["allow_fractional"]
    core.FRACTIONAL_DP = cfg["fractional_dp"]
    core.MIN_TRADE_USD = cfg["min_trade"]
    core.MAX_DAILY_CASH_USE_FRAC = cfg["max_daily_cash_use_frac"]
    core.ALLOW_MARGIN_BUY = cfg["allow_margin"]
    core.MAX_MANUAL_GROSS_LEVERAGE = cfg["max_gross_leverage"]
    core.SIGMA_TARGET = cfg["sigma_target"]

    core.RISK_RULES["BUY_TREND_MODE"] = cfg["trend_mode"]
    core.RISK_RULES["PE_SOURCE"] = cfg["pe_source"]
    core.RISK_RULES["PEG_GROWTH_SOURCE"] = cfg["peg_source"]
    core.RISK_RULES["VALUATION_COMBINE_MODE"] = cfg["valuation_combine"]
    core.RISK_RULES["MACRO_HARD_BLOCK_ENABLE"] = cfg["macro_hard_block"]
    core.RISK_RULES["NO_BUY_BELOW_MA200"] = cfg["no_buy_below_ma200"]
    core.RISK_RULES["PE_HARD_BLOCK_ENABLE"] = cfg["pe_hard_block"]
    core.RISK_RULES["PEG_HARD_BLOCK_ENABLE"] = cfg["peg_hard_block"]

    core.FORECAST_RULES["YEARS"] = cfg["forecast_years"]
    core.FORECAST_RULES["EPS_SOURCE"] = cfg["forecast_eps_source"]


@st.cache_data(ttl=900, show_spinner=False)
def compute_mobile_system(cfg_hashable: tuple) -> Dict[str, Any]:
    cfg = dict(cfg_hashable)
    apply_config(cfg)

    res_macro = core.build_macro_dashboard(
        start=core.DEFAULT_START,
        end=core.DEFAULT_END,
        lookback=core.DEFAULT_LOOKBACK,
        weights=core.DEFAULT_MACRO_WEIGHTS,
        thresholds=core.DEFAULT_MACRO_THRESHOLDS,
        disloc_thr=core.DEFAULT_DISLOC_THR,
        ffill_limit=core.DEFAULT_FFILL_LIMIT,
    )
    if res_macro is None:
        raise RuntimeError("Macro dashboard failed: no data returned from yfinance/FRED.")

    tsm_risk = core.compute_tsm_risk(
        ticker=core.CORE_TICKER,
        end=core.DEFAULT_END,
        rules=core.RISK_RULES,
        avg_cost=core.TSM_AVG_COST,
    )

    holdings = core.compute_tsm_holdings(tsm_risk["last_price"])

    try:
        sigma_ann = core.ann_vol(tsm_risk["ret"], lookback=60)
    except Exception:
        sigma_ann = tsm_risk.get("sigma_60", np.nan)

    try:
        dd_now = core.drawdown_now(tsm_risk["ret"])
    except Exception:
        dd_now = tsm_risk.get("dd_120", np.nan)

    try:
        beta = core.estimate_beta(tsm_risk["ret"], bench="^NDX", lookback_days=260)
    except Exception:
        beta = np.nan

    risk_pack = {"sigma_ann": sigma_ann, "dd_now": dd_now, "beta": beta}

    current_stock_value = float(holdings.loc[core.CORE_TICKER, "market_value"])
    lev_pack = core.leverage_engine(
        sigma_ann=sigma_ann,
        dd_now=dd_now,
        regime=res_macro["regime"],
        crisis_on=res_macro["crisis_on"],
        limits=res_macro["limits"],
        equity_usd=core.EQUITY_USD,
        free_cash_usd=core.FREE_CASH_USD,
        current_stock_usd=current_stock_value,
    )

    buy_plan = core.build_tsm_buy_plan(
        tsm_risk=tsm_risk,
        holdings=holdings,
        lev_pack=lev_pack,
        res_macro=res_macro,
    )

    try:
        forecast_pack = core.compute_tsm_future_price_forecast(
            current_price=tsm_risk["last_price"],
            ticker=core.CORE_TICKER,
            forecast_rules=core.FORECAST_RULES,
        )
    except Exception as e:
        forecast_pack = {"error": str(e), "forecast_df": pd.DataFrame()}

    return {
        "macro": res_macro,
        "tsm_risk": tsm_risk,
        "holdings": holdings,
        "risk_pack": risk_pack,
        "leverage": lev_pack,
        "buy_plan": buy_plan,
        "forecast": forecast_pack,
    }


# -----------------------------
# Sidebar inputs
# -----------------------------

st.sidebar.title("設定")
market_mode = st.sidebar.radio(
    "標的 / 幣別",
    ["TSM ADR / USD", "2330.TW 台股 / TWD", "自訂 ticker"],
    index=0,
)

if market_mode == "TSM ADR / USD":
    default_ticker = "TSM"
    currency = "USD"
elif market_mode == "2330.TW 台股 / TWD":
    default_ticker = "2330.TW"
    currency = "TWD"
else:
    default_ticker = "TSM"
    currency = st.sidebar.selectbox("顯示幣別", ["USD", "TWD"], index=0)

ticker = st.sidebar.text_input("Ticker", default_ticker).strip().upper()

st.sidebar.markdown("---")
st.sidebar.caption("帳戶與持股。若用 2330.TW，下面金額請填 TWD；若用 TSM，請填 USD。")
equity = st.sidebar.number_input(f"總權益 Equity ({currency})", min_value=0.0, value=32810.43 if currency == "USD" else 1000000.0, step=100.0)
free_cash = st.sidebar.number_input(f"可用現金 Free Cash ({currency})", min_value=0.0, value=498.18 if currency == "USD" else 150000.0, step=100.0)
shares = st.sidebar.number_input("目前股數", min_value=0.0, value=75.0 if ticker == "TSM" else 0.0, step=1.0)
avg_cost = st.sidebar.number_input(f"持有均價 ({currency})", min_value=0.0, value=431.12 if ticker == "TSM" else 0.0, step=1.0)

with st.sidebar.expander("買進規則", expanded=False):
    allow_fractional = st.checkbox("允許零股 / fractional", value=False)
    fractional_dp = st.slider("零股小數位", 1, 6, 3)
    min_trade = st.number_input(f"最低交易金額 ({currency})", min_value=0.0, value=2.0 if currency == "USD" else 1000.0, step=1.0)
    max_daily_cash_use_frac = st.slider("單次最多使用現金比例", 0.0, 1.0, 1.0, 0.05)
    allow_margin = st.checkbox("允許融資 / margin", value=False)
    max_gross_leverage = st.slider("最大曝險 / Equity", 0.0, 2.0, 1.20, 0.05)

with st.sidebar.expander("模型參數", expanded=False):
    trend_mode = st.selectbox("趨勢模式", ["BALANCED", "STRICT", "LOOSE"], index=0)
    pe_source = st.selectbox("PE source", ["FORWARD", "TRAILING", "BLENDED", "CONSERVATIVE"], index=0)
    peg_source = st.selectbox("PEG growth source", ["CAPEX_ADJUSTED", "CAPPED_FORWARD_EPS", "MANUAL", "YF_EARNINGS_GROWTH", "BLENDED"], index=0)
    valuation_combine = st.selectbox("PE/PEG 合成", ["MIN", "PRODUCT", "AVERAGE"], index=0)
    sigma_target = st.slider("年化波動目標", 0.01, 0.40, 0.12, 0.01)
    no_buy_below_ma200 = st.checkbox("低於 MA200 禁止買", value=False)
    macro_hard_block = st.checkbox("Macro hard block", value=True)
    pe_hard_block = st.checkbox("PE extreme hard block", value=False)
    peg_hard_block = st.checkbox("PEG extreme hard block", value=False)

with st.sidebar.expander("日期 / 估值預測", expanded=False):
    start_date = st.date_input("資料起始日", value=date(2020, 1, 18))
    end_date = st.date_input("資料截止日", value=date.today())
    forecast_years = st.slider("預測年數", 1, 10, 5)
    forecast_eps_source = st.selectbox("Forecast EPS 起點", ["FORWARD", "TRAILING", "BLENDED"], index=0)

cfg = {
    "ticker": ticker,
    "currency": currency,
    "equity": float(equity),
    "free_cash": float(free_cash),
    "haircut": 1.0,
    "shares": float(shares),
    "avg_cost": float(avg_cost),
    "start": start_date.strftime("%Y-%m-%d"),
    "end": end_date.strftime("%Y-%m-%d"),
    "allow_fractional": bool(allow_fractional),
    "fractional_dp": int(fractional_dp),
    "min_trade": float(min_trade),
    "max_daily_cash_use_frac": float(max_daily_cash_use_frac),
    "allow_margin": bool(allow_margin),
    "max_gross_leverage": float(max_gross_leverage),
    "sigma_target": float(sigma_target),
    "trend_mode": trend_mode,
    "pe_source": pe_source,
    "peg_source": peg_source,
    "valuation_combine": valuation_combine,
    "macro_hard_block": bool(macro_hard_block),
    "no_buy_below_ma200": bool(no_buy_below_ma200),
    "pe_hard_block": bool(pe_hard_block),
    "peg_hard_block": bool(peg_hard_block),
    "forecast_years": int(forecast_years),
    "forecast_eps_source": forecast_eps_source,
}

# hashable form for cache
cfg_hashable = tuple(sorted(cfg.items()))

# -----------------------------
# Main display
# -----------------------------

st.title("TSM / 2330 手機投資儀表板")
st.caption("Manual execution only：這個 App 只給 BUY / HOLD 與部位試算，不會自動下單。")

run = st.button("重新計算", type="primary", use_container_width=True)
if "has_run" not in st.session_state:
    st.session_state.has_run = True

if run:
    st.cache_data.clear()

try:
    with st.spinner("抓資料並計算中…"):
        result = compute_mobile_system(cfg_hashable)
except Exception as e:
    st.error("計算失敗。常見原因是網路、yfinance/FRED 暫時無資料，或 ticker/日期不正確。")
    st.exception(e)
    st.stop()

buy = result["buy_plan"]
macro = result["macro"]
risk = result["tsm_risk"]
holdings = result["holdings"]
risk_pack = result["risk_pack"]
lev = result["leverage"]
forecast = result["forecast"]

valid_buy = buy.get("action") == "BUY" and not pd.isna(buy.get("shares_to_buy", np.nan)) and buy.get("shares_to_buy", 0) > 0
trade_title = f"BUY {core.fmt_shares(buy.get('shares_to_buy'))} 股" if valid_buy else "HOLD"
pill_cls = "pill-buy" if valid_buy else "pill-hold"

st.markdown(
    f"""
    <div class="decision-card">
        <div class="decision-title">{ticker} CASH BUY DECISION</div>
        <div class="decision-main">{trade_title}</div>
        <div style="margin-top:12px;">
            <span class="pill {pill_cls}">ACTION: {buy.get('action')}</span>
            <span class="pill pill-warn">REGIME: {macro.get('regime')}</span>
            <span class="pill {'pill-warn' if macro.get('crisis_on') else 'pill-buy'}">CRISIS: {'ON' if macro.get('crisis_on') else 'OFF'}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("現價", fmoney(buy.get("price"), currency, 2))
c2.metric("建議買進金額", fmoney(buy.get("est_buy_usd"), currency, 0))
c3.metric("估計新均價", fmoney(buy.get("est_new_avg_cost"), currency, 2))
c4.metric("Valuation Scale", f"x{fnum(buy.get('valuation_scale'), 2)}")

if buy.get("reasons"):
    st.markdown("<div class='block-note'><b>系統理由</b><br>" + "<br>".join([f"• {r}" for r in buy.get("reasons", [])]) + "</div>", unsafe_allow_html=True)

main_tab, signal_tab, chart_tab, forecast_tab, raw_tab = st.tabs(["總覽", "訊號", "圖表", "五年估值", "Raw Data"])

with main_tab:
    st.subheader("部位狀態")
    h = holdings.copy().reset_index()
    h["price"] = h["price"].map(lambda x: fmoney(x, currency, 2))
    h["market_value"] = h["market_value"].map(lambda x: fmoney(x, currency, 0))
    h["avg_cost"] = h["avg_cost"].map(lambda x: fmoney(x, currency, 2))
    h["cost_value"] = h["cost_value"].map(lambda x: fmoney(x, currency, 0))
    h["unreal_pnl"] = h["unreal_pnl"].map(lambda x: fmoney(x, currency, 0))
    h["unreal_pnl_pct"] = h["unreal_pnl_pct"].map(fpct)
    h["stock_ratio"] = h["stock_ratio"].map(fpct)
    h["cash_ratio"] = h["cash_ratio"].map(fpct)
    st.dataframe(h, use_container_width=True, hide_index=True)

    st.subheader("風險儀表")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("ATR %", fpct(risk.get("atr_pct")))
    r2.metric("120日回撤", fpct(risk.get("dd_120")))
    r3.metric("60日年化波動", fpct(risk_pack.get("sigma_ann")))
    r4.metric("Beta vs NDX", fnum(risk_pack.get("beta"), 2))

    st.markdown(
        "<div class='small-muted'>這是決策輔助，不是券商下單介面。按 BUY 也只是代表系統建議，仍需你手動去 Firstrade / 券商下單。</div>",
        unsafe_allow_html=True,
    )

with signal_tab:
    st.subheader("核心訊號")
    sig = pd.DataFrame([
        {"Signal": "Trend", "State": buy.get("trend_state"), "Scale": buy.get("trend_scale")},
        {"Signal": "PE", "State": buy.get("pe_state"), "Scale": buy.get("pe_scale")},
        {"Signal": "PEG", "State": buy.get("peg_state"), "Scale": buy.get("peg_scale")},
        {"Signal": "CapEx", "State": f"{buy.get('growth_quality')} / {buy.get('capex_signal', {}).get('capex_trend', 'UNKNOWN')}", "Scale": np.nan},
        {"Signal": "Macro", "State": macro.get("regime"), "Scale": np.nan},
    ])
    st.dataframe(sig, use_container_width=True, hide_index=True)

    st.subheader("Macro factors")
    latest = macro.get("latest", {})
    macro_df = pd.DataFrame([{"Factor": k, "Value": v} for k, v in latest.items()])
    st.dataframe(macro_df, use_container_width=True, hide_index=True)

    st.subheader("估值細節")
    val_df = pd.DataFrame([
        {"Metric": "Selected PE", "Value": buy.get("selected_pe")},
        {"Metric": "Forward PE", "Value": buy.get("forward_pe")},
        {"Metric": "Trailing PE", "Value": buy.get("trailing_pe")},
        {"Metric": "Forward EPS", "Value": buy.get("forward_eps")},
        {"Metric": "Trailing EPS", "Value": buy.get("trailing_eps")},
        {"Metric": "Normalized PEG", "Value": buy.get("peg")},
        {"Metric": "Raw growth %", "Value": buy.get("raw_growth_pct")},
        {"Metric": "Used growth %", "Value": buy.get("used_growth_pct")},
        {"Metric": "Growth source", "Value": buy.get("growth_source")},
        {"Metric": "CapEx source", "Value": buy.get("capex_source")},
    ])
    st.dataframe(val_df, use_container_width=True, hide_index=True)

with chart_tab:
    st.subheader("價格 / MA60 / MA200")
    close = risk.get("close", pd.Series(dtype=float)).copy()
    if close is not None and not close.empty:
        price_chart = pd.DataFrame({
            "Close": close,
            "MA60": close.rolling(int(core.RISK_RULES.get("MA_FAST", 60))).mean(),
            "MA200": close.rolling(int(core.RISK_RULES.get("MA_SLOW", 200))).mean(),
        }).dropna(how="all")
        st.line_chart(price_chart, use_container_width=True)
    else:
        st.info("沒有價格資料。")

    st.subheader("Buy scale components")
    final_cash_usage = safe_float(buy.get("est_buy_usd"), 0.0) / free_cash if free_cash > 0 else 0.0
    scale_df = pd.DataFrame({
        "Component": ["Trend", "PE", "PEG", "Valuation", "Final Cash Usage"],
        "Scale": [buy.get("trend_scale"), buy.get("pe_scale"), buy.get("peg_scale"), buy.get("valuation_scale"), final_cash_usage],
    }).set_index("Component")
    st.bar_chart(scale_df, use_container_width=True)

    capex_table = buy.get("capex_signal", {}).get("capex_table", pd.DataFrame())
    if capex_table is not None and not capex_table.empty:
        st.subheader("CapEx trend")
        st.line_chart(capex_table, use_container_width=True)

with forecast_tab:
    st.subheader(f"{forecast_years} 年情境估值")
    if "error" in forecast:
        st.warning(f"Forecast 無法計算：{forecast['error']}")
    fdf = forecast.get("forecast_df", pd.DataFrame())
    if fdf is not None and not fdf.empty:
        show = fdf.copy()
        money_cols = ["Current Price", "Future EPS", "Price_Current_PE", "Price_Manual_PE", "Price_PEG", "Price_Integrated"]
        pct_cols = ["Growth", "Upside_Current_PE", "Upside_Manual_PE", "Upside_PEG", "Upside_Integrated"]
        for col in money_cols:
            if col in show.columns:
                digits = 2 if col in ["Current Price", "Future EPS"] else 0
                show[col] = show[col].map(lambda x: fmoney(x, currency, digits))
        for col in pct_cols:
            if col in show.columns:
                show[col] = show[col].map(fpct)
        cols = [c for c in ["Scenario", "情境", "Years", "Growth", "Future EPS", "Current PE", "Manual PE", "Scenario PEG", "PEG Implied PE", "Price_Integrated", "Upside_Integrated"] if c in show.columns]
        st.dataframe(show[cols], use_container_width=True, hide_index=True)
        st.markdown(f"<div class='small-muted'>{forecast.get('note', '')}</div>", unsafe_allow_html=True)
    else:
        st.info("沒有 forecast dataframe。")

with raw_tab:
    st.subheader("Raw result objects")
    st.write("Buy plan")
    st.json(to_jsonable({k: v for k, v in buy.items() if k not in ["capex_series"]}), expanded=False)
    st.write("Leverage")
    st.json(to_jsonable(lev), expanded=False)
    st.write("Macro flags")
    st.json(to_jsonable(macro.get("crisis_flags", {})), expanded=False)
