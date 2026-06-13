# ============================================================
# COLAB TSM-ONLY CASH BUY SYSTEM
# Version: Capex-adjusted PEG + QoQ Capex Signal + Capped PEG Fallback + No 0-share Output
#
# - Only holding: TSM
# - Cash buy only by default
# - No stock pool
# - No ETF allocation
# - No futures / options hedge
# - No cash floor
# - MA60 + MA200 balanced trend system
# - PE valuation scale
# - Capex-adjusted normalized PEG valuation scale
# - Old capped PEG kept only as fallback
# - Macro dashboard kept
# - Leverage / volatility engine = advisory only
# - Output only BUY / HOLD
# - Manual execution only, NO auto order
# ============================================================

# ============================================================
# 0) INSTALL
# ============================================================

# Notebook install command removed for Streamlit runtime. See requirements.txt

# ============================================================
# 1) IMPORTS
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf

# pandas_datareader currently can break on Streamlit Cloud depending on the
# pandas version. FRED data is optional in this app, so do not let it crash
# the whole mobile dashboard at import time.
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

from datetime import datetime, timedelta

# These are only needed by the original notebook version. They are optional
# for Streamlit Cloud.
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
except Exception:
    widgets = None
    display = clear_output = HTML = None

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)

# ============================================================
# 2) USER CONFIG
# 每天主要改這裡
# ============================================================

CORE_TICKER = "TSM"

# ----------------------------
# Account inputs
# ----------------------------
EQUITY_USD     = 32810.43
FREE_CASH_USD  = 498.18
HAIRCUT        = 1.00

# ----------------------------
# Current TSM position
# ----------------------------
TSM_SHARES   = 75
TSM_AVG_COST = 431.12

# ----------------------------
# Dates
# ----------------------------
DEFAULT_START = "2020-01-18"
DEFAULT_END   = datetime.today().strftime("%Y-%m-%d")

# ----------------------------
# Buy execution settings
# ----------------------------
ALLOW_FRACTIONAL = False   # False = 只買整股；買不起 1 股就 HOLD
FRACTIONAL_DP = 3

MIN_TRADE_USD = 2.0

# 單日最多使用多少 free cash
MAX_DAILY_CASH_USE_FRAC = 1.00

# 是否允許融資
ALLOW_MARGIN_BUY = False

# 若 ALLOW_MARGIN_BUY=True，最大 TSM 曝險 / equity
MAX_MANUAL_GROSS_LEVERAGE = 1.20

# ----------------------------
# Risk target
# 注意：現在只當風險儀表板，不控制買進
# ----------------------------
SIGMA_TARGET = 0.12

DD_WALLS = [
    (-0.25, 0.60),
    (-0.18, 1.00),
    (-0.10, 1.80),
]

DATA_UNSTABLE_MAX_LEV = 0.50

# ============================================================
# 3) RISK RULES
# ============================================================

RISK_RULES = {
    "ENABLE": True,

    # ========================================================
    # MA60 + MA200 trend system
    # ========================================================
    "MA_FAST": 60,
    "MA_SLOW": 200,

    # STRICT   = 只在 price > MA60 > MA200 時買
    # BALANCED = 分級買進
    # LOOSE    = price > MA60 就可買小量
    "BUY_TREND_MODE": "BALANCED",

    "BUY_SCALE_STRONG": 1.00,
    "BUY_SCALE_RECOVERY": 0.50,
    "BUY_SCALE_NEUTRAL": 0.30,
    "BUY_SCALE_WEAK": 0.00,

    # False = 不用 MA200 一刀切
    "NO_BUY_BELOW_MA200": False,

    # ========================================================
    # ATR / drawdown risk control
    # ========================================================
    "USE_ATR": True,
    "ATR_DAYS": 14,

    "DD_LOOKBACK": 120,

    # 回撤加碼，但不是梭哈
    "ADD_ENABLE": True,
    "ADD_DD": 0.10,
    "ADD_BUY_FRAC_OF_CASH": 0.25,

    # 過熱 / 接刀保護：不賣，只是不買
    "NO_BUY_IF_ATR_ABOVE": 0.055,
    "NO_BUY_IF_DD_BELOW": -0.30,

    # ========================================================
    # PE valuation system
    # ========================================================
    "PE_ENABLE": True,

    # "FORWARD"      = 優先 forward PE
    # "TRAILING"     = 優先 trailing PE
    # "BLENDED"      = 65% forward + 35% trailing
    # "CONSERVATIVE" = forward / trailing 取較高者
    "PE_SOURCE": "FORWARD",

    "PE_MISSING_SCALE": 1.00,

    "PE_CHEAP": 20.0,
    "PE_FAIR": 25.0,
    "PE_WARM": 30.0,
    "PE_EXPENSIVE": 35.0,
    "PE_HARD_BLOCK": 45.0,

    "PE_SCALE_CHEAP": 1.20,
    "PE_SCALE_FAIR": 1.00,
    "PE_SCALE_WARM": 0.70,
    "PE_SCALE_EXPENSIVE": 0.40,
    "PE_SCALE_VERY_EXPENSIVE": 0.15,

    "PE_HARD_BLOCK_ENABLE": False,

    # ========================================================
    # PEG valuation system
    # ========================================================
    "PEG_ENABLE": True,

    # 主模式：
    # "CAPEX_ADJUSTED" = 優先使用 capex-adjusted PEG
    # "CAPPED_FORWARD_EPS" = 舊版 capped PEG
    # "MANUAL" = 手動 growth
    # "YF_EARNINGS_GROWTH" = yfinance earningsGrowth
    # "BLENDED" = 多來源平均
    "PEG_GROWTH_SOURCE": "CAPEX_ADJUSTED",

    # 舊版 fallback 仍保留
    "PEG_FALLBACK_SOURCE": "CAPPED_FORWARD_EPS",

    # 手動長期 EPS growth
    "MANUAL_EPS_GROWTH": 0.28,

    # 舊版 capped PEG 用
    "PEG_MIN_GROWTH": 0.05,
    "PEG_MAX_GROWTH": 0.30,

    # 新版 capex-adjusted PEG 用
    "PEG_UNKNOWN_FALLBACK_GROWTH": 0.12,
    "PEG_MIN_ADJUSTED_GROWTH": 0.05,

    # Capex quality 對 growth 的可信度
    # reliability = raw EPS growth 可相信幾成
    # soft_upper = 非硬 cap，是軟飽和上緣
    "PEG_GROWTH_QUALITY_RULES": {
        "STRONG": {
            "reliability": 0.85,
            "soft_upper": 0.38,
            "scale_bias": 1.00,
        },
        "STABLE": {
            "reliability": 0.65,
            "soft_upper": 0.30,
            "scale_bias": 1.00,
        },
        "WEAK": {
            "reliability": 0.40,
            "soft_upper": 0.22,
            "scale_bias": 0.85,
        },
        "UNKNOWN": {
            "reliability": 0.45,
            "soft_upper": 0.20,
            "scale_bias": 0.80,
        },
    },

    # Capex trend rules
    # QoQ-based: 不再要求一定要有 8 季資料才能算 YoY
    # 只要有 2 季就能算 QoQ；3~4 季可算 recent slope；YoY 只當背景
    "CAPEX_MIN_POINTS": 2,
    "CAPEX_STRONG_YOY": 0.10,
    "CAPEX_WEAK_YOY": -0.10,
    "CAPEX_TREND_WINDOW": 4,
    "CAPEX_UP_SLOPE_REL": 0.03,
    "CAPEX_DOWN_SLOPE_REL": -0.03,
    "CAPEX_QOQ_STRONG": 0.05,
    "CAPEX_QOQ_WEAK": -0.10,

    # PEG 分級
    "PEG_CHEAP": 0.80,
    "PEG_FAIR": 1.20,
    "PEG_WARM": 1.80,
    "PEG_EXPENSIVE": 2.50,
    "PEG_HARD_BLOCK": 3.50,

    "PEG_SCALE_CHEAP": 1.20,
    "PEG_SCALE_FAIR": 1.00,
    "PEG_SCALE_WARM": 0.70,
    "PEG_SCALE_EXPENSIVE": 0.40,
    "PEG_SCALE_VERY_EXPENSIVE": 0.15,

    "PEG_MISSING_SCALE": 1.00,
    "PEG_HARD_BLOCK_ENABLE": False,

    # PE 與 PEG 合成方式
    # MIN     = 取較保守者，建議
    # PRODUCT = PE scale × PEG scale，更嚴格
    # AVERAGE = 平均
    "VALUATION_COMBINE_MODE": "MIN",

    # ========================================================
    # Macro hard block
    # ========================================================
    # DATA-UNSTABLE 不阻止買進
    # 只有 CRISIS / RISK-OFF / Dislocation 阻止
    "MACRO_HARD_BLOCK_ENABLE": True,
}

# ============================================================
# 4) MACRO DATA CONFIG
# ============================================================

FRED_SERIES = {
    "WALCL": "Fed Balance Sheet",
    "RRPONTSYD": "Reverse Repo",
    "WTREGEN": "TGA",
    "SOFR": "SOFR",
    "BAMLH0A0HYM2": "HY Spread",
    "BAMLC0A0CM": "IG Spread",
    "INDPRO": "Industrial Production",
    "RSAFS": "Retail Sales",
    "DGS2": "2Y",
    "DGS10": "10Y",
    "DFII10": "10Y Real Yield",
}

YF_TICKERS = {
    # Market-proxy macro dashboard. 這些比 FRED 即時，圖表不會因刪掉 Shiller/CAPE 後空掉。
    "HYG": "High Yield Credit ETF",
    "LQD": "Investment Grade Credit ETF",
    "^VIX": "VIX",
    "TLT": "Long Treasury ETF",
    "DX-Y.NYB": "DXY Dollar Index",
    "UUP": "Dollar ETF Fallback",
    "GLD": "Gold ETF",
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
}

DEFAULT_MACRO_WEIGHTS = {
    "Liquidity": 1.00,
    "Credit": 1.20,
    "Volatility": 1.00,
    "Growth": 0.60,
    "Rate": 0.80,
    "Geo": 0.50,
}

DEFAULT_MACRO_THRESHOLDS = {
    "VIX": 35.0,
    "Credit": 1.50,
    "Vol": 1.20,
    "MinFlags": 2,
}

DEFAULT_FFILL_LIMIT = 10
DEFAULT_LOOKBACK = 252
DEFAULT_DISLOC_THR = 1.80

# ============================================================
# 5) BASIC HELPERS
# ============================================================

def pretty(x):
    return "NA" if pd.isna(x) else f"{x:+.2f}"


def pct_fmt(x):
    return "NA" if pd.isna(x) else f"{x * 100:.2f}%"


def usd_fmt(x):
    return "NA" if pd.isna(x) else f"${x:,.2f}"


def badge(text, color):
    return f"""
    <span style='display:inline-block;padding:6px 10px;margin:4px;
    border-radius:10px;background:{color};color:white;font-weight:700'>
    {text}</span>
    """


def card(name, val, color):
    # Dark-mode safe card: 明確指定文字顏色，避免 Colab 深色模式白字白底。
    return f"""
    <div style='border-left:6px solid {color};padding:8px 12px;margin:4px;
    background:#ffffff;border-radius:10px;min-width:150px;box-shadow:0 1px 4px rgba(0,0,0,0.08);'>
    <div style='font-size:12px;color:#334155;font-weight:600'>{name}</div>
    <div style='font-size:20px;font-weight:800;color:#111827'>{val}</div>
    </div>
    """


def tone_stress(x):
    if pd.isna(x):
        return "#455a64"
    if x >= 1.2:
        return "#c62828"
    if x <= 0.2:
        return "#2e7d32"
    return "#f9a825"


def plot_line(ax, s, title):
    if s is None:
        ax.text(0.5, 0.5, f"{title}\n(no data)", ha="center", va="center")
        ax.set_axis_off()
        return

    s = pd.Series(s).dropna()
    if len(s) == 0:
        ax.text(0.5, 0.5, f"{title}\n(no data)", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.plot(s.index, s.values)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def shares_from_usd(d_usd, px):
    """
    Convert USD amount to shares.
    Important: return np.nan instead of 0 so the system never prints BUY 0 shares.
    """
    if px is None or px <= 0:
        return np.nan

    if d_usd is None or d_usd <= 0:
        return np.nan

    sh_raw = float(d_usd) / float(px)

    if ALLOW_FRACTIONAL:
        sh = round(sh_raw, FRACTIONAL_DP)

        # 不輸出 0 股；算不到有效股數就交給 build_tsm_buy_plan() 轉 HOLD
        if sh <= 0:
            return np.nan

        return sh

    sh = int(np.floor(sh_raw))

    # 不輸出 0 股；整股模式下買不起 1 股就回傳 np.nan
    if sh <= 0:
        return np.nan

    return sh


def fmt_shares(x):
    if pd.isna(x):
        return "NA"
    if ALLOW_FRACTIONAL:
        return f"{float(x):.{FRACTIONAL_DP}f}"
    return f"{int(x)}"


def fmt_usd_int(x):
    if pd.isna(x):
        return "NA"
    return f"${float(x):,.0f}"


def _safe_float(x):
    try:
        if x is None:
            return np.nan
        v = float(x)
        if np.isfinite(v):
            return v
        return np.nan
    except Exception:
        return np.nan


def normalize_name(x):
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def find_row_by_candidates(df, candidates):
    if df is None or df.empty:
        return None

    norm_map = {normalize_name(idx): idx for idx in df.index}
    cand_norm = [normalize_name(c) for c in candidates]

    for c in cand_norm:
        if c in norm_map:
            return norm_map[c]

    for c in cand_norm:
        for n, raw in norm_map.items():
            if c in n or n in c:
                return raw

    return None


def linear_slope_rel(series):
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return np.nan

    y = s.values.astype(float)
    x = np.arange(len(y))

    avg = np.nanmean(np.abs(y))
    if avg <= 0 or pd.isna(avg):
        return np.nan

    slope = np.polyfit(x, y, 1)[0]
    return slope / avg


def soft_saturate_growth(raw_growth, reliability, soft_upper, min_growth):
    raw_growth = _safe_float(raw_growth)

    if pd.isna(raw_growth) or raw_growth <= 0:
        return np.nan

    reliability = _safe_float(reliability)
    soft_upper = _safe_float(soft_upper)

    if pd.isna(reliability) or reliability <= 0:
        reliability = 0.45

    if pd.isna(soft_upper) or soft_upper <= 0:
        soft_upper = 0.20

    effective_raw = raw_growth * reliability

    # soft saturation，不是硬 cap
    adjusted = soft_upper * (1.0 - np.exp(-effective_raw / max(soft_upper, 1e-9)))
    adjusted = max(adjusted, min_growth)

    return float(adjusted)


# ============================================================
# 6) DATA HELPERS
# ============================================================

def safe_fred(code: str, start: str, end: str) -> pd.Series:
    if pdr is None:
        return pd.Series(dtype=float, name=code)

    try:
        s = pdr.DataReader(code, "fred", start, end)[code]
        s.name = code
        return s
    except Exception:
        return pd.Series(dtype=float, name=code)


def safe_yf_close(ticker: str, start: str = None, end: str = None, interval="1d", period=None) -> pd.Series:
    try:
        if period is not None:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        else:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)

        if df is None or df.empty:
            return pd.Series(dtype=float, name=ticker)

        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                s = df["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
            else:
                return pd.Series(dtype=float, name=ticker)
        else:
            if "Close" not in df.columns:
                return pd.Series(dtype=float, name=ticker)
            s = df["Close"]

        s = pd.Series(s).dropna().copy()
        s.name = ticker
        return s

    except Exception:
        return pd.Series(dtype=float, name=ticker)


def safe_last_price(ticker):
    s = safe_yf_close(ticker, period="20d")
    if s is None or s.empty:
        return None
    return float(s.iloc[-1])


def to_bday_ffill_limit(s: pd.Series, limit_days: int = 10) -> pd.Series:
    if s is None or s.empty:
        return s

    s = s.dropna()
    if s.empty:
        return s

    diffs = pd.Series(s.index).diff().dropna().dt.days
    med_gap = float(diffs.median()) if len(diffs) else 1.0

    if med_gap >= 20:
        lim = 90
    elif med_gap >= 5:
        lim = 25
    else:
        lim = limit_days

    idx = pd.date_range(s.index.min(), s.index.max(), freq="B")
    return s.reindex(idx).ffill(limit=int(lim))


def rolling_z(s: pd.Series, lookback: int, minp: int = 60):
    if s is None or s.empty:
        return s

    mu = s.rolling(lookback, min_periods=minp).mean()
    sd = s.rolling(lookback, min_periods=minp).std()

    z = (s - mu) / sd
    z = z.where(sd > 1e-6)

    if len(z) and pd.isna(z.iloc[-1]):
        last_valid = z.dropna()
        if not last_valid.empty:
            z.iloc[-1] = last_valid.iloc[-1]

    return z.clip(-5, 5)


def pct_chg(s: pd.Series, n: int):
    if s is None or s.empty:
        return s
    return s.pct_change(n)


def diff_n(s: pd.Series, n: int):
    if s is None or s.empty:
        return s
    return s.diff(n)


# ============================================================
# 7) MACRO REGIME ENGINE
# ============================================================

def classify_regime(latest):
    L = latest.get("Liquidity", np.nan)
    C = latest.get("Credit", np.nan)
    V = latest.get("Volatility", np.nan)
    G = latest.get("Growth", np.nan)
    R = latest.get("Rate", np.nan)
    cov = latest.get("Coverage", 0.0)

    if cov < 0.6:
        return "DATA-UNSTABLE"

    if (C >= 1.5) and (V >= 1.2):
        return "CRISIS"

    if (C >= 0.8) or (V >= 0.9):
        return "RISK-OFF"

    if (R >= 0.8) and (L <= 0.3):
        return "LATE-CYCLE"

    if (L >= 0.4) and (G >= 0.2) and (C <= 0.3) and (V <= 0.3):
        return "RISK-ON"

    return "NEUTRAL"


def decision_limits(regime, crisis_on):
    limits = {
        "max_leverage": 0.80,
        "notes": [],
    }

    if regime == "DATA-UNSTABLE":
        limits["max_leverage"] = DATA_UNSTABLE_MAX_LEV
        limits["notes"].append("DATA-UNSTABLE：資料不足，但現在只當 advisory，不阻止現金買進")
        return limits

    if crisis_on or regime == "CRISIS":
        limits["max_leverage"] = 0.00
        limits["notes"].append("CRISIS：禁止買進")
        return limits

    if regime == "RISK-OFF":
        limits["max_leverage"] = 0.50
        limits["notes"].append("RISK-OFF：禁止追價")
        return limits

    if regime == "LATE-CYCLE":
        limits["max_leverage"] = 0.70
        limits["notes"].append("LATE-CYCLE：控制高估值風險")
        return limits

    if regime == "RISK-ON":
        limits["max_leverage"] = 1.00
        limits["notes"].append("RISK-ON：允許正常買進")
        return limits

    limits["max_leverage"] = 0.80
    limits["notes"].append("NEUTRAL：正常但不激進")
    return limits


def _first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns and df[c].dropna().shape[0] > 30:
            return c
    return None


def _z_or_nan(s, lookback):
    if s is None or len(pd.Series(s).dropna()) < 80:
        return pd.Series(np.nan, index=s.index if s is not None else None)
    return rolling_z(pd.Series(s), lookback, minp=60)


def build_macro_dashboard(
    start,
    end,
    lookback,
    weights,
    thresholds,
    disloc_thr=1.8,
    ffill_limit=10
):
    """
    Market-proxy macro dashboard.

    改掉原本太依賴 FRED / Shiller / CAPE 的問題。
    現在至少使用：HYG, LQD, VIX, TLT, DXY/UUP, GLD。

    Factor meaning:
    - Liquidity: 越高越寬鬆 / risk-on
    - Growth: 越高越 risk-on
    - Credit: 越高越信用壓力
    - Volatility: 越高越波動壓力
    - Rate: 越高越利率 / 美元壓力
    - Geo: 越高越避險壓力
    - TotalScore: 越高越風險高
    """

    yfs = {t: safe_yf_close(t, start=start, end=end) for t in YF_TICKERS}
    series = []

    for s in yfs.values():
        if s is not None and not s.empty:
            series.append(to_bday_ffill_limit(s, limit_days=ffill_limit))

    # FRED 仍可抓，但不再是必要條件。抓得到就放 raw，不抓得到也不影響 dashboard。
    fred = {c: safe_fred(c, start, end) for c in FRED_SERIES}
    for s in fred.values():
        if s is not None and not s.empty:
            series.append(to_bday_ffill_limit(s, limit_days=ffill_limit))

    if not series:
        return None

    df = pd.concat(series, axis=1).sort_index()
    df = df.ffill(limit=ffill_limit)

    dollar_col = _first_existing_col(df, ["DX-Y.NYB", "UUP"])
    hyg_col = _first_existing_col(df, ["HYG"])
    lqd_col = _first_existing_col(df, ["LQD"])
    vix_col = _first_existing_col(df, ["^VIX"])
    tlt_col = _first_existing_col(df, ["TLT"])
    gld_col = _first_existing_col(df, ["GLD"])
    spy_col = _first_existing_col(df, ["SPY", "QQQ"])

    ret21 = df.pct_change(21)
    ret63 = df.pct_change(63)
    ret1 = df.pct_change()

    Z = pd.DataFrame(index=df.index)

    # Credit appetite: HYG / LQD 上升 = credit risk appetite 好；下降 = credit stress。
    if hyg_col and lqd_col:
        credit_ratio = (df[hyg_col] / df[lqd_col]).replace([np.inf, -np.inf], np.nan)
        credit_ratio_21 = credit_ratio.pct_change(21)
        Z["z_credit_ratio_21"] = _z_or_nan(credit_ratio_21, lookback)
        Credit = -1.0 * Z["z_credit_ratio_21"]
        Growth_credit = Z["z_credit_ratio_21"]
    else:
        Credit = pd.Series(np.nan, index=df.index)
        Growth_credit = pd.Series(np.nan, index=df.index)

    # Volatility: VIX level z-score.
    if vix_col:
        Z["z_vix"] = _z_or_nan(df[vix_col], lookback)
        Volatility = Z["z_vix"]
        vix_latest = _safe_float(df[vix_col].dropna().iloc[-1]) if df[vix_col].dropna().shape[0] else np.nan
    else:
        Volatility = pd.Series(np.nan, index=df.index)
        vix_latest = np.nan

    # Liquidity: TLT 上漲 + dollar 下跌 = 條件偏寬鬆。
    if tlt_col:
        Z["z_tlt_21"] = _z_or_nan(ret21[tlt_col], lookback)
    else:
        Z["z_tlt_21"] = np.nan

    if dollar_col:
        Z["z_dollar_21"] = _z_or_nan(ret21[dollar_col], lookback)
    else:
        Z["z_dollar_21"] = np.nan

    Liquidity = 0.6 * Z["z_tlt_21"].fillna(0) - 0.6 * Z["z_dollar_21"].fillna(0)
    Rate = -0.8 * Z["z_tlt_21"].fillna(0) + 0.4 * Z["z_dollar_21"].fillna(0)

    # Gold / dollar / VIX避險壓力。
    if gld_col:
        Z["z_gld_21"] = _z_or_nan(ret21[gld_col], lookback)
    else:
        Z["z_gld_21"] = np.nan

    Geo = (
        0.45 * Z["z_gld_21"].fillna(0)
        + 0.35 * Z["z_dollar_21"].fillna(0)
        + 0.20 * Volatility.fillna(0)
    )

    # Growth / risk appetite: HYG/LQD + SPY/QQQ momentum。
    if spy_col:
        Z["z_spy_63"] = _z_or_nan(ret63[spy_col], lookback)
        Growth = 0.65 * Growth_credit.fillna(0) + 0.35 * Z["z_spy_63"].fillna(0)
    else:
        Growth = Growth_credit

    # Fast dislocation: market proxy daily shock。
    shock_parts = []
    for col in [hyg_col, lqd_col, vix_col, tlt_col, dollar_col, gld_col]:
        if col and col in ret1.columns:
            shock_parts.append(_z_or_nan(ret1[col].abs(), lookback).fillna(0))

    if shock_parts:
        Dislocation = sum(shock_parts) / len(shock_parts)
    else:
        Dislocation = pd.Series(np.nan, index=df.index)

    factors = pd.DataFrame({
        "Liquidity": Liquidity,
        "Credit": Credit,
        "Volatility": Volatility,
        "Growth": Growth,
        "Rate": Rate,
        "Geo": Geo,
        "Dislocation": Dislocation,
    }, index=df.index)

    cols = ["Liquidity", "Credit", "Volatility", "Growth", "Rate", "Geo"]
    W = pd.Series(weights)

    avail = factors[cols].notna().astype(float)
    w_eff = avail.mul(W[cols], axis=1)
    w_sum = w_eff.sum(axis=1).replace(0, np.nan)
    w_norm = w_eff.div(w_sum, axis=0)

    # TotalScore 現在明確定義成 risk score：越高越危險。
    risk_components = pd.DataFrame({
        "Liquidity": -factors["Liquidity"],
        "Credit": factors["Credit"],
        "Volatility": factors["Volatility"],
        "Growth": -factors["Growth"],
        "Rate": factors["Rate"],
        "Geo": factors["Geo"],
    }, index=factors.index)

    factors["TotalScore"] = (risk_components[cols] * w_norm).sum(axis=1, skipna=True)
    factors["Coverage"] = avail.sum(axis=1) / len(cols)

    latest_row = factors.dropna(how="all").iloc[-1]
    latest = latest_row.to_dict()

    regime = classify_regime(latest)

    flags = {
        "VIX > threshold": (not pd.isna(vix_latest)) and (vix_latest > thresholds["VIX"]),
        "CreditStress > threshold": (not pd.isna(latest.get("Credit", np.nan))) and (latest["Credit"] > thresholds["Credit"]),
        "VolScore > threshold": (not pd.isna(latest.get("Volatility", np.nan))) and (latest["Volatility"] > thresholds["Vol"]),
        "Dislocation > thr": (not pd.isna(latest.get("Dislocation", np.nan))) and (latest["Dislocation"] > disloc_thr),
    }

    crisis_on = False
    if regime != "DATA-UNSTABLE":
        base = sum(bool(v) for k, v in flags.items() if k != "Dislocation > thr") >= thresholds["MinFlags"]
        veto = bool(flags["Dislocation > thr"])
        crisis_on = base or veto

    limits = decision_limits(regime, crisis_on)

    return {
        "raw": df,
        "Z": Z,
        "factors": factors,
        "latest": latest,
        "regime": regime,
        "crisis_flags": flags,
        "crisis_on": crisis_on,
        "limits": limits,
        "disloc_thr": disloc_thr,
        "vix_latest": float(vix_latest) if not pd.isna(vix_latest) else np.nan,
        "market_cols": {
            "hyg": hyg_col,
            "lqd": lqd_col,
            "vix": vix_col,
            "tlt": tlt_col,
            "dollar": dollar_col,
            "gld": gld_col,
            "spy": spy_col,
        }
    }


# ============================================================
# 8) TSM PRICE / RISK INDICATORS
# ============================================================

def download_ohlc(ticker, end, days=600):
    end_dt = pd.to_datetime(end)
    start_dt = end_dt - pd.Timedelta(days=days * 2)

    df = yf.download(
        ticker,
        start=start_dt.strftime("%Y-%m-%d"),
        end=(end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(f"{ticker} price download failed.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna(how="all")


def classify_ma_trend(last_close, ma60, ma200, rules):
    above_ma60 = last_close > ma60
    above_ma200 = last_close > ma200
    ma60_above_ma200 = ma60 > ma200

    mode = str(rules.get("BUY_TREND_MODE", "BALANCED")).upper()

    if above_ma60 and ma60_above_ma200:
        trend_state = "STRONG"
        trend_scale = float(rules.get("BUY_SCALE_STRONG", 1.0))

    elif above_ma60 and not ma60_above_ma200:
        trend_state = "RECOVERY"
        trend_scale = float(rules.get("BUY_SCALE_RECOVERY", 0.5))

    elif above_ma200 and not above_ma60:
        trend_state = "WEAK_PULLBACK"
        trend_scale = float(rules.get("BUY_SCALE_NEUTRAL", 0.3))

    elif last_close > min(ma60, ma200):
        trend_state = "NEUTRAL"
        trend_scale = float(rules.get("BUY_SCALE_NEUTRAL", 0.3))

    else:
        trend_state = "WEAK"
        trend_scale = float(rules.get("BUY_SCALE_WEAK", 0.0))

    if mode == "STRICT":
        trend_scale = float(rules.get("BUY_SCALE_STRONG", 1.0)) if trend_state == "STRONG" else 0.0

    elif mode == "LOOSE":
        if above_ma60:
            trend_scale = max(trend_scale, float(rules.get("BUY_SCALE_RECOVERY", 0.5)))
        elif above_ma200:
            trend_scale = max(trend_scale, float(rules.get("BUY_SCALE_NEUTRAL", 0.3)))

    if rules.get("NO_BUY_BELOW_MA200", False) and not above_ma200:
        trend_scale = 0.0

    return {
        "ma60": float(ma60),
        "ma200": float(ma200),
        "above_ma60": bool(above_ma60),
        "above_ma200": bool(above_ma200),
        "ma60_above_ma200": bool(ma60_above_ma200),
        "trend_state": trend_state,
        "trend_scale": float(trend_scale),
        "trend_ok": bool(trend_scale > 0),
    }


def compute_tsm_risk(ticker, end, rules, avg_cost=None):
    df = download_ohlc(ticker, end=end, days=600)

    close = df["Close"].dropna()
    high = df["High"].dropna()
    low = df["Low"].dropna()

    ma_fast_n = int(rules.get("MA_FAST", 60))
    ma_slow_n = int(rules.get("MA_SLOW", 200))
    atr_n = int(rules.get("ATR_DAYS", 14))
    dd_lb = int(rules.get("DD_LOOKBACK", 120))

    last_close = float(close.iloc[-1])
    ma60 = float(close.rolling(ma_fast_n).mean().iloc[-1])
    ma200 = float(close.rolling(ma_slow_n).mean().iloc[-1])

    trend_pack = classify_ma_trend(
        last_close=last_close,
        ma60=ma60,
        ma200=ma200,
        rules=rules,
    )

    roll_high = close.rolling(dd_lb, min_periods=min(60, dd_lb)).max()
    peak_120 = float(roll_high.iloc[-1])
    dd_120 = last_close / peak_120 - 1.0

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(atr_n, min_periods=min(atr_n, 10)).mean()
    atr_last = float(atr.iloc[-1])
    atr_pct = atr_last / last_close

    ret = close.pct_change().dropna()
    sigma_60 = float(ret.tail(60).std(ddof=1) * np.sqrt(252))
    sigma_120 = float(ret.tail(120).std(ddof=1) * np.sqrt(252))

    pnl_pct = np.nan
    if avg_cost is not None and avg_cost > 0:
        pnl_pct = last_close / avg_cost - 1.0

    out = {
        "df": df,
        "close": close,
        "ret": ret,
        "last_price": last_close,
        "peak_120": peak_120,
        "dd_120": float(dd_120),
        "atr": atr_last,
        "atr_pct": float(atr_pct),
        "sigma_60": sigma_60,
        "sigma_120": sigma_120,
        "pnl_pct": float(pnl_pct) if not pd.isna(pnl_pct) else np.nan,
    }

    out.update(trend_pack)
    return out


# ============================================================
# 9) PE VALUATION SYSTEM
# ============================================================

def fetch_tsm_fundamentals(ticker):
    out = {
        "trailing_pe": np.nan,
        "forward_pe": np.nan,
        "trailing_eps": np.nan,
        "forward_eps": np.nan,
        "source_ok": False,
        "error": "",
    }

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        out.update({
            "trailing_pe": _safe_float(info.get("trailingPE", np.nan)),
            "forward_pe": _safe_float(info.get("forwardPE", np.nan)),
            "trailing_eps": _safe_float(info.get("trailingEps", np.nan)),
            "forward_eps": _safe_float(info.get("forwardEps", np.nan)),
            "source_ok": True,
        })

    except Exception as e:
        out["error"] = str(e)

    return out


def compute_pe_pack(ticker, price, rules):
    if not rules.get("PE_ENABLE", True):
        return {
            "pe_enabled": False,
            "pe_source": "DISABLED",
            "selected_pe": np.nan,
            "forward_pe": np.nan,
            "trailing_pe": np.nan,
            "forward_eps": np.nan,
            "trailing_eps": np.nan,
            "pe_state": "DISABLED",
            "pe_scale": 1.00,
            "pe_block": False,
            "pe_note": "PE system disabled",
        }

    f = fetch_tsm_fundamentals(ticker)

    forward_pe = _safe_float(f.get("forward_pe", np.nan))
    trailing_pe = _safe_float(f.get("trailing_pe", np.nan))
    forward_eps = _safe_float(f.get("forward_eps", np.nan))
    trailing_eps = _safe_float(f.get("trailing_eps", np.nan))

    if pd.isna(forward_pe) and not pd.isna(forward_eps) and forward_eps > 0:
        forward_pe = price / forward_eps

    if pd.isna(trailing_pe) and not pd.isna(trailing_eps) and trailing_eps > 0:
        trailing_pe = price / trailing_eps

    source = str(rules.get("PE_SOURCE", "FORWARD")).upper()

    if source == "FORWARD":
        selected_pe = forward_pe if not pd.isna(forward_pe) else trailing_pe

    elif source == "TRAILING":
        selected_pe = trailing_pe if not pd.isna(trailing_pe) else forward_pe

    elif source == "BLENDED":
        if not pd.isna(forward_pe) and not pd.isna(trailing_pe):
            selected_pe = 0.65 * forward_pe + 0.35 * trailing_pe
        elif not pd.isna(forward_pe):
            selected_pe = forward_pe
        else:
            selected_pe = trailing_pe

    elif source == "CONSERVATIVE":
        vals = [x for x in [forward_pe, trailing_pe] if not pd.isna(x)]
        selected_pe = max(vals) if vals else np.nan

    else:
        selected_pe = forward_pe if not pd.isna(forward_pe) else trailing_pe

    if pd.isna(selected_pe) or selected_pe <= 0:
        return {
            "pe_enabled": True,
            "pe_source": source,
            "selected_pe": np.nan,
            "forward_pe": forward_pe,
            "trailing_pe": trailing_pe,
            "forward_eps": forward_eps,
            "trailing_eps": trailing_eps,
            "pe_state": "PE_DATA_MISSING",
            "pe_scale": float(rules.get("PE_MISSING_SCALE", 1.0)),
            "pe_block": False,
            "pe_note": "PE missing; using PE_MISSING_SCALE",
        }

    cheap = float(rules.get("PE_CHEAP", 20.0))
    fair = float(rules.get("PE_FAIR", 25.0))
    warm = float(rules.get("PE_WARM", 30.0))
    expensive = float(rules.get("PE_EXPENSIVE", 35.0))
    hard_block = float(rules.get("PE_HARD_BLOCK", 45.0))

    pe_block = False

    if selected_pe <= cheap:
        pe_state = "CHEAP"
        pe_scale = float(rules.get("PE_SCALE_CHEAP", 1.20))
    elif selected_pe <= fair:
        pe_state = "FAIR"
        pe_scale = float(rules.get("PE_SCALE_FAIR", 1.00))
    elif selected_pe <= warm:
        pe_state = "WARM"
        pe_scale = float(rules.get("PE_SCALE_WARM", 0.70))
    elif selected_pe <= expensive:
        pe_state = "EXPENSIVE"
        pe_scale = float(rules.get("PE_SCALE_EXPENSIVE", 0.40))
    elif selected_pe <= hard_block:
        pe_state = "VERY_EXPENSIVE"
        pe_scale = float(rules.get("PE_SCALE_VERY_EXPENSIVE", 0.15))
    else:
        pe_state = "EXTREME"
        if rules.get("PE_HARD_BLOCK_ENABLE", False):
            pe_scale = 0.0
            pe_block = True
        else:
            pe_scale = float(rules.get("PE_SCALE_VERY_EXPENSIVE", 0.15))

    return {
        "pe_enabled": True,
        "pe_source": source,
        "selected_pe": float(selected_pe),
        "forward_pe": forward_pe,
        "trailing_pe": trailing_pe,
        "forward_eps": forward_eps,
        "trailing_eps": trailing_eps,
        "pe_state": pe_state,
        "pe_scale": float(pe_scale),
        "pe_block": bool(pe_block),
        "pe_note": f"{source} PE selected",
    }


# ============================================================
# 10) CAPEX MODULE
# ============================================================

def extract_capex_from_cashflow(ticker):
    """
    優先抓 yfinance quarterly_cashflow 的 Capital Expenditure。
    如果抓不到，用 Operating Cash Flow - Free Cash Flow 當 proxy。
    如果仍失敗，回傳空序列，後面會給 UNKNOWN fallback。
    """

    tk = yf.Ticker(ticker)

    qcf = pd.DataFrame()
    acf = pd.DataFrame()

    try:
        qcf = tk.quarterly_cashflow
    except Exception:
        qcf = pd.DataFrame()

    try:
        acf = tk.cashflow
    except Exception:
        acf = pd.DataFrame()

    capex_candidates = [
        "Capital Expenditure",
        "Capital Expenditures",
        "Capital Spending",
        "Purchase Of PPE",
        "Purchase of Property Plant And Equipment",
        "Purchase Of Property Plant Equipment",
        "Purchase Of Property Plant And Equipment",
    ]

    cfo_candidates = [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Cash Flow From Continuing Operating Activities",
    ]

    fcf_candidates = [
        "Free Cash Flow",
        "FreeCashFlow",
    ]

    capex_series = pd.Series(dtype=float)
    source = "NONE"

    # 1. quarterly direct capex
    if qcf is not None and not qcf.empty:
        capex_row = find_row_by_candidates(qcf, capex_candidates)

        if capex_row is not None:
            raw = qcf.loc[capex_row].copy()
            raw.index = pd.to_datetime(raw.index)
            capex_series = raw.sort_index().astype(float).abs()
            source = f"quarterly_cashflow:{capex_row}"

        # 2. quarterly proxy = CFO - FCF
        if capex_series.empty:
            cfo_row = find_row_by_candidates(qcf, cfo_candidates)
            fcf_row = find_row_by_candidates(qcf, fcf_candidates)

            if cfo_row is not None and fcf_row is not None:
                cfo = qcf.loc[cfo_row].copy()
                fcf = qcf.loc[fcf_row].copy()
                proxy = cfo - fcf
                proxy.index = pd.to_datetime(proxy.index)
                capex_series = proxy.sort_index().astype(float).abs()
                source = f"quarterly_proxy:{cfo_row}-{fcf_row}"

    # 3. annual direct fallback
    if capex_series.empty and acf is not None and not acf.empty:
        capex_row = find_row_by_candidates(acf, capex_candidates)

        if capex_row is not None:
            raw = acf.loc[capex_row].copy()
            raw.index = pd.to_datetime(raw.index)
            capex_series = raw.sort_index().astype(float).abs()
            source = f"annual_cashflow:{capex_row}"

    capex_series = capex_series.dropna()
    capex_series = capex_series[capex_series > 0]

    return capex_series, source


def compute_capex_signal(capex_series, rules):
    """
    QoQ-based CapEx signal.

    目的：
    不再強制依賴 TTM YoY。
    先用最近一季 QoQ + 最近幾季 slope 判斷 capex 是否減弱。

    判斷優先順序：
    1. 有 2 筆以上：可算 QoQ
    2. 有 3~4 筆以上：可算 recent slope
    3. 有 8 筆以上：附帶算 TTM YoY，僅作背景
    """

    if capex_series is None or len(capex_series.dropna()) < 2:
        return {
            "capex_quality": "UNKNOWN",
            "capex_qoq": np.nan,
            "capex_yoy": np.nan,
            "capex_ttm_latest": np.nan,
            "capex_slope_rel": np.nan,
            "capex_trend": "UNKNOWN",
            "capex_score": np.nan,
            "capex_table": pd.DataFrame(),
            "capex_note": "Capex data missing or fewer than 2 quarters",
        }

    s = capex_series.dropna().sort_index()
    s = s[s > 0]

    if len(s) < 2:
        return {
            "capex_quality": "UNKNOWN",
            "capex_qoq": np.nan,
            "capex_yoy": np.nan,
            "capex_ttm_latest": np.nan,
            "capex_slope_rel": np.nan,
            "capex_trend": "UNKNOWN",
            "capex_score": np.nan,
            "capex_table": pd.DataFrame({"capex": s}),
            "capex_note": "Capex data insufficient after cleaning",
        }

    latest_capex = _safe_float(s.iloc[-1])
    prev_capex = _safe_float(s.iloc[-2])

    if not pd.isna(prev_capex) and prev_capex > 0:
        capex_qoq = latest_capex / prev_capex - 1.0
    else:
        capex_qoq = np.nan

    # 最近幾季 slope：資料有 3 筆以上才算，避免 2 點斜率太容易誤判
    window = int(rules.get("CAPEX_TREND_WINDOW", 4))
    recent = s.tail(min(window, len(s)))
    slope_rel = linear_slope_rel(recent) if len(recent) >= 3 else np.nan

    up_slope = float(rules.get("CAPEX_UP_SLOPE_REL", 0.03))
    down_slope = float(rules.get("CAPEX_DOWN_SLOPE_REL", -0.03))

    if pd.isna(slope_rel):
        capex_trend = "UNKNOWN"
    elif slope_rel >= up_slope:
        capex_trend = "UP"
    elif slope_rel <= down_slope:
        capex_trend = "DOWN"
    else:
        capex_trend = "FLAT"

    # TTM / YoY 仍然算，但只當背景，不當必要條件
    ttm = s.rolling(4).sum().dropna()
    capex_ttm_latest = _safe_float(ttm.iloc[-1]) if len(ttm) >= 1 else np.nan

    if len(ttm) >= 5:
        capex_yoy = _safe_float(ttm.iloc[-1] / ttm.iloc[-5] - 1.0)
    else:
        capex_yoy = np.nan

    # QoQ threshold
    qoq_strong = float(rules.get("CAPEX_QOQ_STRONG", 0.05))
    qoq_weak = float(rules.get("CAPEX_QOQ_WEAK", -0.10))

    score_parts = []

    # 1. QoQ score，權重最高，因為比較即時
    if not pd.isna(capex_qoq):
        if capex_qoq >= qoq_strong:
            score_parts.append(("qoq", 1.0, 0.55))
        elif capex_qoq <= qoq_weak:
            score_parts.append(("qoq", -1.0, 0.55))
        else:
            score_parts.append(("qoq", 0.0, 0.55))

    # 2. Recent slope score，輔助判斷是不是連續減弱
    if capex_trend == "UP":
        score_parts.append(("slope", 1.0, 0.35))
    elif capex_trend == "DOWN":
        score_parts.append(("slope", -1.0, 0.35))
    elif capex_trend == "FLAT":
        score_parts.append(("slope", 0.0, 0.35))

    # 3. TTM YoY 只當背景，低權重
    if not pd.isna(capex_yoy):
        strong_yoy = float(rules.get("CAPEX_STRONG_YOY", 0.10))
        weak_yoy = float(rules.get("CAPEX_WEAK_YOY", -0.10))

        if capex_yoy >= strong_yoy:
            score_parts.append(("ttm_yoy", 1.0, 0.10))
        elif capex_yoy <= weak_yoy:
            score_parts.append(("ttm_yoy", -1.0, 0.10))
        else:
            score_parts.append(("ttm_yoy", 0.0, 0.10))

    if not score_parts:
        capex_score = np.nan
        quality = "UNKNOWN"
    else:
        total_w = sum(w for _, _, w in score_parts)
        capex_score = sum(score * w for _, score, w in score_parts) / total_w

        if capex_score >= 0.35:
            quality = "STRONG"
        elif capex_score <= -0.35:
            quality = "WEAK"
        else:
            quality = "STABLE"

    capex_table = pd.DataFrame({"capex": s})
    if len(ttm) > 0:
        capex_table["capex_ttm"] = ttm

    score_detail = ", ".join([
        f"{name}:{score:+.1f}x{weight:.2f}"
        for name, score, weight in score_parts
    ])

    return {
        "capex_quality": quality,
        "capex_qoq": capex_qoq,
        "capex_yoy": capex_yoy,
        "capex_ttm_latest": capex_ttm_latest,
        "capex_slope_rel": slope_rel,
        "capex_trend": capex_trend,
        "capex_score": capex_score,
        "capex_table": capex_table,
        "capex_note": (
            f"QoQ-based capex signal. "
            f"QoQ={pct_fmt(capex_qoq)}, "
            f"trend={capex_trend}, "
            f"score={capex_score if not pd.isna(capex_score) else 'NA'}, "
            f"detail=[{score_detail}]"
        ),
    }


# ============================================================
# 11) PEG VALUATION SYSTEM
# ============================================================

def fetch_yf_growth_info(ticker):
    out = {
        "earnings_growth": np.nan,
        "revenue_growth": np.nan,
        "source_ok": False,
        "error": "",
    }

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        out["earnings_growth"] = _safe_float(info.get("earningsGrowth", np.nan))
        out["revenue_growth"] = _safe_float(info.get("revenueGrowth", np.nan))
        out["source_ok"] = True

    except Exception as e:
        out["error"] = str(e)

    return out


def _cap_growth_for_peg(growth_rate, rules):
    if pd.isna(growth_rate):
        return np.nan

    min_g = float(rules.get("PEG_MIN_GROWTH", 0.05))
    max_g = float(rules.get("PEG_MAX_GROWTH", 0.30))

    if growth_rate < min_g:
        return np.nan

    return float(np.clip(growth_rate, min_g, max_g))


def compute_capex_adjusted_growth(raw_forward_eps_growth, capex_signal, rules):
    quality = capex_signal.get("capex_quality", "UNKNOWN")

    q_rules = rules.get("PEG_GROWTH_QUALITY_RULES", {})
    qr = q_rules.get(quality, q_rules.get("UNKNOWN", {
        "reliability": 0.45,
        "soft_upper": 0.20,
        "scale_bias": 0.80,
    }))

    reliability = float(qr.get("reliability", 0.45))
    soft_upper = float(qr.get("soft_upper", 0.20))
    scale_bias = float(qr.get("scale_bias", 0.80))

    min_adj = float(rules.get("PEG_MIN_ADJUSTED_GROWTH", 0.05))
    unknown_fallback = float(rules.get("PEG_UNKNOWN_FALLBACK_GROWTH", 0.12))

    if pd.isna(raw_forward_eps_growth) or raw_forward_eps_growth <= 0:
        adjusted_growth = unknown_fallback
        note = "Raw forward EPS growth missing; using conservative fallback"
    else:
        adjusted_growth = soft_saturate_growth(
            raw_growth=raw_forward_eps_growth,
            reliability=reliability,
            soft_upper=soft_upper,
            min_growth=min_adj
        )
        note = f"Capex-adjusted growth by quality={quality}"

    return {
        "adjusted_growth_rate": float(adjusted_growth),
        "growth_quality": quality,
        "growth_reliability": reliability,
        "growth_soft_upper": soft_upper,
        "growth_scale_bias": scale_bias,
        "growth_note": note,
    }


def compute_peg_pack(ticker, pe_pack, capex_signal, rules):
    if not rules.get("PEG_ENABLE", True):
        return {
            "peg_enabled": False,
            "peg": np.nan,
            "raw_growth_rate": np.nan,
            "used_growth_rate": np.nan,
            "raw_growth_pct": np.nan,
            "used_growth_pct": np.nan,
            "growth_source": "DISABLED",
            "growth_quality": "DISABLED",
            "growth_reliability": np.nan,
            "growth_soft_upper": np.nan,
            "capex_quality": "DISABLED",
            "peg_state": "DISABLED",
            "peg_scale": 1.00,
            "peg_block": False,
            "peg_note": "PEG system disabled",
        }

    selected_pe = _safe_float(pe_pack.get("selected_pe", np.nan))
    forward_eps = _safe_float(pe_pack.get("forward_eps", np.nan))
    trailing_eps = _safe_float(pe_pack.get("trailing_eps", np.nan))

    yf_growth_info = fetch_yf_growth_info(ticker)
    yf_earnings_growth = _safe_float(yf_growth_info.get("earnings_growth", np.nan))
    manual_growth = _safe_float(rules.get("MANUAL_EPS_GROWTH", np.nan))

    raw_forward_eps_growth = np.nan
    if not pd.isna(forward_eps) and not pd.isna(trailing_eps) and trailing_eps > 0:
        raw_forward_eps_growth = forward_eps / trailing_eps - 1.0

    source = str(rules.get("PEG_GROWTH_SOURCE", "CAPEX_ADJUSTED")).upper()
    chosen_source = source

    raw_growth_rate = np.nan
    used_growth_rate = np.nan

    growth_quality = capex_signal.get("capex_quality", "UNKNOWN")
    growth_reliability = np.nan
    growth_soft_upper = np.nan
    growth_scale_bias = 1.00
    growth_note = ""

    # ------------------------------------------------------------
    # 主模式：CAPEX_ADJUSTED
    # ------------------------------------------------------------
    if source == "CAPEX_ADJUSTED":
        raw_growth_rate = raw_forward_eps_growth

        capex_adj = compute_capex_adjusted_growth(
            raw_forward_eps_growth=raw_forward_eps_growth,
            capex_signal=capex_signal,
            rules=rules,
        )

        used_growth_rate = capex_adj["adjusted_growth_rate"]
        growth_quality = capex_adj["growth_quality"]
        growth_reliability = capex_adj["growth_reliability"]
        growth_soft_upper = capex_adj["growth_soft_upper"]
        growth_scale_bias = capex_adj["growth_scale_bias"]
        growth_note = capex_adj["growth_note"]
        chosen_source = "CAPEX_ADJUSTED"

    # ------------------------------------------------------------
    # 舊版 fallback / alternative modes
    # ------------------------------------------------------------
    elif source == "CAPPED_FORWARD_EPS":
        raw_growth_rate = raw_forward_eps_growth
        used_growth_rate = _cap_growth_for_peg(raw_forward_eps_growth, rules)
        chosen_source = "CAPPED_FORWARD_EPS"
        growth_note = "Old capped forward EPS growth"

    elif source == "MANUAL":
        raw_growth_rate = manual_growth
        used_growth_rate = _cap_growth_for_peg(manual_growth, rules)
        chosen_source = "MANUAL"
        growth_note = "Manual EPS growth"

    elif source == "YF_EARNINGS_GROWTH":
        raw_growth_rate = yf_earnings_growth
        used_growth_rate = _cap_growth_for_peg(yf_earnings_growth, rules)
        chosen_source = "YF_EARNINGS_GROWTH"
        growth_note = "YFinance earningsGrowth"

    elif source == "BLENDED":
        candidates = []

        capped_forward = _cap_growth_for_peg(raw_forward_eps_growth, rules)
        capped_yf = _cap_growth_for_peg(yf_earnings_growth, rules)
        capped_manual = _cap_growth_for_peg(manual_growth, rules)

        for x in [capped_forward, capped_yf, capped_manual]:
            if not pd.isna(x):
                candidates.append(x)

        used_growth_rate = float(np.mean(candidates)) if candidates else np.nan

        raw_candidates = [x for x in [raw_forward_eps_growth, yf_earnings_growth, manual_growth] if not pd.isna(x)]
        raw_growth_rate = float(np.mean(raw_candidates)) if raw_candidates else np.nan

        chosen_source = "BLENDED"
        growth_note = "Blended capped growth"

    else:
        raw_growth_rate = raw_forward_eps_growth
        used_growth_rate = _cap_growth_for_peg(raw_forward_eps_growth, rules)
        chosen_source = "CAPPED_FORWARD_EPS_FALLBACK"
        growth_note = "Unknown source; fallback to capped forward EPS"

    # ------------------------------------------------------------
    # 如果主模式 CAPEX_ADJUSTED 還是失敗， fallback 到舊版 capped PEG
    # ------------------------------------------------------------
    if pd.isna(used_growth_rate) or used_growth_rate <= 0:
        fallback = str(rules.get("PEG_FALLBACK_SOURCE", "CAPPED_FORWARD_EPS")).upper()

        if fallback == "CAPPED_FORWARD_EPS":
            fb_growth = _cap_growth_for_peg(raw_forward_eps_growth, rules)
            if not pd.isna(fb_growth):
                used_growth_rate = fb_growth
                chosen_source = "FALLBACK_CAPPED_FORWARD_EPS"
                growth_note = "Capex-adjusted failed; fallback to old capped PEG"

        if pd.isna(used_growth_rate) or used_growth_rate <= 0:
            used_growth_rate = float(rules.get("PEG_UNKNOWN_FALLBACK_GROWTH", 0.12))
            chosen_source = "UNKNOWN_FALLBACK"
            growth_note = "All PEG growth sources failed; using conservative fallback"

    # ------------------------------------------------------------
    # PE missing
    # ------------------------------------------------------------
    if pd.isna(selected_pe) or selected_pe <= 0:
        return {
            "peg_enabled": True,
            "peg": np.nan,
            "raw_growth_rate": raw_growth_rate,
            "used_growth_rate": used_growth_rate,
            "raw_growth_pct": raw_growth_rate * 100 if not pd.isna(raw_growth_rate) else np.nan,
            "used_growth_pct": used_growth_rate * 100 if not pd.isna(used_growth_rate) else np.nan,
            "forward_eps_growth": raw_forward_eps_growth,
            "yf_earnings_growth": yf_earnings_growth,
            "manual_growth": manual_growth,
            "growth_source": chosen_source,
            "growth_quality": growth_quality,
            "growth_reliability": growth_reliability,
            "growth_soft_upper": growth_soft_upper,
            "capex_quality": capex_signal.get("capex_quality", "UNKNOWN"),
            "peg_state": "PE_MISSING",
            "peg_scale": float(rules.get("PEG_MISSING_SCALE", 1.0)),
            "peg_block": False,
            "peg_note": "Cannot compute PEG because PE is missing",
        }

    used_growth_pct = used_growth_rate * 100.0
    raw_growth_pct = raw_growth_rate * 100.0 if not pd.isna(raw_growth_rate) else np.nan

    peg = selected_pe / used_growth_pct

    cheap = float(rules.get("PEG_CHEAP", 0.80))
    fair = float(rules.get("PEG_FAIR", 1.20))
    warm = float(rules.get("PEG_WARM", 1.80))
    expensive = float(rules.get("PEG_EXPENSIVE", 2.50))
    hard_block = float(rules.get("PEG_HARD_BLOCK", 3.50))

    peg_block = False

    if peg <= cheap:
        peg_state = "CHEAP"
        peg_scale = float(rules.get("PEG_SCALE_CHEAP", 1.20))
    elif peg <= fair:
        peg_state = "FAIR"
        peg_scale = float(rules.get("PEG_SCALE_FAIR", 1.00))
    elif peg <= warm:
        peg_state = "WARM"
        peg_scale = float(rules.get("PEG_SCALE_WARM", 0.70))
    elif peg <= expensive:
        peg_state = "EXPENSIVE"
        peg_scale = float(rules.get("PEG_SCALE_EXPENSIVE", 0.40))
    elif peg <= hard_block:
        peg_state = "VERY_EXPENSIVE"
        peg_scale = float(rules.get("PEG_SCALE_VERY_EXPENSIVE", 0.15))
    else:
        peg_state = "EXTREME"

        if rules.get("PEG_HARD_BLOCK_ENABLE", False):
            peg_scale = 0.0
            peg_block = True
        else:
            peg_scale = float(rules.get("PEG_SCALE_VERY_EXPENSIVE", 0.15))

    # capex weak 時額外 bias，避免 PEG scale 太樂觀
    peg_scale = peg_scale * float(growth_scale_bias)

    return {
        "peg_enabled": True,
        "peg": float(peg),
        "raw_growth_rate": raw_growth_rate,
        "used_growth_rate": float(used_growth_rate),
        "raw_growth_pct": raw_growth_pct,
        "used_growth_pct": float(used_growth_pct),
        "forward_eps_growth": raw_forward_eps_growth,
        "yf_earnings_growth": yf_earnings_growth,
        "manual_growth": manual_growth,
        "growth_source": chosen_source,
        "growth_quality": growth_quality,
        "growth_reliability": growth_reliability,
        "growth_soft_upper": growth_soft_upper,
        "growth_scale_bias": growth_scale_bias,
        "capex_quality": capex_signal.get("capex_quality", "UNKNOWN"),
        "peg_state": peg_state,
        "peg_scale": float(peg_scale),
        "peg_block": bool(peg_block),
        "peg_note": f"{growth_note}; PEG = PE / adjusted growth%",
    }


def combine_valuation_scale(pe_scale, peg_scale, rules):
    mode = str(rules.get("VALUATION_COMBINE_MODE", "MIN")).upper()

    pe_scale = float(pe_scale)
    peg_scale = float(peg_scale)

    if mode == "PRODUCT":
        return pe_scale * peg_scale

    if mode == "AVERAGE":
        return (pe_scale + peg_scale) / 2.0

    return min(pe_scale, peg_scale)


# ============================================================
# 12) RISK MATH
# ============================================================

def ann_vol(daily_ret: pd.Series, lookback=60):
    s = daily_ret.dropna()
    if len(s) < 30:
        raise ValueError("報酬序列太短，無法估算年化波動。")
    w = s.iloc[-min(len(s), lookback):]
    return float(w.std(ddof=1) * np.sqrt(252))


def equity_curve(daily_ret: pd.Series, base=1.0):
    return base * (1.0 + daily_ret).cumprod()


def drawdown_now(daily_ret: pd.Series):
    eq = equity_curve(daily_ret)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.iloc[-1])


def estimate_beta(asset_ret: pd.Series, bench="^NDX", lookback_days=260):
    raw = yf.download(bench, period=f"{lookback_days}d", interval="1d", auto_adjust=True, progress=False)

    if raw is None or raw.empty:
        raise ValueError("benchmark download failed")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]

    b = close.pct_change().dropna()
    b.name = "bench"

    a = asset_ret.copy()
    a.name = "asset"

    df = pd.concat([a, b], axis=1).dropna()

    if len(df) < 60:
        raise ValueError(f"beta sample too short: {len(df)}")

    cov = float(np.cov(df["asset"].values, df["bench"].values, ddof=1)[0, 1])
    var = float(np.var(df["bench"].values, ddof=1))

    if var <= 1e-12:
        raise ValueError("benchmark variance zero")

    beta = cov / var
    return float(np.clip(beta, 0.2, 3.0))


# ============================================================
# 13) TSM HOLDINGS TABLE
# ============================================================

def compute_tsm_holdings(tsm_price):
    market_value = TSM_SHARES * tsm_price
    cost_value = TSM_SHARES * TSM_AVG_COST
    unreal_pnl = market_value - cost_value
    unreal_pnl_pct = unreal_pnl / cost_value if cost_value > 0 else np.nan

    stock_ratio = market_value / EQUITY_USD if EQUITY_USD > 0 else np.nan
    cash_ratio = FREE_CASH_USD / EQUITY_USD if EQUITY_USD > 0 else np.nan

    df = pd.DataFrame([{
        "ticker": CORE_TICKER,
        "shares": TSM_SHARES,
        "price": tsm_price,
        "market_value": market_value,
        "avg_cost": TSM_AVG_COST,
        "cost_value": cost_value,
        "unreal_pnl": unreal_pnl,
        "unreal_pnl_pct": unreal_pnl_pct,
        "stock_ratio": stock_ratio,
        "cash_ratio": cash_ratio,
    }]).set_index("ticker")

    return df


# ============================================================
# 14) LEVERAGE ENGINE
# Advisory only
# ============================================================

def dd_governor(L_raw, dd_now):
    L = float(L_raw)

    for wall, cap in DD_WALLS:
        if dd_now < wall:
            L = min(L, cap)
            break

    return float(max(L, 0.0))


def leverage_from_cash_or_margin(equity_usd, free_cash_usd, current_stock_usd):
    if equity_usd <= 0:
        return 1.0

    if ALLOW_MARGIN_BUY:
        return float(MAX_MANUAL_GROSS_LEVERAGE)

    max_stock_usd = current_stock_usd + max(0.0, free_cash_usd) * HAIRCUT
    return float(max_stock_usd / equity_usd)


def leverage_engine(sigma_ann, dd_now, regime, crisis_on, limits, equity_usd, free_cash_usd, current_stock_usd):
    if sigma_ann and sigma_ann > 1e-9:
        L_vol = SIGMA_TARGET / sigma_ann
    else:
        L_vol = 0.0

    L_vol = max(0.0, float(L_vol))
    L_dd = dd_governor(L_vol, dd_now)

    if crisis_on or regime == "CRISIS":
        L_reg = 0.0
    elif regime == "DATA-UNSTABLE":
        L_reg = min(L_dd, DATA_UNSTABLE_MAX_LEV)
    else:
        L_reg = min(L_dd, float(limits.get("max_leverage", 1.0)))

    L_buying_power = leverage_from_cash_or_margin(
        equity_usd=equity_usd,
        free_cash_usd=free_cash_usd,
        current_stock_usd=current_stock_usd,
    )

    L_final = min(L_reg, L_buying_power)

    return {
        "L_vol": float(L_vol),
        "L_dd": float(L_dd),
        "L_reg": float(L_reg),
        "L_buying_power": float(L_buying_power),
        "L_final": float(max(L_final, 0.0)),
    }


# ============================================================
# 15) TSM BUY PLAN
# ============================================================

def build_tsm_buy_plan(tsm_risk, holdings, lev_pack, res_macro):
    px = tsm_risk["last_price"]

    trend_state = tsm_risk["trend_state"]
    trend_scale = float(tsm_risk["trend_scale"])

    pe_pack = compute_pe_pack(CORE_TICKER, px, RISK_RULES)
    pe_scale = float(pe_pack["pe_scale"])
    pe_state = pe_pack["pe_state"]

    capex_series, capex_source = extract_capex_from_cashflow(CORE_TICKER)
    capex_signal = compute_capex_signal(capex_series, RISK_RULES)
    capex_signal["capex_source"] = capex_source

    peg_pack = compute_peg_pack(CORE_TICKER, pe_pack, capex_signal, RISK_RULES)
    peg_scale = float(peg_pack["peg_scale"])
    peg_state = peg_pack["peg_state"]

    valuation_scale = combine_valuation_scale(pe_scale, peg_scale, RISK_RULES)

    dd_120 = tsm_risk["dd_120"]
    atr_pct = tsm_risk["atr_pct"]

    regime = res_macro["regime"]
    crisis_on = res_macro["crisis_on"]
    disloc = res_macro["latest"].get("Dislocation", np.nan)
    disloc_thr = res_macro["disloc_thr"]

    current_tsm_usd = float(holdings.loc[CORE_TICKER, "market_value"])
    stock_ratio = current_tsm_usd / EQUITY_USD
    cash_ratio = FREE_CASH_USD / EQUITY_USD

    max_cash_buy_usd = max(0.0, FREE_CASH_USD * HAIRCUT) * MAX_DAILY_CASH_USE_FRAC

    if ALLOW_MARGIN_BUY:
        gross_cap_usd = EQUITY_USD * MAX_MANUAL_GROSS_LEVERAGE
        max_buy_usd = max(0.0, gross_cap_usd - current_tsm_usd)
    else:
        max_buy_usd = max_cash_buy_usd

    target_tsm_usd = current_tsm_usd + max_buy_usd
    raw_delta = max_buy_usd

    reasons = []

    macro_hard_block_enable = RISK_RULES.get("MACRO_HARD_BLOCK_ENABLE", True)

    stress = (
        macro_hard_block_enable
        and (
            crisis_on
            or regime in ["CRISIS", "RISK-OFF"]
            or (not pd.isna(disloc) and disloc > disloc_thr)
        )
    )

    if stress:
        reasons.append("Macro stress / RISK-OFF / Dislocation：禁止買進")

    if regime == "DATA-UNSTABLE":
        reasons.append("DATA-UNSTABLE：只提示資料品質，不阻止現金買進")

    if trend_scale <= 0:
        reasons.append(f"Trend state = {trend_state}，trend_scale=0，暫停買進")

    if RISK_RULES.get("NO_BUY_BELOW_MA200", False) and not tsm_risk["above_ma200"]:
        reasons.append("TSM below MA200：NO_BUY_BELOW_MA200=True，禁止買進")

    if atr_pct > RISK_RULES.get("NO_BUY_IF_ATR_ABOVE", 999):
        reasons.append(f"ATR% 過高：{atr_pct:.2%}，不追價")

    if dd_120 < RISK_RULES.get("NO_BUY_IF_DD_BELOW", -999):
        reasons.append(f"120D DD 過深：{dd_120:.2%}，避免接刀")

    if pe_pack.get("pe_block", False):
        reasons.append(f"PE 過高硬擋：PE={pe_pack['selected_pe']:.2f}, state={pe_state}")

    if peg_pack.get("peg_block", False):
        reasons.append(f"PEG 過高硬擋：PEG={peg_pack['peg']:.2f}, state={peg_state}")

    block_reasons = [
        r for r in reasons
        if not r.startswith("DATA-UNSTABLE")
    ]

    buy_allowed = len(block_reasons) == 0

    if not buy_allowed:
        suggested_buy_usd = 0.0
        action = "HOLD"

    elif raw_delta <= MIN_TRADE_USD:
        suggested_buy_usd = 0.0
        action = "HOLD"
        reasons.append("沒有足夠可用現金 / margin 額度可買")

    else:
        suggested_buy_usd = raw_delta * trend_scale * valuation_scale

        reasons.append(f"MA trend scale applied：{trend_state} x{trend_scale:.2f}")
        reasons.append(f"PE valuation scale applied：{pe_state} x{pe_scale:.2f}")
        reasons.append(f"PEG valuation scale applied：{peg_state} x{peg_scale:.2f}")
        reasons.append(f"Final valuation scale：x{valuation_scale:.2f}")

        if RISK_RULES.get("ADD_ENABLE", True) and dd_120 <= -float(RISK_RULES["ADD_DD"]):
            add_cap = raw_delta * float(RISK_RULES["ADD_BUY_FRAC_OF_CASH"])
            suggested_buy_usd = min(suggested_buy_usd, add_cap)
            reasons.append(f"回撤加碼模式：120D DD={dd_120:.2%}，只用可買金額的一部分")

        suggested_buy_usd = min(suggested_buy_usd, max_buy_usd)

        if suggested_buy_usd < MIN_TRADE_USD:
            suggested_buy_usd = 0.0
            action = "HOLD"
            reasons.append("可買金額低於 MIN_TRADE_USD")
        else:
            action = "BUY"

    shares = shares_from_usd(suggested_buy_usd, px) if suggested_buy_usd > 0 else np.nan

    # 如果計算後買不到有效股數，直接轉 HOLD，避免 BUY 0 shares / shares_to_buy = 0.0
    if pd.isna(shares) or shares <= 0:
        shares = np.nan
        est_usd = 0.0

        if action == "BUY":
            action = "HOLD"
            reasons.append("計算後股數為 0，取消買進")
    else:
        est_usd = shares * px

    est_new_shares = TSM_SHARES + (shares if not pd.isna(shares) else 0)
    est_new_avg_cost = (
        (TSM_SHARES * TSM_AVG_COST + est_usd) / est_new_shares
        if est_new_shares > 0 else np.nan
    )

    return {
        "action": action,
        "price": px,
        "shares_to_buy": shares,
        "est_buy_usd": est_usd,
        "est_new_shares": est_new_shares,
        "est_new_avg_cost": est_new_avg_cost,
        "target_tsm_usd": target_tsm_usd,
        "current_tsm_usd": current_tsm_usd,
        "raw_delta": raw_delta,
        "max_buy_usd": max_buy_usd,
        "stock_ratio": stock_ratio,
        "cash_ratio": cash_ratio,
        "buy_allowed": buy_allowed,

        "trend_state": trend_state,
        "trend_scale": trend_scale,

        "pe_pack": pe_pack,
        "pe_state": pe_state,
        "pe_scale": pe_scale,
        "selected_pe": pe_pack["selected_pe"],
        "forward_pe": pe_pack["forward_pe"],
        "trailing_pe": pe_pack["trailing_pe"],
        "forward_eps": pe_pack["forward_eps"],
        "trailing_eps": pe_pack["trailing_eps"],

        "capex_series": capex_series,
        "capex_signal": capex_signal,
        "capex_source": capex_source,

        "peg_pack": peg_pack,
        "peg_state": peg_state,
        "peg_scale": peg_scale,
        "peg": peg_pack["peg"],
        "raw_growth_pct": peg_pack["raw_growth_pct"],
        "used_growth_pct": peg_pack["used_growth_pct"],
        "growth_source": peg_pack["growth_source"],
        "growth_quality": peg_pack["growth_quality"],
        "growth_reliability": peg_pack["growth_reliability"],
        "growth_soft_upper": peg_pack["growth_soft_upper"],

        "valuation_scale": valuation_scale,

        "advisory_L_final": lev_pack["L_final"],
        "reasons": reasons,
    }


# ============================================================
# 16) RENDER
# ============================================================

# Original notebook renderer output. In Streamlit, ipywidgets may be unavailable,
# so never create widgets at import time unless the package exists.
out = widgets.Output() if widgets is not None else None

def render_all(res_macro, tsm_risk, risk_pack, lev_pack, holdings, buy_plan):
    if out is None or display is None or HTML is None:
        return

    out.clear_output()

    with out:
        regime = res_macro["regime"]
        crisis_on = res_macro["crisis_on"]
        latest = res_macro["latest"]
        disloc = latest.get("Dislocation", np.nan)

        action = buy_plan["action"]

        header = (
            badge(f"REGIME: {regime}", "#1565c0")
            + badge(f"CRISIS: {'ON' if crisis_on else 'OFF'}", "#c62828" if crisis_on else "#2e7d32")
            + badge(f"ACTION: {action}", "#2e7d32" if action == "BUY" else "#455a64")
        )
        display(HTML(header))

        # ====================================================
        # Executive decision panel - 最重要的先看這裡
        # ====================================================
        valid_trade = (
            buy_plan["action"] == "BUY"
            and not pd.isna(buy_plan["shares_to_buy"])
            and buy_plan["shares_to_buy"] > 0
        )

        trade_title = f"BUY {fmt_shares(buy_plan['shares_to_buy'])} share" if valid_trade else "HOLD"
        trade_color = "#1b5e20" if valid_trade else "#37474f"
        unused_cash = max(0.0, buy_plan["max_buy_usd"] - buy_plan["est_buy_usd"])

        decision_html = f"""
        <div style='margin-top:10px;margin-bottom:14px;padding:16px 18px;border-radius:16px;
                    background:#111827;color:white;box-shadow:0 2px 10px rgba(0,0,0,0.12);'>
            <div style='font-size:12px;letter-spacing:1px;color:#9ca3af;font-weight:700;'>TSM CASH BUY DECISION</div>
            <div style='display:flex;flex-wrap:wrap;gap:18px;align-items:center;margin-top:8px;'>
                <div style='font-size:34px;font-weight:900;color:#ffffff;'>{trade_title}</div>
                <div style='padding:8px 12px;border-radius:999px;background:{trade_color};font-weight:800;'>ACTION: {buy_plan['action']}</div>
                <div style='font-size:15px;color:#d1d5db;'>Price <b>${buy_plan['price']:,.2f}</b></div>
                <div style='font-size:15px;color:#d1d5db;'>Est Buy <b>{fmt_usd_int(buy_plan['est_buy_usd'])}</b></div>
                <div style='font-size:15px;color:#d1d5db;'>Unused Cash <b>{fmt_usd_int(unused_cash)}</b></div>
                <div style='font-size:15px;color:#d1d5db;'>Valuation Scale <b>x{buy_plan['valuation_scale']:.2f}</b></div>
            </div>
        </div>
        """
        display(HTML(decision_html))

        # Signal summary strip
        signal_html = f"""
        <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:12px;'>
            <div style='background:#eef2ff;border-radius:12px;padding:10px;'><div style='font-size:12px;color:#475569;'>Trend</div><div style='font-size:20px;font-weight:800;color:#111827;'>{buy_plan['trend_state']} x{buy_plan['trend_scale']:.2f}</div></div>
            <div style='background:#ecfdf5;border-radius:12px;padding:10px;'><div style='font-size:12px;color:#475569;'>PE</div><div style='font-size:20px;font-weight:800;color:#111827;'>{buy_plan['pe_state']} x{buy_plan['pe_scale']:.2f}</div></div>
            <div style='background:#fff7ed;border-radius:12px;padding:10px;'><div style='font-size:12px;color:#475569;'>PEG</div><div style='font-size:20px;font-weight:800;color:#111827;'>{buy_plan['peg_state']} x{buy_plan['peg_scale']:.2f}</div></div>
            <div style='background:#f0fdf4;border-radius:12px;padding:10px;'><div style='font-size:12px;color:#475569;'>CapEx</div><div style='font-size:20px;font-weight:800;color:#111827;'>{buy_plan['growth_quality']} / {buy_plan['capex_signal'].get('capex_trend','UNKNOWN')}</div></div>
            <div style='background:#f8fafc;border-radius:12px;padding:10px;'><div style='font-size:12px;color:#475569;'>Macro</div><div style='font-size:20px;font-weight:800;color:#111827;'>{regime}</div></div>
        </div>
        """
        display(HTML(signal_html))

        # Macro cards
        cards = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;'>"
        for k in ["Liquidity", "Credit", "Volatility", "Growth", "Rate", "Geo", "TotalScore", "Coverage", "Dislocation"]:
            val = latest.get(k, np.nan)

            if k == "Coverage":
                disp = f"{val * 100:.0f}%" if not pd.isna(val) else "NA"
                color = "#1565c0" if (not pd.isna(val) and val >= 0.8) else ("#f9a825" if (not pd.isna(val) and val >= 0.6) else "#c62828")
                cards += card(k, disp, color)

            elif k in ["Credit", "Volatility", "Dislocation"]:
                cards += card(k, pretty(val), tone_stress(val))

            else:
                if pd.isna(val):
                    c = "#455a64"
                elif val >= 0.8:
                    c = "#2e7d32"
                elif val <= -0.8:
                    c = "#c62828"
                else:
                    c = "#f9a825"
                cards += card(k, pretty(val), c)

        cards += "</div>"
        display(HTML(cards))

        # TSM cards
        trend_color = (
            "#2e7d32" if tsm_risk["trend_scale"] >= 1.0
            else ("#f9a825" if tsm_risk["trend_scale"] > 0 else "#c62828")
        )

        tsm_cards = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;'>"
        tsm_cards += card("TSM Price", f"{tsm_risk['last_price']:.2f}", "#37474f")
        tsm_cards += card("MA60", f"{tsm_risk['ma60']:.2f}", "#37474f")
        tsm_cards += card("MA200", f"{tsm_risk['ma200']:.2f}", "#37474f")
        tsm_cards += card("Trend State", f"{tsm_risk['trend_state']} x{tsm_risk['trend_scale']:.2f}", trend_color)
        tsm_cards += card("120D DD", f"{tsm_risk['dd_120'] * 100:.1f}%", "#c62828" if tsm_risk["dd_120"] < -0.15 else "#f9a825")
        tsm_cards += card("ATR%", f"{tsm_risk['atr_pct'] * 100:.2f}%", "#37474f")
        tsm_cards += card(
            "PnL%",
            "NA" if pd.isna(tsm_risk["pnl_pct"]) else f"{tsm_risk['pnl_pct'] * 100:.1f}%",
            "#2e7d32" if not pd.isna(tsm_risk["pnl_pct"]) and tsm_risk["pnl_pct"] > 0 else "#c62828"
        )
        tsm_cards += "</div>"
        display(HTML(tsm_cards))

        # Account cards
        account_cards = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;'>"
        account_cards += card("Equity", f"{EQUITY_USD:,.0f}", "#37474f")
        account_cards += card("Free Cash", f"{FREE_CASH_USD:,.0f}", "#37474f")
        account_cards += card("Cash Ratio", f"{buy_plan['cash_ratio']:.2%}", "#37474f")
        account_cards += card("TSM Ratio", f"{buy_plan['stock_ratio']:.2%}", "#37474f")
        account_cards += card("L_final Advisory", f"{lev_pack['L_final']:.2f}x", "#455a64")
        account_cards += "</div>"
        display(HTML(account_cards))

        # PE cards
        pe_color = (
            "#2e7d32" if buy_plan["pe_scale"] >= 1.0
            else ("#f9a825" if buy_plan["pe_scale"] > 0.30 else "#c62828")
        )

        pe_cards = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;'>"
        pe_cards += card("Selected PE", "NA" if pd.isna(buy_plan["selected_pe"]) else f"{buy_plan['selected_pe']:.2f}", pe_color)
        pe_cards += card("PE State", f"{buy_plan['pe_state']} x{buy_plan['pe_scale']:.2f}", pe_color)
        pe_cards += card("Forward PE", "NA" if pd.isna(buy_plan["forward_pe"]) else f"{buy_plan['forward_pe']:.2f}", "#37474f")
        pe_cards += card("Trailing PE", "NA" if pd.isna(buy_plan["trailing_pe"]) else f"{buy_plan['trailing_pe']:.2f}", "#37474f")
        pe_cards += "</div>"
        display(HTML(pe_cards))

        # PEG cards
        peg_color = (
            "#2e7d32" if buy_plan["peg_scale"] >= 1.0
            else ("#f9a825" if buy_plan["peg_scale"] > 0.30 else "#c62828")
        )

        peg_cards = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;'>"
        peg_cards += card("Normalized PEG", "NA" if pd.isna(buy_plan["peg"]) else f"{buy_plan['peg']:.2f}", peg_color)
        peg_cards += card("PEG State", f"{buy_plan['peg_state']} x{buy_plan['peg_scale']:.2f}", peg_color)
        peg_cards += card("Raw EPS Growth", "NA" if pd.isna(buy_plan["raw_growth_pct"]) else f"{buy_plan['raw_growth_pct']:.1f}%", "#37474f")
        peg_cards += card("Used PEG Growth", "NA" if pd.isna(buy_plan["used_growth_pct"]) else f"{buy_plan['used_growth_pct']:.1f}%", "#37474f")
        peg_cards += card("Growth Quality", f"{buy_plan['growth_quality']}", "#37474f")
        peg_cards += card("Valuation Scale", f"x{buy_plan['valuation_scale']:.2f}", peg_color)
        peg_cards += "</div>"
        display(HTML(peg_cards))

        # Capex cards
        capex_signal = buy_plan["capex_signal"]

        capex_quality = capex_signal.get("capex_quality", "UNKNOWN")
        capex_color = (
            "#2e7d32" if capex_quality == "STRONG"
            else ("#f9a825" if capex_quality == "STABLE" else "#c62828" if capex_quality == "WEAK" else "#455a64")
        )

        capex_cards = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;'>"
        capex_cards += card("Capex Quality", capex_quality, capex_color)
        capex_cards += card("Capex Trend", capex_signal.get("capex_trend", "UNKNOWN"), capex_color)
        capex_cards += card("Capex QoQ", pct_fmt(capex_signal.get("capex_qoq", np.nan)), capex_color)
        capex_cards += card("Capex YoY", pct_fmt(capex_signal.get("capex_yoy", np.nan)), capex_color)
        capex_cards += card("Capex Slope", pct_fmt(capex_signal.get("capex_slope_rel", np.nan)), capex_color)
        capex_cards += card("Capex Score", pretty(capex_signal.get("capex_score", np.nan)), capex_color)
        capex_cards += card("Capex Source", buy_plan.get("capex_source", "NONE"), "#37474f")
        capex_cards += "</div>"
        display(HTML(capex_cards))

        print("\n=== Holdings ===")
        ht = holdings.copy()
        ht["price"] = ht["price"].map(lambda x: round(float(x), 2))
        ht["market_value"] = ht["market_value"].map(lambda x: round(float(x), 0))
        ht["avg_cost"] = ht["avg_cost"].map(lambda x: round(float(x), 2))
        ht["unreal_pnl"] = ht["unreal_pnl"].map(lambda x: round(float(x), 0))
        ht["unreal_pnl_pct"] = ht["unreal_pnl_pct"].map(lambda x: round(float(x) * 100, 2))
        ht["stock_ratio"] = ht["stock_ratio"].map(lambda x: round(float(x) * 100, 2))
        ht["cash_ratio"] = ht["cash_ratio"].map(lambda x: round(float(x) * 100, 2))
        display(ht)

        print("\n=== Macro Triggers ===")
        for k, v in res_macro["crisis_flags"].items():
            print(f" - {k}: {v}")
        print(f"VIX latest: {res_macro.get('vix_latest', np.nan)}")
        print(f"Dislocation: {disloc:+.2f} / threshold {res_macro['disloc_thr']}")

        print("\n=== Regime limits ===")
        for k, v in res_macro["limits"].items():
            if k != "notes":
                print(f"{k}: {v}")
        for n in res_macro["limits"].get("notes", []):
            print(f"- {n}")

        print("\n=== TSM Risk ===")
        print(f"Trend State : {tsm_risk['trend_state']}")
        print(f"Trend Scale : {tsm_risk['trend_scale']:.2f}")
        print(f"Above MA60  : {tsm_risk['above_ma60']}")
        print(f"Above MA200 : {tsm_risk['above_ma200']}")
        print(f"MA60 > MA200: {tsm_risk['ma60_above_ma200']}")
        print(f"Ann vol 60D : {risk_pack['sigma_ann']:.2%}")
        print(f"DD now      : {risk_pack['dd_now']:.2%}")
        print(f"Beta ^NDX   : {risk_pack['beta']:.2f}")

        print("\n=== PE Valuation ===")
        print(f"PE Source       : {buy_plan['pe_pack']['pe_source']}")
        print(f"PE State        : {buy_plan['pe_state']}")
        print(f"PE Scale        : {buy_plan['pe_scale']:.2f}")
        print(f"Selected PE     : {buy_plan['selected_pe'] if not pd.isna(buy_plan['selected_pe']) else 'NA'}")
        print(f"Forward PE      : {buy_plan['forward_pe'] if not pd.isna(buy_plan['forward_pe']) else 'NA'}")
        print(f"Trailing PE     : {buy_plan['trailing_pe'] if not pd.isna(buy_plan['trailing_pe']) else 'NA'}")
        print(f"Forward EPS     : {buy_plan['forward_eps'] if not pd.isna(buy_plan['forward_eps']) else 'NA'}")
        print(f"Trailing EPS    : {buy_plan['trailing_eps'] if not pd.isna(buy_plan['trailing_eps']) else 'NA'}")
        print(f"PE Note         : {buy_plan['pe_pack']['pe_note']}")

        print("\n=== Capex-adjusted PEG Valuation ===")
        print(f"PEG Source          : {buy_plan['growth_source']}")
        print(f"Growth Quality      : {buy_plan['growth_quality']}")
        print(f"Growth Reliability  : {buy_plan['growth_reliability'] if not pd.isna(buy_plan['growth_reliability']) else 'NA'}")
        print(f"Growth Soft Upper   : {buy_plan['growth_soft_upper'] if not pd.isna(buy_plan['growth_soft_upper']) else 'NA'}")
        print(f"PEG State           : {buy_plan['peg_state']}")
        print(f"PEG Scale           : {buy_plan['peg_scale']:.2f}")
        print(f"PEG                 : {buy_plan['peg'] if not pd.isna(buy_plan['peg']) else 'NA'}")
        print(f"Raw EPS Growth      : {buy_plan['raw_growth_pct'] if not pd.isna(buy_plan['raw_growth_pct']) else 'NA'}%")
        print(f"Used PEG Growth     : {buy_plan['used_growth_pct'] if not pd.isna(buy_plan['used_growth_pct']) else 'NA'}%")
        print(f"Valuation Scale     : {buy_plan['valuation_scale']:.2f}")
        print(f"PEG Note            : {buy_plan['peg_pack']['peg_note']}")

        print("\n=== Capex Detail ===")
        print(f"Capex Source        : {buy_plan['capex_source']}")
        print(f"Capex Quality       : {capex_signal.get('capex_quality', 'UNKNOWN')}")
        print(f"Capex Trend         : {capex_signal.get('capex_trend', 'UNKNOWN')}")
        print(f"Capex QoQ           : {pct_fmt(capex_signal.get('capex_qoq', np.nan))}")
        print(f"Capex YoY           : {pct_fmt(capex_signal.get('capex_yoy', np.nan))}")
        print(f"Capex Slope Rel     : {pct_fmt(capex_signal.get('capex_slope_rel', np.nan))}")
        print(f"Capex Score         : {pretty(capex_signal.get('capex_score', np.nan))}")
        print(f"Capex TTM Latest    : {usd_fmt(capex_signal.get('capex_ttm_latest', np.nan))}")
        print(f"Capex Note          : {capex_signal.get('capex_note', '')}")

        print("\n=== Leverage Engine / Advisory Only ===")
        print("注意：以下只當風險儀表板，不控制 target_tsm_usd。")
        print(f"L_vol          : {lev_pack['L_vol']:.3f}")
        print(f"L_dd           : {lev_pack['L_dd']:.3f}")
        print(f"L_reg          : {lev_pack['L_reg']:.3f}")
        print(f"L_buying_power : {lev_pack['L_buying_power']:.3f}")
        print(f"L_final        : {lev_pack['L_final']:.3f}")

        print("\n================= TSM BUY PLAN =================")
        print(f"Action              : {buy_plan['action']}")
        print(f"Trend State         : {buy_plan['trend_state']}")
        print(f"Trend Scale         : {buy_plan['trend_scale']:.2f}")
        print(f"PE State            : {buy_plan['pe_state']}")
        print(f"PE Scale            : {buy_plan['pe_scale']:.2f}")
        print(f"PEG State           : {buy_plan['peg_state']}")
        print(f"PEG Scale           : {buy_plan['peg_scale']:.2f}")
        print(f"PEG                 : {buy_plan['peg'] if not pd.isna(buy_plan['peg']) else 'NA'}")
        print(f"Raw EPS Growth      : {buy_plan['raw_growth_pct'] if not pd.isna(buy_plan['raw_growth_pct']) else 'NA'}%")
        print(f"Used PEG Growth     : {buy_plan['used_growth_pct'] if not pd.isna(buy_plan['used_growth_pct']) else 'NA'}%")
        print(f"Valuation Scale     : {buy_plan['valuation_scale']:.2f}")
        print(f"Selected PE         : {buy_plan['selected_pe'] if not pd.isna(buy_plan['selected_pe']) else 'NA'}")
        print(f"Target TSM USD      : {buy_plan['target_tsm_usd']:,.0f}")
        print(f"Current TSM USD     : {buy_plan['current_tsm_usd']:,.0f}")
        print(f"Raw Buy Capacity    : {buy_plan['raw_delta']:,.0f}")
        print(f"Max Buy USD         : {buy_plan['max_buy_usd']:,.0f}")
        print(f"L_final Advisory    : {buy_plan['advisory_L_final']:.3f}")

        if (
            buy_plan["action"] == "BUY"
            and not pd.isna(buy_plan["shares_to_buy"])
            and buy_plan["shares_to_buy"] > 0
        ):
            print("\nTrade suggestion:")
            print(f"BUY {fmt_shares(buy_plan['shares_to_buy'])} shares of {CORE_TICKER}")
            print(f"Estimated price : {buy_plan['price']:.2f}")
            print(f"Estimated USD   : {buy_plan['est_buy_usd']:,.2f}")
            if ALLOW_FRACTIONAL:
                print(f"Est new shares  : {buy_plan['est_new_shares']:.3f}")
            else:
                print(f"Est new shares  : {int(buy_plan['est_new_shares'])}")
            print(f"Est new avg cost: {buy_plan['est_new_avg_cost']:.2f}")
        else:
            print("\nTrade suggestion: HOLD / DO NOTHING")

        print("\nReasons:")
        if buy_plan["reasons"]:
            for r in buy_plan["reasons"]:
                print(f"- {r}")
        else:
            print("- Buy allowed. No special block.")

        # ====================================================
        # CHARTS
        # ====================================================

        # Macro charts - market proxy based, no Shiller/CAPE dependency
        factors = res_macro["factors"].copy()
        raw_macro = res_macro.get("raw", pd.DataFrame()).copy()

        fig = plt.figure(figsize=(14, 10))

        ax1 = plt.subplot(4, 1, 1)
        plot_line(ax1, factors["TotalScore"], "Total Macro Risk Score - Market Proxy")
        ax1.axhline(0, linewidth=1)

        ax2 = plt.subplot(4, 1, 2)
        plot_line(ax2, factors["Dislocation"], "Fast Dislocation: HYG / LQD / VIX / TLT / DXY / GLD")
        ax2.axhline(res_macro["disloc_thr"], linestyle="--", linewidth=1)

        ax3 = plt.subplot(4, 1, 3)
        plot_line(ax3, factors["Credit"], "Credit Stress: -Z(21D HYG/LQD)")
        ax3.axhline(DEFAULT_MACRO_THRESHOLDS["Credit"], linestyle="--", linewidth=1)

        ax4 = plt.subplot(4, 1, 4)
        plot_line(ax4, factors["Volatility"], "Volatility Stress: Z(VIX)")
        ax4.axhline(DEFAULT_MACRO_THRESHOLDS["Vol"], linestyle="--", linewidth=1)

        plt.tight_layout()
        plt.show()

        # Proxy assets chart
        cols_to_plot = [c for c in ["HYG", "LQD", "^VIX", "TLT", "DX-Y.NYB", "UUP", "GLD"] if c in raw_macro.columns]
        if cols_to_plot:
            norm = raw_macro[cols_to_plot].dropna(how="all").copy()
            norm = norm.ffill()
            norm = norm / norm.iloc[0] * 100

            plt.figure(figsize=(14, 6))
            for c in norm.columns:
                plt.plot(norm.index, norm[c], label=c)
            plt.title("Macro Proxy Assets Indexed to 100: HYG / LQD / VIX / TLT / DXY or UUP / GLD")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
        else:
            print("Macro proxy asset chart skipped: no HYG/LQD/VIX/TLT/DXY/UUP/GLD data.")

        # Price / MA chart
        dfp = tsm_risk["df"].copy()
        close = dfp["Close"]
        ma60 = close.rolling(RISK_RULES["MA_FAST"]).mean()
        ma200 = close.rolling(RISK_RULES["MA_SLOW"]).mean()

        plt.figure(figsize=(14, 6))
        plt.plot(close.index, close.values, label="TSM Close")
        plt.plot(ma60.index, ma60.values, label="MA60")
        plt.plot(ma200.index, ma200.values, label="MA200")
        plt.title("TSM Price / MA60 / MA200")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

        # Growth chart
        raw_g = buy_plan["raw_growth_pct"]
        used_g = buy_plan["used_growth_pct"]

        plt.figure(figsize=(8, 5))
        plt.bar(["Raw EPS Growth", "Capex-adjusted PEG Growth"], [
            raw_g if not pd.isna(raw_g) else 0,
            used_g if not pd.isna(used_g) else 0
        ])
        plt.title("Raw EPS Growth vs Capex-adjusted PEG Growth")
        plt.ylabel("Growth %")
        plt.grid(True, axis="y", alpha=0.3)
        plt.show()

        # Capex trend chart
        capex_table = capex_signal.get("capex_table", pd.DataFrame())
        if capex_table is not None and not capex_table.empty:
            plt.figure(figsize=(14, 6))

            if "capex" in capex_table.columns:
                plt.bar(capex_table.index, capex_table["capex"], width=30, alpha=0.5, label="Quarterly Capex")

            if "capex_ttm" in capex_table.columns:
                plt.plot(capex_table.index, capex_table["capex_ttm"], marker="o", label="TTM Capex")

            plt.title("TSM Capex Trend / QoQ-based Signal")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
        else:
            print("Capex chart skipped: no capex data.")

        # PE / PEG threshold chart
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        pe_val = buy_plan["selected_pe"]
        plt.bar(["Selected PE"], [pe_val if not pd.isna(pe_val) else 0])
        for th in [
            RISK_RULES["PE_CHEAP"],
            RISK_RULES["PE_FAIR"],
            RISK_RULES["PE_WARM"],
            RISK_RULES["PE_EXPENSIVE"],
            RISK_RULES["PE_HARD_BLOCK"],
        ]:
            plt.axhline(th, linestyle="--", linewidth=1)
        plt.title("PE Threshold")
        plt.grid(True, axis="y", alpha=0.3)

        plt.subplot(1, 2, 2)
        peg_val = buy_plan["peg"]
        plt.bar(["Normalized PEG"], [peg_val if not pd.isna(peg_val) else 0])
        for th in [
            RISK_RULES["PEG_CHEAP"],
            RISK_RULES["PEG_FAIR"],
            RISK_RULES["PEG_WARM"],
            RISK_RULES["PEG_EXPENSIVE"],
            RISK_RULES["PEG_HARD_BLOCK"],
        ]:
            plt.axhline(th, linestyle="--", linewidth=1)
        plt.title("PEG Threshold")
        plt.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Buy scale components
        final_cash_usage = buy_plan["est_buy_usd"] / FREE_CASH_USD if FREE_CASH_USD > 0 else 0.0

        plt.figure(figsize=(10, 5))
        plt.bar(
            ["Trend", "PE", "PEG", "Valuation", "Final Cash Usage"],
            [
                buy_plan["trend_scale"],
                buy_plan["pe_scale"],
                buy_plan["peg_scale"],
                buy_plan["valuation_scale"],
                final_cash_usage,
            ]
        )
        plt.title("Buy Scale Components")
        plt.ylim(0, max(1.2, buy_plan["trend_scale"], buy_plan["pe_scale"], buy_plan["peg_scale"], buy_plan["valuation_scale"], final_cash_usage) * 1.15)
        plt.grid(True, axis="y", alpha=0.3)
        plt.show()


# ============================================================
# 17) MAIN RUN
# ============================================================

def run_system():
    print("Running TSM-only cash buy system...")

    start = DEFAULT_START
    end = DEFAULT_END

    # Macro
    res_macro = build_macro_dashboard(
        start=start,
        end=end,
        lookback=DEFAULT_LOOKBACK,
        weights=DEFAULT_MACRO_WEIGHTS,
        thresholds=DEFAULT_MACRO_THRESHOLDS,
        disloc_thr=DEFAULT_DISLOC_THR,
        ffill_limit=DEFAULT_FFILL_LIMIT,
    )

    if res_macro is None:
        raise RuntimeError("Macro dashboard failed: no data.")

    # TSM risk
    tsm_risk = compute_tsm_risk(
        ticker=CORE_TICKER,
        end=end,
        rules=RISK_RULES,
        avg_cost=TSM_AVG_COST,
    )

    # Holdings
    holdings = compute_tsm_holdings(tsm_risk["last_price"])

    # Risk pack
    try:
        sigma_ann = ann_vol(tsm_risk["ret"], lookback=60)
    except Exception:
        sigma_ann = tsm_risk.get("sigma_60", np.nan)

    try:
        dd_now = drawdown_now(tsm_risk["ret"])
    except Exception:
        dd_now = tsm_risk.get("dd_120", np.nan)

    try:
        beta = estimate_beta(tsm_risk["ret"], bench="^NDX", lookback_days=260)
    except Exception:
        beta = np.nan

    risk_pack = {
        "sigma_ann": sigma_ann,
        "dd_now": dd_now,
        "beta": beta,
    }

    current_stock_usd = float(holdings.loc[CORE_TICKER, "market_value"])

    lev_pack = leverage_engine(
        sigma_ann=sigma_ann,
        dd_now=dd_now,
        regime=res_macro["regime"],
        crisis_on=res_macro["crisis_on"],
        limits=res_macro["limits"],
        equity_usd=EQUITY_USD,
        free_cash_usd=FREE_CASH_USD,
        current_stock_usd=current_stock_usd,
    )

    buy_plan = build_tsm_buy_plan(
        tsm_risk=tsm_risk,
        holdings=holdings,
        lev_pack=lev_pack,
        res_macro=res_macro,
    )

    render_all(
        res_macro=res_macro,
        tsm_risk=tsm_risk,
        risk_pack=risk_pack,
        lev_pack=lev_pack,
        holdings=holdings,
        buy_plan=buy_plan,
    )

    if display is not None and out is not None:
        display(out)

    result = {
        "macro": res_macro,
        "tsm_risk": tsm_risk,
        "holdings": holdings,
        "risk_pack": risk_pack,
        "leverage": lev_pack,
        "buy_plan": buy_plan,
    }

    return result


# ============================================================
# 18) EXECUTE
# ============================================================

# Disabled for app import: result = run_system()


# ============================================================
# TSM FUTURE PRICE FORECAST MODULE
# Plug-in for existing TSM-only system
#
# 直接貼到你原本 Colab 最後一格執行
#
# 功能：
# 1. 保留原本 BUY/HOLD 系統
# 2. 新增多算法未來股價估值
# 3. 分成悲觀 / 基準 / 樂觀三情境
# 4. 輸出：
#    - Current PE roll-forward 股價
#    - Manual PE 股價
#    - PEG implied 股價
#    - Integrated Formula 統整公式股價
# 5. 使用 UI card + table 輸出
# ============================================================

import numpy as np
import pandas as pd
# Forecast UI helpers are optional. Streamlit does not need IPython.
try:
    from IPython.display import display as _ipython_display, HTML as _ipython_HTML
    if display is None:
        display = _ipython_display
    if HTML is None:
        HTML = _ipython_HTML
except Exception:
    pass

# ============================================================
# 1) FORECAST CONFIG
# ============================================================

FORECAST_RULES = {
    "ENABLE": True,

    # 預測年數
    "YEARS": 5,

    # EPS 起點
    # FORWARD  = forward EPS
    # TRAILING = trailing EPS
    # BLENDED  = 70% forward + 30% trailing
    "EPS_SOURCE": "FORWARD",

    # 如果資料抓不到 growth，使用保守 fallback
    "FALLBACK_GROWTH": 0.12,

    # 防止 growth 跑太極端
    "MIN_GROWTH": 0.03,
    "MAX_GROWTH": 0.45,

    # 統整公式權重
    # Integrated PE =
    #   current_pe_weight * current_PE
    # + peg_pe_weight     * PEG_implied_PE
    # + manual_pe_weight  * scenario_manual_PE
    "INTEGRATED_WEIGHTS": {
        "current_pe": 0.25,
        "peg_pe": 0.55,
        "manual_pe": 0.20,
    },

    # 情境設定
    # growth_mult 會乘上你系統算出的 capex-adjusted growth
    # peg 是市場願意給的 PEG
    # manual_pe 是獨立 PE 估值法
    "SCENARIOS": {
        "BEAR": {
            "label": "悲觀",
            "growth_mult": 0.70,
            "peg": 0.60,
            "manual_pe": 16.0,
            "color": "#c62828",
        },
        "BASE": {
            "label": "基準",
            "growth_mult": 1.00,
            "peg": 0.80,
            "manual_pe": 22.0,
            "color": "#1565c0",
        },
        "BULL": {
            "label": "樂觀",
            "growth_mult": 1.20,
            "peg": 1.10,
            "manual_pe": 30.0,
            "color": "#2e7d32",
        },
    },
}


# ============================================================
# 2) BASIC FORMAT HELPERS
# ============================================================

def _fc_safe_float(x):
    try:
        if x is None:
            return np.nan
        v = float(x)
        if np.isfinite(v):
            return v
        return np.nan
    except Exception:
        return np.nan


def _fc_pct(x):
    x = _fc_safe_float(x)
    if pd.isna(x):
        return "NA"
    return f"{x * 100:.2f}%"


def _fc_usd(x):
    x = _fc_safe_float(x)
    if pd.isna(x):
        return "NA"
    return f"${x:,.2f}"


def _fc_usd0(x):
    x = _fc_safe_float(x)
    if pd.isna(x):
        return "NA"
    return f"${x:,.0f}"


def _fc_upside(x):
    x = _fc_safe_float(x)
    if pd.isna(x):
        return "NA"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:.1f}%"


def _fc_card(name, value, color):
    return f"""
    <div style='border-left:6px solid {color};padding:10px 14px;margin:5px;
    background:#ffffff;border-radius:12px;min-width:190px;box-shadow:0 1px 5px rgba(0,0,0,0.10);'>
        <div style='font-size:12px;color:#334155;font-weight:700'>{name}</div>
        <div style='font-size:22px;font-weight:900;color:#111827'>{value}</div>
    </div>
    """


def _fc_weighted_average(values, weights):
    total_v = 0.0
    total_w = 0.0

    for k, v in values.items():
        v = _fc_safe_float(v)
        w = _fc_safe_float(weights.get(k, 0.0))

        if pd.isna(v) or pd.isna(w) or w <= 0:
            continue

        total_v += v * w
        total_w += w

    if total_w <= 0:
        return np.nan

    return total_v / total_w


# ============================================================
# 3) EPS SELECTOR
# ============================================================

def forecast_select_eps(pe_pack, rules=FORECAST_RULES):
    source = str(rules.get("EPS_SOURCE", "FORWARD")).upper()

    forward_eps = _fc_safe_float(pe_pack.get("forward_eps", np.nan))
    trailing_eps = _fc_safe_float(pe_pack.get("trailing_eps", np.nan))

    if source == "FORWARD":
        if not pd.isna(forward_eps) and forward_eps > 0:
            return float(forward_eps), "FORWARD_EPS"
        if not pd.isna(trailing_eps) and trailing_eps > 0:
            return float(trailing_eps), "TRAILING_EPS_FALLBACK"

    if source == "TRAILING":
        if not pd.isna(trailing_eps) and trailing_eps > 0:
            return float(trailing_eps), "TRAILING_EPS"
        if not pd.isna(forward_eps) and forward_eps > 0:
            return float(forward_eps), "FORWARD_EPS_FALLBACK"

    if source == "BLENDED":
        if not pd.isna(forward_eps) and forward_eps > 0 and not pd.isna(trailing_eps) and trailing_eps > 0:
            return float(0.70 * forward_eps + 0.30 * trailing_eps), "BLENDED_EPS"
        if not pd.isna(forward_eps) and forward_eps > 0:
            return float(forward_eps), "FORWARD_EPS_FALLBACK"
        if not pd.isna(trailing_eps) and trailing_eps > 0:
            return float(trailing_eps), "TRAILING_EPS_FALLBACK"

    return np.nan, "EPS_MISSING"


# ============================================================
# 4) CORE FORECAST FORMULA
# ============================================================

def compute_tsm_future_price_forecast(
    current_price=None,
    ticker=None,
    forecast_rules=FORECAST_RULES,
):
    """
    核心公式：

    Future EPS:
        future_eps = EPS0 * (1 + growth) ** years

    Current PE 法:
        price_current_pe = future_eps * current_pe

    Manual PE 法:
        price_manual_pe = future_eps * scenario_manual_pe

    PEG 法:
        peg_implied_pe = scenario_peg * growth_percent
        price_peg = future_eps * peg_implied_pe

    統整公式:
        integrated_pe =
            0.25 * current_pe
          + 0.55 * peg_implied_pe
          + 0.20 * manual_pe

        price_integrated = future_eps * integrated_pe
    """

    # 使用你原本系統的 ticker
    if ticker is None:
        ticker = CORE_TICKER if "CORE_TICKER" in globals() else "TSM"

    # 抓即時/最近股價
    if current_price is None:
        if "safe_last_price" in globals():
            current_price = safe_last_price(ticker)
        else:
            import yfinance as yf
            px_df = yf.download(ticker, period="20d", interval="1d", auto_adjust=True, progress=False)
            if isinstance(px_df.columns, pd.MultiIndex):
                current_price = float(px_df["Close"].iloc[:, 0].dropna().iloc[-1])
            else:
                current_price = float(px_df["Close"].dropna().iloc[-1])

    current_price = _fc_safe_float(current_price)

    # 沿用你原本的 PE / CapEx / PEG 系統
    pe_pack = compute_pe_pack(ticker, current_price, RISK_RULES)

    capex_series, capex_source = extract_capex_from_cashflow(ticker)
    capex_signal = compute_capex_signal(capex_series, RISK_RULES)
    capex_signal["capex_source"] = capex_source

    peg_pack = compute_peg_pack(ticker, pe_pack, capex_signal, RISK_RULES)

    eps0, eps_source = forecast_select_eps(pe_pack, forecast_rules)

    current_pe = _fc_safe_float(pe_pack.get("selected_pe", np.nan))
    base_growth = _fc_safe_float(peg_pack.get("used_growth_rate", np.nan))

    if pd.isna(base_growth) or base_growth <= 0:
        base_growth = float(forecast_rules.get("FALLBACK_GROWTH", 0.12))

    years = int(forecast_rules.get("YEARS", 3))
    min_growth = float(forecast_rules.get("MIN_GROWTH", 0.03))
    max_growth = float(forecast_rules.get("MAX_GROWTH", 0.45))
    scenarios = forecast_rules.get("SCENARIOS", {})
    weights = forecast_rules.get("INTEGRATED_WEIGHTS", {})

    rows = []

    for scenario_key, sc in scenarios.items():
        label = sc.get("label", scenario_key)
        color = sc.get("color", "#37474f")

        growth_mult = _fc_safe_float(sc.get("growth_mult", 1.0))
        scenario_growth = base_growth * growth_mult
        scenario_growth = float(np.clip(scenario_growth, min_growth, max_growth))

        growth_pct = scenario_growth * 100.0

        scenario_peg = _fc_safe_float(sc.get("peg", np.nan))
        manual_pe = _fc_safe_float(sc.get("manual_pe", np.nan))

        if not pd.isna(eps0) and eps0 > 0:
            future_eps = eps0 * ((1.0 + scenario_growth) ** years)
        else:
            future_eps = np.nan

        # Method A: current PE 不變
        price_current_pe = future_eps * current_pe if not pd.isna(future_eps) and not pd.isna(current_pe) else np.nan

        # Method B: manual PE 情境估值
        price_manual_pe = future_eps * manual_pe if not pd.isna(future_eps) and not pd.isna(manual_pe) else np.nan

        # Method C: PEG implied PE
        peg_implied_pe = scenario_peg * growth_pct if not pd.isna(scenario_peg) else np.nan
        price_peg = future_eps * peg_implied_pe if not pd.isna(future_eps) and not pd.isna(peg_implied_pe) else np.nan

        # Method D: integrated formula
        integrated_pe = _fc_weighted_average(
            values={
                "current_pe": current_pe,
                "peg_pe": peg_implied_pe,
                "manual_pe": manual_pe,
            },
            weights=weights,
        )

        price_integrated = future_eps * integrated_pe if not pd.isna(future_eps) and not pd.isna(integrated_pe) else np.nan

        rows.append({
            "Scenario": scenario_key,
            "情境": label,
            "Years": years,

            "Current Price": current_price,

            "EPS0": eps0,
            "EPS Source": eps_source,
            "Growth": scenario_growth,
            "Growth %": growth_pct,
            "Future EPS": future_eps,

            "Current PE": current_pe,
            "Manual PE": manual_pe,
            "Scenario PEG": scenario_peg,
            "PEG Implied PE": peg_implied_pe,
            "Integrated PE": integrated_pe,

            "Price_Current_PE": price_current_pe,
            "Price_Manual_PE": price_manual_pe,
            "Price_PEG": price_peg,
            "Price_Integrated": price_integrated,

            "Upside_Current_PE": price_current_pe / current_price - 1.0 if current_price > 0 and not pd.isna(price_current_pe) else np.nan,
            "Upside_Manual_PE": price_manual_pe / current_price - 1.0 if current_price > 0 and not pd.isna(price_manual_pe) else np.nan,
            "Upside_PEG": price_peg / current_price - 1.0 if current_price > 0 and not pd.isna(price_peg) else np.nan,
            "Upside_Integrated": price_integrated / current_price - 1.0 if current_price > 0 and not pd.isna(price_integrated) else np.nan,

            "Color": color,
        })

    df = pd.DataFrame(rows)

    pack = {
        "ticker": ticker,
        "current_price": current_price,
        "pe_pack": pe_pack,
        "peg_pack": peg_pack,
        "capex_signal": capex_signal,
        "forecast_df": df,
        "note": (
            f"Forecast uses {eps_source}; "
            f"base_growth={_fc_pct(base_growth)}; "
            f"capex_quality={capex_signal.get('capex_quality', 'UNKNOWN')}; "
            f"capex_trend={capex_signal.get('capex_trend', 'UNKNOWN')}; "
            f"PEG={peg_pack.get('peg', np.nan)}."
        ),
    }

    return pack


# ============================================================
# 5) FORECAST UI
# ============================================================

def forecast_summary_cards_html(pack):
    df = pack["forecast_df"]

    bear = df[df["Scenario"] == "BEAR"]
    base = df[df["Scenario"] == "BASE"]
    bull = df[df["Scenario"] == "BULL"]

    def get_val(row_df, col):
        if row_df.empty:
            return np.nan
        return row_df.iloc[0].get(col, np.nan)

    html = """
    <div style='margin-top:12px;padding:14px 16px;border-radius:16px;background:#f8fafc;
                border:1px solid #e5e7eb;'>
        <div style='font-size:14px;letter-spacing:1px;color:#475569;font-weight:900;margin-bottom:8px;'>
            TSM FUTURE PRICE FORECAST / 統整股價預測
        </div>
        <div style='display:flex;gap:10px;flex-wrap:wrap;'>
    """

    html += _fc_card(
        "Bear Integrated",
        f"{_fc_usd0(get_val(bear, 'Price_Integrated'))} / {_fc_upside(get_val(bear, 'Upside_Integrated'))}",
        "#c62828"
    )

    html += _fc_card(
        "Base Integrated",
        f"{_fc_usd0(get_val(base, 'Price_Integrated'))} / {_fc_upside(get_val(base, 'Upside_Integrated'))}",
        "#1565c0"
    )

    html += _fc_card(
        "Bull Integrated",
        f"{_fc_usd0(get_val(bull, 'Price_Integrated'))} / {_fc_upside(get_val(bull, 'Upside_Integrated'))}",
        "#2e7d32"
    )

    html += """
        </div>
    </div>
    """

    return html


def forecast_detail_table_html(pack):
    df = pack["forecast_df"]

    if df is None or df.empty:
        return "<div style='color:#c62828;font-weight:900;'>Forecast data missing</div>"

    html = """
    <div style='margin-top:12px;padding:14px 16px;border-radius:16px;background:#f8fafc;
                border:1px solid #e5e7eb;'>
        <div style='font-size:13px;letter-spacing:1px;color:#475569;font-weight:900;margin-bottom:8px;'>
            MULTI-METHOD VALUATION / 多算法估值表
        </div>

        <table style='width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden;
                      font-size:13px;'>
            <thead>
                <tr style='background:#111827;color:white;'>
                    <th style='padding:8px;text-align:left;'>情境</th>
                    <th style='padding:8px;text-align:right;'>Growth</th>
                    <th style='padding:8px;text-align:right;'>Future EPS</th>
                    <th style='padding:8px;text-align:right;'>Current PE法</th>
                    <th style='padding:8px;text-align:right;'>Manual PE法</th>
                    <th style='padding:8px;text-align:right;'>PEG法</th>
                    <th style='padding:8px;text-align:right;'>統整公式</th>
                    <th style='padding:8px;text-align:right;'>統整Upside</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, r in df.iterrows():
        color = r.get("Color", "#37474f")

        html += f"""
            <tr style='border-bottom:1px solid #e5e7eb;'>
                <td style='padding:8px;font-weight:900;color:{color};'>{r["情境"]}</td>
                <td style='padding:8px;text-align:right;color:#111827;'>{r["Growth %"]:.1f}%</td>
                <td style='padding:8px;text-align:right;color:#111827;'>{r["Future EPS"]:.2f}</td>
                <td style='padding:8px;text-align:right;color:#111827;'>{_fc_usd0(r["Price_Current_PE"])}</td>
                <td style='padding:8px;text-align:right;color:#111827;'>{_fc_usd0(r["Price_Manual_PE"])}</td>
                <td style='padding:8px;text-align:right;color:#111827;'>{_fc_usd0(r["Price_PEG"])}</td>
                <td style='padding:8px;text-align:right;font-weight:900;color:{color};'>{_fc_usd0(r["Price_Integrated"])}</td>
                <td style='padding:8px;text-align:right;font-weight:900;color:{color};'>{_fc_upside(r["Upside_Integrated"])}</td>
            </tr>
        """

    html += """
            </tbody>
        </table>
    """

    html += f"""
        <div style='margin-top:10px;font-size:12px;color:#475569;line-height:1.6;'>
            <b>Formula:</b><br>
            Future EPS = EPS0 × (1 + Growth) ^ Years<br>
            PEG Implied PE = Scenario PEG × Growth %<br>
            Integrated PE = 0.25 × Current PE + 0.55 × PEG Implied PE + 0.20 × Manual PE<br>
            Integrated Price = Future EPS × Integrated PE<br><br>
            <b>Note:</b> {pack.get("note", "")}
        </div>
    </div>
    """

    return html


def forecast_raw_dataframe(pack):
    df = pack["forecast_df"].copy()

    show_cols = [
        "Scenario", "情境", "Years",
        "Current Price",
        "EPS0", "EPS Source",
        "Growth", "Future EPS",
        "Current PE", "Manual PE", "Scenario PEG", "PEG Implied PE", "Integrated PE",
        "Price_Current_PE", "Price_Manual_PE", "Price_PEG", "Price_Integrated",
        "Upside_Current_PE", "Upside_Manual_PE", "Upside_PEG", "Upside_Integrated",
    ]

    show_cols = [c for c in show_cols if c in df.columns]
    return df[show_cols]


def run_tsm_future_price_forecast(current_price=None):
    pack = compute_tsm_future_price_forecast(current_price=current_price)

    if display is not None and HTML is not None:
        display(HTML(forecast_summary_cards_html(pack)))
        display(HTML(forecast_detail_table_html(pack)))

        print("=== Forecast Raw Data ===")
        display(forecast_raw_dataframe(pack))
    else:
        print("Forecast UI display is unavailable outside Notebook/Colab.")
        print(forecast_raw_dataframe(pack))

    print("\n=== Inputs ===")
    print("Ticker              :", pack["ticker"])
    print("Current Price       :", _fc_usd(pack["current_price"]))
    print("Selected PE         :", pack["pe_pack"].get("selected_pe", np.nan))
    print("Forward EPS         :", pack["pe_pack"].get("forward_eps", np.nan))
    print("Trailing EPS        :", pack["pe_pack"].get("trailing_eps", np.nan))
    print("Used Growth         :", _fc_pct(pack["peg_pack"].get("used_growth_rate", np.nan)))
    print("PEG                 :", pack["peg_pack"].get("peg", np.nan))
    print("CapEx Quality       :", pack["capex_signal"].get("capex_quality", "UNKNOWN"))
    print("CapEx Trend         :", pack["capex_signal"].get("capex_trend", "UNKNOWN"))
    print("CapEx QoQ           :", _fc_pct(pack["capex_signal"].get("capex_qoq", np.nan)))
    print("CapEx Slope Rel     :", _fc_pct(pack["capex_signal"].get("capex_slope_rel", np.nan)))
    print("CapEx Source        :", pack["capex_signal"].get("capex_source", "UNKNOWN"))

    return pack


# ============================================================
# 6) RUN
# ============================================================

# Disabled for app import: forecast_pack = run_tsm_future_price_forecast()
