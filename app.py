import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecasting as fx

RED, GREY = "#C41230", "#6b7280"
st.set_page_config(page_title="Forecast Studio", page_icon="📈", layout="wide")

st.markdown(
    """<style>
    .block-container{padding-top:2rem;max-width:1200px}
    div[data-testid="stMetric"]{background:#f7f7f5;border:1px solid #e5e5e0;border-radius:10px;padding:12px 16px}
    </style>""",
    unsafe_allow_html=True,
)

st.title("📈 Forecast Studio")
st.caption("Bring any time series (sales, traffic, a stock price). Pick a model, test it honestly on the most "
           "recent data, and turn the forecast into a decision. The shaded band is a range of likely values, not a promise.")

# --------------------------------------------------------------------------- 1. DATA
SAMPLE = Path(__file__).parent / "data" / "daily_sales.csv"
with st.sidebar:
    st.header("1 · Data")
    sources = ["Sample: shop sales", "Upload my own CSV"] + (["Stock from Yahoo Finance"])
    src = st.radio("Source", sources, label_visibility="collapsed")

raw, default_date, default_val = None, None, None
if src == "Sample: shop sales":
    raw = pd.read_csv(SAMPLE)
elif src == "Upload my own CSV":
    up = st.sidebar.file_uploader("CSV or Excel file", type=["csv", "xlsx"])
    if up is not None:
        try:
            raw = pd.read_excel(up) if up.name.endswith("xlsx") else pd.read_csv(up, sep=None, engine="python")
        except Exception as e:
            st.sidebar.error(f"Could not read the file: {e}")
    st.sidebar.caption("Needs one date column and one numeric column.")
else:
    ticker = st.sidebar.text_input("Ticker", "AAPL").strip().upper()
    c1, c2 = st.sidebar.columns(2)
    start = c1.date_input("From", pd.Timestamp("2019-01-01"))
    end = c2.date_input("To", pd.Timestamp.today())
    if st.sidebar.button("Download prices") or st.session_state.get("tkr") == (ticker, str(start), str(end)):
        try:
            import yfinance as yf

            data = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if data.empty:
                raise ValueError("no rows returned (check the ticker or dates)")
            st.session_state["stock"] = data[["Close"]].reset_index().rename(columns={"Date": "date", "Close": "close"})
            st.session_state["tkr"] = (ticker, str(start), str(end))
        except Exception as e:
            st.sidebar.error(f"Yahoo Finance download failed: {e}")
    raw = st.session_state.get("stock")

if raw is None:
    st.info("Choose a data source in the sidebar to begin. The sample data is three years of daily sales for a made-up shop.")
    st.stop()

st.subheader("Your data")
with st.expander("First rows", expanded=False):
    st.dataframe(raw.head(10), use_container_width=True)

cols = list(raw.columns)
guess_d = next((c for c in cols if "date" in str(c).lower() or "time" in str(c).lower()), cols[0])
num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(raw[c])] or cols
guess_v = "close" if "close" in num_cols else ("sales" if "sales" in num_cols else num_cols[-1])
a, b, c = st.columns(3)
date_col = a.selectbox("Which column has the dates?", cols, index=cols.index(guess_d))
val_col = b.selectbox("Which column has the numbers?", cols, index=cols.index(guess_v) if guess_v in cols else 0)
auto = fx.detect_freq(pd.to_datetime(raw[date_col], errors="coerce").dropna()) if len(raw) > 2 else "D"
freq = c.selectbox("Frequency", list(fx.FREQS), index=list(fx.FREQS).index(auto),
                   format_func=lambda k: fx.FREQS[k]["label"] + (" (detected)" if k == auto else ""))

try:
    df, info = fx.clean_series(raw, date_col, val_col, freq=freq)
except Exception as e:
    st.error(str(e))
    st.stop()
notes = []
if info["dropped"]:
    notes.append(f"{info['dropped']} unreadable rows dropped")
if info["duplicates"]:
    notes.append(f"{info['duplicates']} duplicate dates averaged")
if info["filled"]:
    notes.append(f"{info['filled']} missing dates filled by interpolation")
st.caption(f"{len(df):,} observations · {df['ds'].min():%d %b %Y} → {df['ds'].max():%d %b %Y}" + (" · " + " · ".join(notes) if notes else ""))

# --------------------------------------------------------------------------- 2. MODEL
with st.sidebar:
    st.header("2 · Forecast")
    horizon = st.slider("How far ahead to forecast (periods)", 5, 365, 60)
    width = st.slider("Uncertainty band", 0.5, 0.95, 0.80, 0.05, help="Share of future values the band should contain.")
    st.header("3 · Model")
    choices = [m for m in fx.MODELS if m != "Prophet" or fx.HAS_PROPHET]
    model = st.selectbox("Model", choices)
    if not fx.HAS_PROPHET:
        st.caption("Prophet is not installed here, so it is hidden.")
    params = dict(season=fx.FREQS[freq]["season"])
    country = None
    if model in ("Prophet", "SARIMAX"):
        if st.checkbox("Include public holidays", value=False):
            country = st.selectbox("Country", ["US", "DE", "GB", "FR", "CA", "IN", "JP", "BR", "AU", "ES", "IT"])
    if model == "Prophet":
        params["cps"] = st.select_slider("Trend flexibility (changepoint_prior_scale)",
                                         [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0], 0.05,
                                         help="Higher = the trend can bend more.")
        params["mode"] = st.radio("Seasonality mode", ["additive", "multiplicative"], horizontal=True,
                                  help="Multiplicative lets seasonal swings grow with the level.")
        params["monthly"] = st.checkbox("Add monthly seasonality", value=False)
    elif model == "SARIMAX":
        st.caption("Regular order (p, d, q)")
        x1, x2, x3 = st.columns(3)
        p_, d_, q_ = x1.number_input("p", 0, 8, 1), x2.number_input("d", 0, 2, 1), x3.number_input("q", 0, 8, 1)
        params["order"] = (int(p_), int(d_), int(q_))
        params["season"] = st.number_input("Season length", 1, 400, fx.FREQS[freq]["season"],
                                           help="7 for daily data with weekly pattern; 1 turns the seasonal part off.")
        if params["season"] > 1:
            y1, y2, y3 = st.columns(3)
            params["seasonal"] = (int(y1.number_input("P", 0, 3, 1)), int(y2.number_input("D", 0, 2, 1)), int(y3.number_input("Q", 0, 3, 1)))
    params["country"] = country
    holdout = st.slider("Test period for the backtest", 5, max(6, min(180, len(df) // 3)), min(60, max(5, len(df) // 5)),
                        help="The last part of the data is hidden from the model and used for scoring.")

if len(df) < holdout + 20:
    st.error("Not enough data for this test period. Lower the test period.")
    st.stop()


@st.cache_data(show_spinner=False)
def cached_forecast(df_, model_, horizon_, freq_, width_, params_):
    fd = fx.future_dates(df_["ds"].iloc[-1], freq_, horizon_)
    return fx.run_model(model_, df_, fd, width_, **params_)


@st.cache_data(show_spinner=False)
def cached_backtest(df_, model_, holdout_, freq_, width_, params_):
    return fx.backtest(df_, model_, holdout_, freq_, width_, **params_)


tab_f, tab_b, tab_d, tab_t, tab_h = st.tabs(["🔮 Forecast", "🧪 Backtest: can it beat naive?", "📦 Decision", "🎛️ ARIMA tuning", "ℹ️ Help"])

# --------------------------------------------------------------------------- FORECAST
with tab_f:
    with st.spinner("Fitting model..."):
        try:
            fc = cached_forecast(df, model, horizon, freq, width, params)
        except Exception as e:
            st.error(f"The model could not be fitted: {e}")
            st.stop()
    hist_n = st.slider("History to show (periods)", 30, len(df), min(len(df), 365 * 2 if freq == "D" else len(df)))
    h = df.tail(hist_n)
    fig = go.Figure()
    fig.add_scatter(x=fc["ds"], y=fc["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fc["ds"], y=fc["lo"], fill="tonexty", fillcolor="rgba(196,18,48,0.18)", line=dict(width=0),
                    name=f"{int(width*100)}% band", hoverinfo="skip")
    fig.add_scatter(x=h["ds"], y=h["y"], name="history", line=dict(color="black", width=1.3))
    fig.add_scatter(x=fc["ds"], y=fc["yhat"], name="forecast", line=dict(color=RED, width=2.2))
    fig.update_layout(title=f"{model}: forecast for the next {horizon} periods", height=480, hovermode="x unified",
                      legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=70, b=10), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Last observed", f"{df['y'].iloc[-1]:,.2f}")
    m2.metric(f"Forecast at period {horizon}", f"{fc['yhat'].iloc[-1]:,.2f}", f"{fc['yhat'].iloc[-1]-df['y'].iloc[-1]:+,.2f}")
    m3.metric("Average forecast", f"{fc['yhat'].mean():,.2f}")
    out = fc.rename(columns={"ds": "date", "yhat": "forecast", "lo": "low", "hi": "high"}).round(2)
    st.dataframe(out, use_container_width=True, height=300)
    st.download_button("Download forecast as CSV", out.to_csv(index=False).encode(), "forecast.csv", "text/csv")

# --------------------------------------------------------------------------- BACKTEST
with tab_b:
    st.markdown(f"The model sees data up to **{df['ds'].iloc[-holdout-1]:%d %b %Y}**, forecasts the next **{holdout}** periods, "
                "and is scored against what really happened. Every model is compared with two baselines.")
    with st.spinner("Backtesting..."):
        try:
            tr, te, f, sc = cached_backtest(df, model, holdout, freq, width, params)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()
        fd = pd.DatetimeIndex(te["ds"])
        nv = fx.run_model("Naive (tomorrow = today)", tr, fd, width)
        sn = fx.run_model("Seasonal naive", tr, fd, width, season=params["season"])
    s_nv, s_sn = fx.scores(te["y"], nv["yhat"]), fx.scores(te["y"], sn["yhat"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{model} MAE", f"{sc['MAE']:,.2f}")
    c2.metric("Naive MAE", f"{s_nv['MAE']:,.2f}", f"{(sc['MAE']/s_nv['MAE']-1)*100:+.0f}% vs naive", delta_color="inverse")
    c3.metric("Seasonal naive MAE", f"{s_sn['MAE']:,.2f}", f"{(sc['MAE']/s_sn['MAE']-1)*100:+.0f}% vs seasonal", delta_color="inverse")
    c4.metric("Band coverage", f"{sc['Coverage %']:.0f}%", f"target {int(width*100)}%", delta_color="off")
    if model.startswith("Naive"):
        st.info("You are looking at the naive baseline itself.")
    elif sc["MAE"] < min(s_nv["MAE"], s_sn["MAE"]):
        gain = (1 - sc["MAE"] / min(s_nv["MAE"], s_sn["MAE"])) * 100
        st.success(f"✅ {model} beats both baselines by {gain:.0f}% on this test window. One window is limited evidence; for stock prices, "
                   "be sceptical and check different windows.")
    else:
        st.warning(f"⚠️ {model} does **not** beat the best simple baseline on this window. "
                   "That is common for stock prices: the best guess for tomorrow is often today.")
    fig = go.Figure()
    fig.add_scatter(x=f["ds"], y=f["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=f["ds"], y=f["lo"], fill="tonexty", fillcolor="rgba(196,18,48,0.15)", line=dict(width=0), name="band", hoverinfo="skip")
    fig.add_scatter(x=tr["ds"].tail(holdout * 2), y=tr["y"].tail(holdout * 2), name="train (tail)", line=dict(color=GREY))
    fig.add_scatter(x=te["ds"], y=te["y"], name="actual", line=dict(color="black", width=1.6))
    fig.add_scatter(x=f["ds"], y=f["yhat"], name=model, line=dict(color=RED, width=2.2))
    fig.add_scatter(x=nv["ds"], y=nv["yhat"], name="naive", line=dict(color="#2563eb", dash="dot"))
    fig.add_scatter(x=sn["ds"], y=sn["yhat"], name="seasonal naive", line=dict(color="#16a34a", dash="dot"))
    fig.update_layout(height=460, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1),
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    table = pd.DataFrame({"Naive": s_nv, "Seasonal naive": s_sn, model: sc}).T.round(2)
    st.dataframe(table, use_container_width=True)

    if st.button("Compare every available model with these settings"):
        rows = {}
        bar = st.progress(0.0)
        for i, mname in enumerate([m for m in fx.MODELS if m != "Prophet" or fx.HAS_PROPHET]):
            try:
                pr = dict(params)
                if mname == "Prophet":
                    pr.setdefault("cps", 0.05)
                    pr.setdefault("mode", "additive")
                if mname == "SARIMAX":
                    pr.setdefault("order", (1, 1, 1))
                    pr.setdefault("seasonal", (1, 1, 1))
                _, _, _, s = fx.backtest(df, mname, holdout, freq, width, **pr)
                rows[mname] = s
            except Exception as e:
                rows[mname] = {"MAE": np.nan}
            bar.progress((i + 1) / len(fx.MODELS))
        st.dataframe(pd.DataFrame(rows).T.sort_values("MAE").round(2), use_container_width=True)

# --------------------------------------------------------------------------- DECISION
with tab_d:
    st.markdown("**Stock-out costs more than leftovers: order the point forecast, or more?** "
                "Set the two costs. The ordering rule is the *newsvendor* rule: order up to the forecast quantile "
                "equal to `cost_under / (cost_under + cost_over)`.")
    x1, x2 = st.columns(2)
    cu = x1.number_input("Cost of one unit short (lost margin + goodwill)", 0.01, 10_000.0, 10.0)
    co = x2.number_input("Cost of one unit left over (waste, storage)", 0.01, 10_000.0, 2.0)
    nvd = fx.newsvendor(cached_forecast(df, model, horizon, freq, width, params), width, cu, co)
    cr = nvd.attrs["critical_ratio"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Target service level", f"{cr*100:.0f}%", help="Chance of NOT running out.")
    k2.metric("Extra units vs point forecast", f"{nvd['buffer'].sum():,.0f}")
    k3.metric("Total to order", f"{nvd['order'].sum():,.0f}", f"{nvd['buffer'].sum()/nvd['yhat'].sum()*100:+.1f}%")
    fig = go.Figure()
    fig.add_scatter(x=nvd["ds"], y=nvd["yhat"], name="point forecast", line=dict(color=RED))
    fig.add_scatter(x=nvd["ds"], y=nvd["order"], name="recommended order", line=dict(color="black", dash="dash"))
    fig.update_layout(height=380, template="plotly_white", hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Assumes errors are roughly normal and that the band is well calibrated: check the coverage in the Backtest tab. "
               "Also consider shelf life, supplier lead time and minimum order size, and whether unmet demand is lost or backordered.")
    st.download_button("Download order plan", nvd.round(2).to_csv(index=False).encode(), "order_plan.csv", "text/csv")

# --------------------------------------------------------------------------- TUNING
with tab_t:
    st.markdown("Fit SARIMAX for every (p, q) combination on the **training** data, compare **AIC / BIC** (lower is better), "
                "then see the **test MAE** of each. Does adding lags help?")
    g1, g2 = st.columns(2)
    pmax, qmax = g1.slider("Max p (AR lags)", 1, 6, 3), g2.slider("Max q (MA lags)", 1, 6, 3)
    st.caption("Each fit takes a few seconds, so a 4×4 grid can take a few minutes on a free server.")
    if st.button("Run grid search"):
        bar = st.progress(0.0)
        grid = fx.arima_grid(df, holdout, pmax, qmax, season=params["season"], country=country, progress=bar.progress)
        st.session_state["grid"] = grid
    if "grid" in st.session_state:
        grid = st.session_state["grid"].copy()
        ba, bb = grid.loc[grid["AIC"].idxmin()], grid.loc[grid["BIC"].idxmin()]
        st.success(f"Best by AIC: (p={int(ba.p)}, q={int(ba.q)}) → test MAE {ba.MAE:.2f}.  Best by BIC: (p={int(bb.p)}, q={int(bb.q)}) → test MAE {bb.MAE:.2f}.")
        piv = grid.pivot(index="p", columns="q", values="AIC")
        hm = go.Figure(go.Heatmap(z=piv.values, x=[f"q={c}" for c in piv.columns], y=[f"p={r}" for r in piv.index],
                                  colorscale="Reds_r", colorbar=dict(title="AIC")))
        hm.update_layout(height=340, title="AIC (darker red = better)", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(hm, use_container_width=True)
        st.dataframe(grid.sort_values("AIC").round(1).reset_index(drop=True), use_container_width=True)

# --------------------------------------------------------------------------- HELP
with tab_h:
    st.markdown("""
**How to use**
1. Pick a data source: the sample shop sales, your own CSV/Excel (one date column, one number column), or a stock ticker.
2. Choose a horizon, a model and its settings in the sidebar.
3. **Forecast** shows the prediction and band. **Backtest** hides the last part of the data and checks the model against the naive baselines. **Decision** converts the band into an order quantity. **ARIMA tuning** compares lags with AIC/BIC.

**Models**
- *Prophet*: trend + weekly + yearly seasonality + holidays. Try a larger trend flexibility, multiplicative mode, or monthly seasonality, **one change at a time**.
- *SARIMAX*: ARIMA with a seasonal part and optional holiday regressors.
- *Seasonal naive*: repeat the last season. *Naive*: tomorrow = today.

**Honest warning**: a model that does not beat the naive baseline has no edge. Past accuracy does not guarantee future accuracy, and this is not financial advice.
""")
