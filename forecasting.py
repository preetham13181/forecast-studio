"""Forecasting engine used by the Streamlit app (no Streamlit imports, so it is easy to test)."""
from __future__ import annotations

import itertools
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:  # pragma: no cover
    HAS_PROPHET = False

try:
    import holidays as _holidays
    HAS_HOLIDAYS = True
except Exception:  # pragma: no cover
    HAS_HOLIDAYS = False

from statsmodels.tsa.statespace.sarimax import SARIMAX

# --------------------------------------------------------------------------- frequency helpers
FREQS = {
    "D": dict(label="Daily", offset=pd.offsets.Day(), season=7),
    "B": dict(label="Business days (stocks)", offset=pd.offsets.BDay(), season=5),
    "W": dict(label="Weekly", offset=pd.offsets.Week(), season=52),
    "M": dict(label="Monthly", offset=pd.offsets.MonthBegin(), season=12),
}


def detect_freq(ds: pd.Series) -> str:
    """Guess the sampling frequency from the dates."""
    ds = pd.to_datetime(ds).sort_values()
    gaps = ds.diff().dropna().dt.days
    if gaps.empty:
        return "D"
    med = gaps.median()
    if med >= 27:
        return "M"
    if med >= 6:
        return "W"
    weekend_share = (ds.dt.weekday >= 5).mean()
    if weekend_share < 0.02:
        return "B"
    return "D"


def future_dates(last: pd.Timestamp, freq: str, n: int) -> pd.DatetimeIndex:
    off = FREQS[freq]["offset"]
    return pd.date_range(start=last + off, periods=n, freq=off)


def clean_series(df, date_col, value_col, freq=None, dup="mean", fill="interpolate"):
    """Return (clean DataFrame[ds, y], info dict). Raises ValueError with a friendly message."""
    d = df[[date_col, value_col]].copy()
    d.columns = ["ds", "y"]
    d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    bad = int(d["ds"].isna().sum() + d["y"].isna().sum())
    d = d.dropna()
    if len(d) < 20:
        raise ValueError("Need at least 20 valid rows (a date and a number) to forecast.")
    dups = int(d["ds"].duplicated().sum())
    d = d.groupby("ds", as_index=False)["y"].agg(dup).sort_values("ds").reset_index(drop=True)
    freq = freq or detect_freq(d["ds"])
    added = 0
    if freq in ("D", "B"):
        full = pd.date_range(d["ds"].min(), d["ds"].max(), freq=FREQS[freq]["offset"])
        d = d.set_index("ds").reindex(full)
        added = int(d["y"].isna().sum())
        d["y"] = d["y"].interpolate() if fill == "interpolate" else d["y"].ffill()
        d = d.dropna().rename_axis("ds").reset_index()
    return d, dict(freq=freq, dropped=bad, duplicates=dups, filled=added)


# --------------------------------------------------------------------------- holidays
FALLBACK_HOLIDAYS = ["thanksgiving", "christmas", "new_year"]


def holiday_matrix(dates, country: str | None, min_count: int = 1) -> pd.DataFrame:
    """One 0/1 column per holiday name (like the course notebook). Known in advance, so no leakage."""
    dates = pd.DatetimeIndex(dates)
    if not country:
        return pd.DataFrame(index=range(len(dates)))
    if HAS_HOLIDAYS:
        years = range(dates.min().year - 1, dates.max().year + 2)
        cal = _holidays.country_holidays(country, years=years)
        names = pd.Series({pd.Timestamp(k): v.split(" (")[0] for k, v in cal.items()})
        out = pd.DataFrame(0.0, index=range(len(dates)), columns=sorted(set(names.values)))
        for i, d in enumerate(dates):
            if d in names.index:
                out.loc[i, names[d]] = 1.0
        return out
    s = pd.Series(dates)
    return pd.DataFrame({
        "thanksgiving": ((s.dt.month == 11) & (s.dt.weekday == 3) & s.dt.day.between(22, 28)).astype(float),
        "christmas": ((s.dt.month == 12) & (s.dt.day == 25)).astype(float),
        "new_year": ((s.dt.month == 1) & (s.dt.day == 1)).astype(float),
    })


# --------------------------------------------------------------------------- models
MODELS = ["Prophet", "SARIMAX", "Seasonal naive", "Naive (tomorrow = today)"]


def _z(width: float) -> float:
    return norm.ppf(0.5 + width / 2)


def fc_naive(train, fdates, width, **_):
    y = train["y"].values
    h = np.arange(1, len(fdates) + 1)
    sig = np.std(np.diff(y), ddof=1) * np.sqrt(h)
    yhat = np.repeat(y[-1], len(fdates))
    return pd.DataFrame({"ds": fdates, "yhat": yhat, "lo": yhat - _z(width) * sig, "hi": yhat + _z(width) * sig})


def fc_seasonal_naive(train, fdates, width, season=7, **_):
    y = train["y"].values
    season = max(1, min(season, len(y) - 1))
    yhat = np.tile(y[-season:], len(fdates) // season + 1)[: len(fdates)]
    sig = np.std(y[season:] - y[:-season], ddof=1) / np.sqrt(2) * np.sqrt(np.ceil(np.arange(1, len(fdates) + 1) / season))
    return pd.DataFrame({"ds": fdates, "yhat": yhat, "lo": yhat - _z(width) * sig, "hi": yhat + _z(width) * sig})


def fc_prophet(train, fdates, width, cps=0.05, mode="additive", monthly=False, country=None, **_):
    if not HAS_PROPHET:
        raise RuntimeError("Prophet is not installed in this environment.")
    m = Prophet(interval_width=width, changepoint_prior_scale=cps, seasonality_mode=mode, daily_seasonality=False)
    if monthly:
        m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    if country:
        m.add_country_holidays(country_name=country)
    m.fit(train[["ds", "y"]])
    f = m.predict(pd.DataFrame({"ds": fdates}))
    return pd.DataFrame({"ds": fdates, "yhat": f["yhat"].values, "lo": f["yhat_lower"].values, "hi": f["yhat_upper"].values})


def fc_sarimax(train, fdates, width, order=(1, 1, 1), seasonal=(1, 1, 1), season=7, country=None, **_):
    ex_tr = ex_fu = None
    if country:
        allm = holiday_matrix(list(train["ds"]) + list(fdates), country)
        keep = allm.iloc[: len(train)].sum() > 0          # only holidays seen in training
        allm = allm.loc[:, keep]
        if allm.shape[1]:
            ex_tr, ex_fu = allm.iloc[: len(train)].values, allm.iloc[len(train):].values
    seas = (seasonal[0], seasonal[1], seasonal[2], season) if season and season > 1 and any(seasonal) else (0, 0, 0, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(train["y"].values, exog=ex_tr, order=order, seasonal_order=seas).fit(disp=False, maxiter=100)
        pr = res.get_forecast(len(fdates), exog=ex_fu)
        ci = pr.conf_int(alpha=1 - width)
    ci = np.asarray(ci)
    return pd.DataFrame({"ds": fdates, "yhat": np.asarray(pr.predicted_mean), "lo": ci[:, 0], "hi": ci[:, 1]})


RUNNERS = {
    "Prophet": fc_prophet,
    "SARIMAX": fc_sarimax,
    "Seasonal naive": fc_seasonal_naive,
    "Naive (tomorrow = today)": fc_naive,
}


def run_model(name, train, fdates, width=0.8, **params):
    return RUNNERS[name](train, fdates, width, **params)


# --------------------------------------------------------------------------- evaluation
def scores(actual, pred, lo=None, hi=None) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    out = {"MAE": float(np.mean(np.abs(a - p))), "RMSE": float(np.sqrt(np.mean((a - p) ** 2)))}
    out["MAPE %"] = float(np.mean(np.abs((a - p) / a)) * 100) if np.all(a != 0) else float("nan")
    if lo is not None:
        out["Coverage %"] = float(np.mean((a >= lo) & (a <= hi)) * 100)
    return out


def backtest(df, model, holdout, freq, width=0.8, **params):
    train, test = df.iloc[:-holdout], df.iloc[-holdout:]
    fd = pd.DatetimeIndex(test["ds"])
    f = run_model(model, train, fd, width, **params)
    return train, test, f, scores(test["y"], f["yhat"], f["lo"], f["hi"])


def arima_grid(df, holdout, pmax, qmax, d=1, seasonal=(1, 1, 1), season=7, country=None, progress=None):
    """Fit SARIMAX for every (p, q); return AIC/BIC on train and MAE on the holdout."""
    train, test = df.iloc[:-holdout], df.iloc[-holdout:]
    fd = pd.DatetimeIndex(test["ds"])
    rows, combos = [], list(itertools.product(range(pmax + 1), range(qmax + 1)))
    ex_tr = ex_fu = None
    if country:
        allm = holiday_matrix(list(train["ds"]) + list(fd), country)
        allm = allm.loc[:, allm.iloc[: len(train)].sum() > 0]
        if allm.shape[1]:
            ex_tr, ex_fu = allm.iloc[: len(train)].values, allm.iloc[len(train):].values
    seas = (seasonal[0], seasonal[1], seasonal[2], season) if season > 1 and any(seasonal) else (0, 0, 0, 0)
    for i, (p, q) in enumerate(combos):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = SARIMAX(train["y"].values, exog=ex_tr, order=(p, d, q), seasonal_order=seas).fit(disp=False, maxiter=100)
                fc = r.forecast(len(fd), exog=ex_fu)
            rows.append(dict(p=p, q=q, AIC=r.aic, BIC=r.bic, MAE=float(np.mean(np.abs(test["y"].values - fc)))))
        except Exception:
            rows.append(dict(p=p, q=q, AIC=np.nan, BIC=np.nan, MAE=np.nan))
        if progress:
            progress((i + 1) / len(combos))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- business decision
def newsvendor(f: pd.DataFrame, width: float, cost_under: float, cost_over: float) -> pd.DataFrame:
    """Order quantity = forecast + z * sigma, where z comes from the critical ratio cu / (cu + co)."""
    cr = cost_under / (cost_under + cost_over)
    sigma = (f["hi"] - f["lo"]) / (2 * _z(width))
    out = f[["ds", "yhat"]].copy()
    out["order"] = f["yhat"] + norm.ppf(cr) * sigma
    out["buffer"] = out["order"] - out["yhat"]
    out.attrs["critical_ratio"] = cr
    return out
