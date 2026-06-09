"""
Options Edge — Earnings Volatility Scan (Streamlit)

DISCLAIMER:
This software is provided solely for educational and research purposes.
It is not intended to provide investment advice, and no investment recommendations are made herein.
The developers are not financial advisors and accept no responsibility for any financial decisions
or losses resulting from the use of this software.
Always consult a professional financial advisor before making any investment decisions.

NOTE ON FIDELITY:
The five analysis functions below (filter_dates, yang_zhang, build_term_structure,
get_current_price, compute_recommendation) are copied VERBATIM from the original
calculator.py. The four gate keys and every threshold are byte-for-byte unchanged. The
earnings-calendar batch scan simply feeds tickers into the SAME compute_recommendation;
it does not alter any calculation.
"""

import time
from io import StringIO
from datetime import datetime, timedelta

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import interp1d


# ============================================================================
# ===== ANALYSIS LOGIC — VERBATIM FROM calculator.py (do not modify) =========
# ============================================================================

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)

    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i + 1]]
            break

    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")


def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc ** 2

    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc ** 2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        return result.iloc[-1]
    else:
        return result.dropna()


def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:
            return ivs[0]
        elif dte > days[-1]:
            return ivs[-1]
        else:
            return float(spline(dte))

    return term_spline


def get_current_price(ticker):
    todays_data = ticker.history(period='5d')
    if todays_data.empty:
        return None
    return todays_data['Close'].iloc[-1]


def compute_recommendation(ticker):
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return "No stock symbol provided."

        try:
            stock = yf.Ticker(ticker)
            if len(stock.options) == 0:
                raise KeyError()
        except KeyError:
            return f"Error: No options found for stock symbol '{ticker}'."

        exp_dates = list(stock.options)
        try:
            exp_dates = filter_dates(exp_dates)
        except Exception:
            return "Error: Not enough option data."

        options_chains = {}
        for exp_date in exp_dates:
            options_chains[exp_date] = stock.option_chain(exp_date)

        try:
            underlying_price = get_current_price(stock)
            if underlying_price is None:
                raise ValueError("No market price found.")
        except Exception:
            return "Error: Unable to retrieve underlying stock price."

        atm_iv = {}
        straddle = None
        i = 0
        for exp_date, chain in options_chains.items():
            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                continue

            call_diffs = (calls['strike'] - underlying_price).abs()
            call_idx = call_diffs.idxmin()
            call_iv = calls.loc[call_idx, 'impliedVolatility']

            put_diffs = (puts['strike'] - underlying_price).abs()
            put_idx = put_diffs.idxmin()
            put_iv = puts.loc[put_idx, 'impliedVolatility']

            atm_iv_value = (call_iv + put_iv) / 2.0
            atm_iv[exp_date] = atm_iv_value

            if i == 0:
                call_bid = calls.loc[call_idx, 'bid']
                call_ask = calls.loc[call_idx, 'ask']
                put_bid = puts.loc[put_idx, 'bid']
                put_ask = puts.loc[put_idx, 'ask']

                if call_bid is not None and call_ask is not None:
                    call_mid = (call_bid + call_ask) / 2.0
                else:
                    call_mid = None

                if put_bid is not None and put_ask is not None:
                    put_mid = (put_bid + put_ask) / 2.0
                else:
                    put_mid = None

                if call_mid is not None and put_mid is not None:
                    straddle = (call_mid + put_mid)

            i += 1

        if not atm_iv:
            return "Error: Could not determine ATM IV for any expiration dates."

        today = datetime.today().date()
        dtes = []
        ivs = []
        for exp_date, iv in atm_iv.items():
            exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
            days_to_expiry = (exp_date_obj - today).days
            dtes.append(days_to_expiry)
            ivs.append(iv)

        term_spline = build_term_structure(dtes, ivs)

        ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])

        price_history = stock.history(period='3mo')
        iv30_rv30 = term_spline(30) / yang_zhang(price_history)

        avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]

        expected_move = str(round(straddle / underlying_price * 100, 2)) + "%" if straddle else None

        # Gate keys below are IDENTICAL to the original; the "_" keys are raw values for display only.
        return {
            'avg_volume': avg_volume >= 1500000,
            'iv30_rv30': iv30_rv30 >= 1.25,
            'ts_slope_0_45': ts_slope_0_45 <= -0.00406,
            'expected_move': expected_move,
            '_avg_volume_raw': float(avg_volume),
            '_iv30_rv30_raw': float(iv30_rv30),
            '_ts_slope_raw': float(ts_slope_0_45),
            '_underlying': float(underlying_price),
        }
    except Exception:
        raise Exception('Error occured processing')


# ============================================================================
# ===== HELPERS layered ON TOP (do not touch the logic above) ================
# ============================================================================

def verdict_of(av, iv, ts):
    """Same branching as the original GUI."""
    if av and iv and ts:
        return "Recommended"
    elif ts and ((av and not iv) or (iv and not av)):
        return "Consider"
    return "Avoid"


@st.cache_data(ttl=900, show_spinner=False)
def scan_one(ticker):
    """Run compute_recommendation on one ticker; never raise. Cached 15 min."""
    try:
        res = compute_recommendation(ticker)
    except Exception:
        return {"Ticker": ticker, "Verdict": "Error", "_rank": 4, "Note": "scan failed"}
    if isinstance(res, str):
        return {"Ticker": ticker, "Verdict": "Error", "_rank": 4, "Note": res.replace("Error: ", "")}
    av, iv, ts = res['avg_volume'], res['iv30_rv30'], res['ts_slope_0_45']
    v = verdict_of(av, iv, ts)
    rank = {"Recommended": 0, "Consider": 1, "Avoid": 2}[v]
    return {
        "Ticker": ticker,
        "Verdict": v,
        "Price": round(res['_underlying'], 2),
        "IV30/RV30": round(res['_iv30_rv30_raw'], 2),
        "Slope": round(res['_ts_slope_raw'], 5),
        "AvgVol(M)": round(res['_avg_volume_raw'] / 1e6, 2),
        "Exp.Move": res['expected_move'] or "—",
        "Vol": "pass" if av else "fail",
        "IV/RV": "pass" if iv else "fail",
        "Slope?": "pass" if ts else "fail",
        "_rank": rank,
        "Note": "",
    }


def _finnhub_key():
    """Read the Finnhub API key from Streamlit secrets; '' if not set."""
    try:
        return str(st.secrets["FINNHUB_API_KEY"]).strip()
    except Exception:
        return ""


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_earnings_calendar(date_str):
    """
    Pull earnings for a single date (YYYY-MM-DD) from the Finnhub API (free tier).
    Returns a DataFrame with 'Symbol' and 'Call Time' columns, or empty on failure.
    Requires FINNHUB_API_KEY in Streamlit secrets. Honors the date server-side, so
    it does not have the Yahoo-scrape date/blocking problems.
    """
    key = _finnhub_key()
    if not key:
        return pd.DataFrame(columns=["Symbol", "Call Time"])
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={"from": date_str, "to": date_str, "token": key},
            timeout=25,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Call Time"])

    rows = data.get("earningsCalendar") or []
    if not rows:
        return pd.DataFrame(columns=["Symbol", "Call Time"])

    df = pd.DataFrame(rows)
    hour_map = {"bmo": "Before Open", "amc": "After Close", "dmh": "During Market"}
    symbols = df.get("symbol", pd.Series(dtype=str)).astype(str).str.strip().str.upper()
    hours = df.get("hour", pd.Series(dtype=str)).astype(str).str.lower().map(hour_map).fillna("Unspecified")
    out = pd.DataFrame({"Symbol": symbols, "Call Time": hours})
    out = out[out["Symbol"].str.len() > 0].drop_duplicates(subset="Symbol").reset_index(drop=True)
    return out


def run_batch(tickers):
    results = []
    prog = st.progress(0.0, text="Scanning...")
    n = len(tickers)
    for idx, t in enumerate(tickers):
        results.append(scan_one(t))
        prog.progress((idx + 1) / n, text=f"Scanning {t}  ({idx + 1}/{n})")
        time.sleep(0.3)  # be gentle with the data source
    prog.empty()
    df = pd.DataFrame(results).sort_values(["_rank", "IV30/RV30"], ascending=[True, False])
    return df.drop(columns=["_rank"])


# ============================================================================
# ===== STREAMLIT FRONT-END ==================================================
# ============================================================================

st.set_page_config(page_title="Options Edge - Earnings Scan", page_icon="*", layout="centered")

st.markdown("""
<style>
  .block-container{max-width:620px;padding-top:2rem;}
  .eyebrow{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.28em;
    text-transform:uppercase;color:#C9A227;}
  .verdict{font-weight:800;font-size:34px;letter-spacing:.01em;margin:.1rem 0 .2rem;}
  .vsub{font-family:ui-monospace,monospace;color:#8A92A0;font-size:14px;margin-bottom:1rem;}
  .metric{display:flex;justify-content:space-between;align-items:baseline;
    padding:13px 2px;border-top:1px solid #2a2f3a;}
  .metric .nm{font-size:14px;}
  .metric .nm small{display:block;font-family:ui-monospace,monospace;color:#8A92A0;font-size:11px;}
  .metric .vl{font-family:ui-monospace,monospace;font-size:16px;}
  .chip{font-family:ui-monospace,monospace;font-size:11px;font-weight:700;letter-spacing:.1em;
    padding:3px 8px;border-radius:5px;margin-left:10px;}
  .pass{color:#4FB477;background:rgba(79,180,119,.15);}
  .fail{color:#D7544C;background:rgba(215,84,76,.15);}
  .disclaimer{font-family:ui-monospace,monospace;font-size:10.5px;color:#5c6470;line-height:1.6;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Options Edge - Earnings Volatility Scan</div>', unsafe_allow_html=True)
st.title("Pre-Earnings Volatility Check")

tab_single, tab_cal = st.tabs(["Single ticker", "Earnings date"])

# ---------- Single ticker tab ----------
with tab_single:
    st.caption("Live option chain + 3 months of prices, server-side via yfinance.")
    with st.form("scan"):
        ticker = st.text_input("Stock symbol", placeholder="AAPL", max_chars=8).strip().upper()
        submitted = st.form_submit_button("Run scan", type="primary", use_container_width=True)

    if submitted:
        if not ticker:
            st.warning("Enter a ticker symbol first.")
        else:
            with st.spinner(f"Scanning {ticker}..."):
                try:
                    result = compute_recommendation(ticker)
                except Exception as e:
                    result = f"Error: {e}"

            if isinstance(result, str):
                st.error(result)
            else:
                av = result['avg_volume']; iv = result['iv30_rv30']; ts = result['ts_slope_0_45']
                title = verdict_of(av, iv, ts)
                color = {"Recommended": "#4FB477", "Consider": "#E0A33E", "Avoid": "#D7544C"}[title]

                st.markdown(
                    f'<div class="verdict" style="color:{color}">{title.upper()}</div>'
                    f'<div class="vsub">{ticker} - ${result["_underlying"]:.2f}</div>',
                    unsafe_allow_html=True,
                )

                def row(name, sub, val, ok):
                    chip = "pass" if ok else "fail"
                    lab = "PASS" if ok else "FAIL"
                    return (f'<div class="metric"><div class="nm">{name}<small>{sub}</small></div>'
                            f'<div><span class="vl">{val}</span>'
                            f'<span class="chip {chip}">{lab}</span></div></div>')

                vol_raw = result['_avg_volume_raw']
                vol_disp = f"{vol_raw/1e6:.2f}M" if vol_raw >= 1e6 else f"{vol_raw:,.0f}"
                html = ""
                html += row("Avg volume (30d)", "threshold >= 1.5M", vol_disp, av)
                html += row("IV30 / RV30", "threshold >= 1.25", f"{result['_iv30_rv30_raw']:.2f}", iv)
                html += row("Term slope 0-45", "threshold <= -0.00406", f"{result['_ts_slope_raw']:.5f}", ts)
                move = result['expected_move'] if result['expected_move'] else "-"
                html += row("Expected move", "front-month straddle / spot", move, bool(result['expected_move']))
                st.markdown(html, unsafe_allow_html=True)

# ---------- Earnings date tab ----------
with tab_cal:
    st.caption("Pick a date - the earnings list pulls automatically. Then trim the names and batch-run the same check.")

    cal_date = st.date_input("Earnings date", value=datetime.today().date(), key="cal_date")
    date_str = cal_date.strftime("%Y-%m-%d")

    # Auto-fetch whenever the date changes (cached, so repeats are instant).
    # Clear stale batch results when the date moves.
    if st.session_state.get("last_fetched_date") != date_str:
        st.session_state.pop("batch_results", None)
        st.session_state["last_fetched_date"] = date_str
    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("Re-pull", use_container_width=True, help="Force a fresh pull"):
            fetch_earnings_calendar.clear()
    with st.spinner(f"Pulling earnings calendar for {date_str}..."):
        cal_df = fetch_earnings_calendar(date_str)

    if cal_df.empty:
        if not _finnhub_key():
            st.error(
                "No Finnhub API key found. Add FINNHUB_API_KEY in the app's Secrets "
                "(Manage app -> Settings -> Secrets), or paste tickers manually below."
            )
        else:
            st.warning(
                f"No earnings found for {date_str} (markets may be closed that day). "
                "Try another date, or paste tickers manually below."
            )
    else:
        syms = [str(x).strip().upper() for x in cal_df["Symbol"].tolist() if pd.notna(x) and str(x).strip()]
        st.success(f"{len(syms)} companies reporting on {date_str}.")
        with st.expander("Show fetched tickers"):
            st.write(", ".join(syms))

    all_syms = [str(x).strip().upper() for x in cal_df["Symbol"].tolist() if pd.notna(x) and str(x).strip()] if not cal_df.empty else []

    time_col = next((c for c in cal_df.columns if "call time" in str(c).lower()), None) if not cal_df.empty else None
    if time_col:
        times = sorted(cal_df[time_col].dropna().astype(str).unique().tolist())
        pick_times = st.multiselect("Earnings call time", times, default=times, key=f"times_{date_str}")
        all_syms = [str(x).strip().upper() for x in cal_df[cal_df[time_col].astype(str).isin(pick_times)]["Symbol"].tolist() if pd.notna(x) and str(x).strip()]

    manual = st.text_input("Add / paste tickers (comma or space separated)", "", key=f"manual_{date_str}")
    manual_syms = [s.strip().upper() for s in manual.replace(",", " ").split() if s.strip()]
    pool = list(dict.fromkeys(all_syms + manual_syms))

    default_cap = min(25, max(1, len(pool))) if pool else 25
    cap = st.number_input("Max tickers to scan", min_value=1, max_value=200,
                          value=default_cap, step=5, key=f"cap_{date_str}",
                          help="Each scan makes several data calls - keep this modest to avoid rate-limits.")
    selected = st.multiselect("Tickers to scan (trim as you like)", pool,
                              default=pool[:cap], key=f"sel_{date_str}")
    selected = selected[:cap]

    if st.button(f"Scan {len(selected)} ticker(s)", type="primary",
                 use_container_width=True, disabled=not selected):
        st.session_state["batch_results"] = run_batch(selected)

    results = st.session_state.get("batch_results")
    if results is not None and not results.empty:
        counts = results["Verdict"].value_counts().to_dict()
        st.markdown(
            f"**{counts.get('Recommended',0)}** recommended - "
            f"**{counts.get('Consider',0)}** consider - "
            f"**{counts.get('Avoid',0)}** avoid - "
            f"**{counts.get('Error',0)}** errors"
        )
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button(
            "Download results (CSV)",
            results.to_csv(index=False).encode("utf-8"),
            file_name=f"earnings_scan_{date_str}.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.markdown(
    '<div class="disclaimer">For educational and research purposes only. Not investment advice, '
    'and no recommendation is made. The author is not a financial advisor and accepts no '
    'responsibility for decisions or losses from use of this tool. Consult a professional before trading.</div>',
    unsafe_allow_html=True,
)
