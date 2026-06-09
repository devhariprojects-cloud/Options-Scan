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
