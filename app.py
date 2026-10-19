"""
Administrator dashboard for the Suspicious Login Detection System.

    python detector.py                      # creates alerts.db from the simulated logs
    export DASHBOARD_PASSWORD="choose-one"  # secret comes from the environment, never from code
    streamlit run app.py

Security choices (see report, section "Cloud security"):
  * dashboard is closed unless a password is configured and entered
  * IP addresses are masked by default
  * only parameterised SQL; status values are allow-listed; every review is written to an audit log
  * stored incidents carry a SHA-256 fingerprint; a banner appears if any record was altered
"""
import hmac
import os
import sqlite3

import pandas as pd
import streamlit as st

from detector import STATUSES, mask_ip, update_status, verify_integrity

DB_PATH = os.environ.get("ALERTS_DB", "alerts.db")
LEVEL_ORDER = ["High", "Medium", "Low"]

st.set_page_config(page_title="Suspicious Login Review", layout="wide")


# ---------------------------------------------------------------- access control
def require_login():
    secret = os.environ.get("DASHBOARD_PASSWORD")
    if not secret:
        st.error("DASHBOARD_PASSWORD is not set. For safety the dashboard stays closed until an "
                 "administrator password is configured in the environment.")
        st.stop()
    if st.session_state.get("ok"):
        return
    st.title("Suspicious Login Review")
    name = st.text_input("Reviewer name")
    pw = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if name.strip() and hmac.compare_digest(pw.encode(), secret.encode()):
            st.session_state["ok"], st.session_state["reviewer"] = True, name.strip()
            st.rerun()
        else:
            st.error("Invalid name or password.")
    st.stop()


@st.cache_data(ttl=10)
def load(db_path):
    con = sqlite3.connect(db_path)
    try:
        inc = pd.read_sql_query("SELECT * FROM incidents", con)
        alerts = pd.read_sql_query("SELECT * FROM alerts", con)
        try:
            watch = pd.read_sql_query("SELECT * FROM watchlist", con)
        except Exception:
            watch = pd.DataFrame()
        meta = pd.read_sql_query("SELECT * FROM run_meta", con)
    finally:
        con.close()
    for df, cols in ((inc, ["start", "end"]), (alerts, ["timestamp"]), (watch, ["timestamp"])):
        for c in cols:
            if c in df:
                df[c] = pd.to_datetime(df[c], utc=True)
    return inc, alerts, watch, meta


require_login()
if not os.path.exists(DB_PATH):
    st.error(f"{DB_PATH} not found. Run `python detector.py` first.")
    st.stop()

inc, alerts, watch, meta = load(DB_PATH)
st.title("Suspicious Login Review")
st.caption(f"Signed in as {st.session_state['reviewer']}  |  decisions are advisory: the system never blocks accounts automatically")

tampered = verify_integrity(DB_PATH)
if tampered:
    st.error(f"Integrity check FAILED for incident(s): {tampered}. Stored records were modified outside this application.")

# ---------------------------------------------------------------- sidebar filters
with st.sidebar:
    st.header("Filters")
    levels = st.multiselect("Risk level", LEVEL_ORDER, default=["High", "Medium"])
    statuses = st.multiselect("Review status", list(STATUSES), default=["new", "investigating"])
    user_q = st.text_input("User contains")
    show_full_ip = st.checkbox("Show full IP addresses", value=False)
    if len(inc):
        lo, hi = inc["start"].min().date(), inc["end"].max().date()
        dates = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    else:
        dates = None

view = inc[inc["risk_level"].isin(levels) & inc["status"].isin(statuses)].copy()
if user_q:
    view = view[view["users"].str.contains(user_q, case=False, na=False)]
if dates and len(dates) == 2:
    view = view[(view["start"].dt.date >= dates[0]) & (view["start"].dt.date <= dates[1])]
view["ips"] = view["ips"] if show_full_ip else view["ips"].map(
    lambda s: ", ".join(mask_ip(x.strip()) for x in str(s).split(",")))
view["level_rank"] = view["risk_level"].map({l: i for i, l in enumerate(LEVEL_ORDER)})
view = view.sort_values(["level_rank", "max_score", "start"], ascending=[True, False, False])

# ---------------------------------------------------------------- overview
c1, c2, c3, c4 = st.columns(4)
c1.metric("Incidents shown", len(view))
c2.metric("High risk", int((view["risk_level"] == "High").sum()))
c3.metric("Awaiting review", int((inc["status"] == "new").sum()))
c4.metric("Events analysed", int(meta["clean_rows"].iloc[0]) if "clean_rows" in meta else "n/a")

tab_inc, tab_charts, tab_watch = st.tabs(["Incidents", "Trends", "Low-risk watchlist"])

with tab_inc:
    st.dataframe(view[["incident_id", "risk_level", "max_score", "start", "n_alerts", "users", "ips", "rules",
                       "status"]], use_container_width=True, hide_index=True)
    if len(view):
        pick = st.selectbox("Open incident", view["incident_id"].tolist())
        row = inc[inc["incident_id"] == pick].iloc[0]
        st.subheader(f"Incident {pick}  |  {row['risk_level']} ({row['max_score']}/100)")
        st.write("**Why it was flagged:** " + row["summary"])
        detail = alerts[alerts["incident_id"] == pick].sort_values("timestamp").copy()
        if not show_full_ip:
            detail["ip"] = detail["ip"].map(mask_ip)
        st.dataframe(detail[["timestamp", "user", "ip", "country", "device", "success", "risk_score", "reasons"]],
                     use_container_width=True, hide_index=True)
        col_a, col_b = st.columns([2, 1])
        new_status = col_a.selectbox("Set status", list(STATUSES), index=list(STATUSES).index(row["status"]))
        if col_b.button("Save review"):
            update_status(DB_PATH, int(pick), new_status, st.session_state["reviewer"])
            st.cache_data.clear()
            st.success("Saved and written to the audit log.")
            st.rerun()

with tab_charts:
    if len(inc):
        per_day = inc.assign(day=inc["start"].dt.date).groupby(["day", "risk_level"]).size().unstack(fill_value=0)
        st.subheader("Incidents per day")
        st.bar_chart(per_day)
        st.subheader("Incidents by triggered rule")
        rules = inc["rules"].str.split(", ").explode().value_counts()
        st.bar_chart(rules)
        st.subheader("Review outcomes (feedback for tuning thresholds)")
        st.bar_chart(inc["status"].value_counts())

with tab_watch:
    st.write("Events with weak evidence (for example, only the ML model found them unusual). "
             "They do not raise alerts, but are worth a glance during a quiet day.")
    if len(watch):
        w = watch.sort_values("risk_score", ascending=False).copy()
        if not show_full_ip:
            w["ip"] = w["ip"].map(mask_ip)
        st.dataframe(w[["timestamp", "user", "ip", "country", "device", "risk_score", "reasons"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("Watchlist is empty.")
