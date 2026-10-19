"""
Python-Based Suspicious Login Detection System
==============================================
Pipeline:  generate/read logs -> validate & clean -> feature engineering
           -> explainable rules + Isolation Forest -> risk score & reasons
           -> incidents -> SQLite storage -> evaluation.

Run end-to-end:   python detector.py
Then dashboard:   streamlit run app.py
"""

# ==== SECTION: config ====
import hashlib
import ipaddress
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

CONFIG = {
    "seed": 42,
    # --- time windows ---
    "window_minutes": 10,          # short window for burst detection
    "long_window_hours": 24,       # long window for slow / low-volume attacks
    "incident_gap_minutes": 30,    # alerts closer than this are merged into one incident
    # --- rule thresholds (adjustable starting points, not proof of an attack) ---
    "brute_force_fails": 5,        # R1: failed logins for ONE account in the window
    "spray_users": 4,              # R2: ONE IP failing against this many accounts in the window
    "success_after_fails": 3,      # R3: successful login right after this many failures
    "max_speed_kmh": 900,          # R4: faster than a passenger jet = impossible travel
    "odd_hour_dev": 6.0,           # R5: hours away from the user's usual login time
    "slow_spray_users": 8,         # R6: distinct accounts failed from one IP in 24h
    "min_history": 5,              # successful logins needed before "new X" is meaningful
    # --- risk scoring ---
    "points": {"R1": 50, "R1_heavy": 15, "R2": 50, "R2_heavy": 15, "R3": 20, "R3b": 25,
               "R4": 60, "R5": 40, "R6": 45, "ML": 25},
    "travel_max_known_countries": 3,   # R4 is skipped for 'frequent hoppers' (VPN users, frequent flyers)
    # ML alone (25) stays Low = "watchlist"; it becomes an alert only together with another signal.
    "level_medium": 30,            # score >= 30 -> Medium  (this is also the alert threshold)
    "level_high": 60,              # score >= 60 -> High
    "alert_threshold": 30,
    # --- machine learning ---
    "ml_contamination": 0.02,
}

# country -> (latitude, longitude, UTC offset hours, typical IPv4 first octets)
COUNTRY_INFO = {
    "IN": (20.6, 78.9, 5.5, [103, 106, 117, 49]),
    "US": (39.8, -98.6, -5.0, [24, 66, 98, 73]),
    "DE": (51.2, 10.4, 1.0, [84, 91, 217, 88]),
    "SG": (1.35, 103.8, 8.0, [116, 175, 202]),
    "GB": (55.4, -3.4, 0.0, [81, 86, 92]),
    "RU": (61.5, 105.3, 3.0, [5, 46, 95, 178]),
    "CN": (35.9, 104.2, 8.0, [36, 58, 111, 223]),
    "BR": (-14.2, -51.9, -3.0, [177, 186, 200]),
    "NG": (9.1, 8.7, 1.0, [41, 105, 197]),
}


# ==== SECTION: generate ====
def _rand_ip(rng, country):
    prefix = int(rng.choice(COUNTRY_INFO[country][3]))
    return f"{prefix}.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"


def generate_logs(n_users=25, days=30, seed=42, start="2026-09-01", messy=True):
    """Simulated cloud authentication service.

    Produces normal behaviour, realistic 'hard negatives' (travellers, a VPN user,
    forgotten passwords, new devices, shared office IP) and labelled attacks.
    Columns starting with gt_ are GROUND TRUTH used only for evaluation; the detector
    never reads them.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start, tz="UTC")
    rows = []
    aid = {"n": 0}

    def add(ts, user, ip, country, device, success, atype="normal", attack_id=0):
        rows.append((ts, user, ip, country, device, int(success),
                     0 if atype == "normal" else 1, atype, attack_id))

    def new_attack():
        aid["n"] += 1
        return aid["n"]

    # ---- user directory ----
    home = rng.choice(["IN", "US", "DE", "SG", "GB"], size=n_users, p=[.5, .2, .12, .12, .06])
    office = {c: _rand_ip(rng, c) for c in set(home)}          # shared office NAT address
    users = []
    for i in range(n_users):
        c = str(home[i])
        users.append({
            "user": f"user{i + 1:02d}", "country": c,
            "devices": ["laptop", "phone"] if rng.random() < .8 else ["laptop"],
            "ips": [office[c] if rng.random() < .5 else _rand_ip(rng, c), _rand_ip(rng, c)],
            "mean_hour": float(rng.normal(12, 1.5)),
            "trip": None, "vpn": False, "swap_day": None,
        })
    for i in (3, 7):                                            # two business travellers
        tc = str(rng.choice([c for c in ["US", "DE", "SG", "GB"] if c != users[i]["country"]]))
        users[i].update(trip=(10, 13), trip_country=tc, trip_ip=_rand_ip(rng, tc))
    users[11]["vpn"] = True                                     # one VPN user
    for i in rng.choice(n_users, size=6, replace=False):        # six users replace their laptop
        users[int(i)]["swap_day"] = int(rng.integers(5, 25))

    # ---- normal behaviour ----
    for u in users:
        off_home = COUNTRY_INFO[u["country"]][2]
        for d in range(days):
            if u["trip"] and d in (u["trip"][0] - 1, u["trip"][1] + 1):
                continue                                         # travel days: no logins
            weekday = (t0 + pd.Timedelta(days=d)).weekday() < 5
            for _ in range(rng.poisson(6 if weekday else 1.5)):
                country, off = u["country"], off_home
                if u["trip"] and u["trip"][0] <= d <= u["trip"][1]:
                    country, ip = u["trip_country"], u["trip_ip"]
                    off = COUNTRY_INFO[country][2]
                elif u["vpn"] and rng.random() < .02:
                    country = str(rng.choice([c for c in ["US", "DE", "SG", "GB"] if c != u["country"]]))
                    ip = _rand_ip(rng, country)
                elif rng.random() < .8:
                    ip = u["ips"][int(rng.integers(len(u["ips"])))]
                else:
                    ip = _rand_ip(rng, country)                  # dynamic ISP address
                device = u["devices"][0] if (len(u["devices"]) == 1 or rng.random() < .7) else u["devices"][1]
                if u["swap_day"] is not None and d >= u["swap_day"] and device == "laptop":
                    device = "laptop2"
                local = float(np.clip(rng.normal(u["mean_hour"], 2.0), 6, 20))
                ts = t0 + pd.Timedelta(days=d, hours=local - off)
                r = rng.random()
                n_fail = 1 if r < .025 else (int(rng.integers(2, 5)) if r < .03 else 0)
                for _f in range(n_fail):                         # ordinary typos
                    add(ts, u["user"], ip, country, device, 0)
                    ts += pd.Timedelta(seconds=float(rng.uniform(10, 40)))
                add(ts, u["user"], ip, country, device, 1)

    # a few legitimate "forgot my password" bursts (hard negatives for the brute-force rule)
    for _ in range(3):
        u = users[int(rng.integers(n_users))]
        ts = t0 + pd.Timedelta(days=float(rng.uniform(6, 29)))
        for _f in range(int(rng.integers(5, 7))):
            add(ts, u["user"], u["ips"][0], u["country"], u["devices"][0], 0)
            ts += pd.Timedelta(seconds=float(rng.uniform(15, 45)))
        add(ts, u["user"], u["ips"][0], u["country"], u["devices"][0], 1)

    # ---- attacks (labelled) ----
    def when():
        return t0 + pd.Timedelta(days=float(rng.uniform(4, 29)))

    def attacker():
        c = str(rng.choice(["RU", "CN", "BR", "NG"]))
        return c, _rand_ip(rng, c)

    def a_device():
        return str(rng.choice(["unknown", "unknown", "tablet", "laptop", "phone"]))

    for _ in range(6):                                           # brute force
        a, (c, ip) = new_attack(), attacker()
        u, ts = users[int(rng.integers(n_users))], when()
        for _f in range(int(rng.integers(10, 40))):
            add(ts, u["user"], ip, c, "unknown", 0, "brute_force", a)
            ts += pd.Timedelta(seconds=float(rng.uniform(2, 15)))

    for _ in range(4):                                           # password spraying
        a, (c, ip) = new_attack(), attacker()
        ts = when()
        for idx in rng.choice(n_users, size=int(rng.integers(6, 13)), replace=False):
            for _f in range(int(rng.integers(1, 3))):
                add(ts, users[int(idx)]["user"], ip, c, "unknown", 0, "password_spray", a)
                ts += pd.Timedelta(seconds=float(rng.uniform(10, 40)))

    for _ in range(5):                                           # failures then success (compromise)
        a, (c, ip) = new_attack(), attacker()
        u, ts, dev = users[int(rng.integers(n_users))], when(), a_device()
        for _f in range(int(rng.integers(3, 9))):
            add(ts, u["user"], ip, c, dev, 0, "compromise", a)
            ts += pd.Timedelta(seconds=float(rng.uniform(5, 30)))
        add(ts, u["user"], ip, c, dev, 1, "compromise", a)

    for _ in range(5):                                           # impossible travel (stolen credentials)
        a = new_attack()
        u, ts = users[int(rng.integers(n_users))], when()
        add(ts, u["user"], u["ips"][0], u["country"], u["devices"][0], 1)          # genuine user
        c = str(rng.choice([x for x in ["RU", "CN", "BR", "NG", "US"] if x != u["country"]]))
        add(ts + pd.Timedelta(minutes=float(rng.uniform(5, 45))), u["user"], _rand_ip(rng, c), c,
            a_device(), 1, "impossible_travel", a)

    for _ in range(5):                                           # odd-hour login from a new place
        a = new_attack()
        u = users[int(rng.integers(n_users))]
        off = COUNTRY_INFO[u["country"]][2]
        ts = t0 + pd.Timedelta(days=int(rng.integers(5, 29)), hours=float(rng.uniform(0.5, 4.0)) - off)
        add(ts, u["user"], _rand_ip(rng, u["country"]), u["country"], a_device(), 1, "odd_hour", a)

    for _ in range(5):                                           # subtle: new device+IP, normal-looking hour
        a = new_attack()
        u = users[int(rng.integers(n_users))]
        off = COUNTRY_INFO[u["country"]][2]
        local = u["mean_hour"] + float(rng.choice([-1, 1])) * float(rng.uniform(3.5, 5.5))
        ts = t0 + pd.Timedelta(days=int(rng.integers(5, 29)), hours=local - off)
        add(ts, u["user"], _rand_ip(rng, u["country"]), u["country"], "tablet", 1, "subtle_takeover", a)

    for _ in range(3):                                           # slow, low-volume spraying
        a, (c, ip) = new_attack(), attacker()
        ts = when()
        for idx in rng.choice(n_users, size=int(rng.integers(15, 22)), replace=False):
            add(ts, users[int(idx)]["user"], ip, c, "unknown", 0, "slow_spray", a)
            ts += pd.Timedelta(minutes=float(rng.uniform(20, 50)))

    df = pd.DataFrame(rows, columns=["timestamp", "user", "ip", "country", "device", "success",
                                     "gt_label", "gt_attack_type", "gt_attack_id"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "event_id", np.arange(1, len(df) + 1))

    # ---- real-world mess (so validation code has something to do) ----
    if messy:
        out = df.copy()
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        normal = out.index[out["gt_label"] == 0].to_numpy()
        for i in rng.choice(len(out), size=int(.15 * len(out)), replace=False):   # local-time offsets
            ts = df.at[i, "timestamp"].tz_convert("Asia/Kolkata")
            out.at[i, "timestamp"] = ts.strftime("%Y-%m-%dT%H:%M:%S%z")
        out = out.astype({"user": object, "ip": object})
        out.loc[rng.choice(normal, 15, replace=False), "user"] = np.nan
        out.loc[rng.choice(normal, 10, replace=False), "ip"] = np.nan
        out.loc[rng.choice(normal, 10, replace=False), "ip"] = "999.12.1.7"
        out.loc[rng.choice(normal, 5, replace=False), "timestamp"] = "not-a-time"
        dup = out.sample(frac=.01, random_state=seed)
        out = pd.concat([out, dup], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
        return out
    return df


# ==== SECTION: preprocess ====
def _is_valid_ip(value):
    try:
        ipaddress.ip_address(str(value).strip())
        return True
    except ValueError:
        return False


def preprocess(raw):
    """Validate, normalise (UTC) and de-duplicate raw login records. Returns (clean_df, report)."""
    report = {"raw_rows": len(raw)}
    df = raw.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="ISO8601")
    df["success"] = pd.to_numeric(df["success"], errors="coerce")
    for col in ("user", "ip"):
        df[col] = df[col].astype("string").str.strip()

    n = len(df)
    df = df.dropna(subset=["timestamp", "user", "ip", "success"])
    df = df[df["user"] != ""]
    report["dropped_missing_or_bad_fields"] = n - len(df)

    n = len(df)
    ok_ip = {v: _is_valid_ip(v) for v in df["ip"].unique()}
    df = df[df["ip"].map(ok_ip)]
    report["dropped_invalid_ip"] = n - len(df)

    df["success"] = df["success"].astype(int).clip(0, 1)
    df["device"] = df["device"].fillna("unknown") if "device" in df else "unknown"
    df["country"] = df["country"].fillna("UNK") if "country" in df else "UNK"

    n = len(df)
    df = df.drop_duplicates(subset=["timestamp", "user", "ip", "device", "success"])
    report["dropped_duplicates"] = n - len(df)

    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if "event_id" not in df:                       # real logs often have no id column
        df.insert(0, "event_id", np.arange(1, len(df) + 1))
    report["clean_rows"] = len(df)
    return df, report


# ==== SECTION: features ====
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _window_counts(t, flag, window_s):
    """For each event: number of flagged events in the window (t - W, t], including itself."""
    cum = np.concatenate([[0], np.cumsum(flag)])
    left = np.searchsorted(t, t - window_s, side="right")
    return cum[np.arange(len(t)) + 1] - cum[left]


def build_features(df, cfg=CONFIG):
    """Turn clean login events into detection features.

    Every feature only looks at the CURRENT and PAST events, so the same code works
    in a streaming setting (no look-ahead / data leakage).
    """
    f = df.copy()
    n = len(f)
    t = (f["timestamp"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    failed = (f["success"] == 0).to_numpy().astype(int)
    users = f["user"].to_numpy()
    w_short, w_long = cfg["window_minutes"] * 60.0, cfg["long_window_hours"] * 3600.0

    # -- burst / volume features --------------------------------------------------
    fails_user = np.zeros(n, int)
    for _, idx in f.groupby("user").indices.items():
        fails_user[idx] = _window_counts(t[idx], failed[idx], w_short)

    fails_ip10, fails_ip24 = np.zeros(n, int), np.zeros(n, int)
    dist10, dist24 = np.zeros(n, int), np.zeros(n, int)
    for _, idx in f.groupby("ip").indices.items():
        ti, fi, ui = t[idx], failed[idx], users[idx]
        fails_ip10[idx] = _window_counts(ti, fi, w_short)
        fails_ip24[idx] = _window_counts(ti, fi, w_long)
        if fi.sum() == 0:
            continue
        for out, w in ((dist10, w_short), (dist24, w_long)):
            left = np.searchsorted(ti, ti - w, side="right")
            for k in np.flatnonzero(fi):                       # only failed events need the count
                seg = slice(left[k], k + 1)
                out[idx[k]] = len(set(ui[seg][fi[seg] == 1]))
    f["fails_user_10m"], f["fails_ip_10m"], f["distinct_users_ip_10m"] = fails_user, fails_ip10, dist10
    f["fails_ip_24h"], f["distinct_users_ip_24h"] = fails_ip24, dist24

    # -- per-user history ("is this normal FOR THIS USER?") -----------------------
    s = (f["success"] == 1).astype(int)
    f["prior_successes"] = s.groupby(f["user"]).cumsum() - s
    known = f["prior_successes"] >= cfg["min_history"]
    succ = f["success"] == 1
    for col, name in (("ip", "new_ip"), ("device", "new_device"), ("country", "new_country")):
        first = ~f.loc[succ].duplicated(subset=["user", col])
        f[name] = 0
        f.loc[first[first].index, name] = 1
        f.loc[~known, name] = 0

    first_c = (~f.loc[succ].duplicated(subset=["user", "country"])).astype(int)
    seen = first_c.groupby(f.loc[succ, "user"]).cumsum()
    f["countries_seen_before"] = 0
    f.loc[succ, "countries_seen_before"] = seen - first_c

    hr = f["timestamp"].dt.hour + f["timestamp"].dt.minute / 60
    ang = 2 * np.pi * hr / 24
    cs = (np.sin(ang) * s).groupby(f["user"]).cumsum() - np.sin(ang) * s
    cc = (np.cos(ang) * s).groupby(f["user"]).cumsum() - np.cos(ang) * s
    mean_hr = (np.arctan2(cs, cc) % (2 * np.pi)) * 24 / (2 * np.pi)
    diff = (hr - mean_hr).abs()
    f["hour_dev"] = np.minimum(diff, 24 - diff).where(known, 0.0)

    # -- impossible travel: speed needed between consecutive successful logins ----
    sd = f.loc[succ, ["user", "timestamp", "country"]].copy()
    g = sd.groupby("user")
    sd["prev_country"], sd["prev_time"] = g["country"].shift(1), g["timestamp"].shift(1)
    hours = (sd["timestamp"] - sd["prev_time"]).dt.total_seconds() / 3600
    lat = lambda c: c.map(lambda x: COUNTRY_INFO.get(x, (np.nan,) * 4)[0])
    lon = lambda c: c.map(lambda x: COUNTRY_INFO.get(x, (np.nan,) * 4)[1])
    dist = haversine_km(lat(sd["prev_country"]), lon(sd["prev_country"]), lat(sd["country"]), lon(sd["country"]))
    f["travel_speed_kmh"] = 0.0
    f.loc[sd.index, "travel_speed_kmh"] = (dist / np.maximum(hours, 1 / 60)).fillna(0.0)
    f["log_speed"] = np.log1p(f["travel_speed_kmh"].clip(upper=20000))
    f["failed"] = failed
    return f


# ==== SECTION: ml ====
from sklearn.ensemble import IsolationForest

ML_FEATURES = ["new_ip", "new_device", "new_country", "hour_dev", "log_speed"]


def add_ml_scores(feat, cfg=CONFIG):
    """Unsupervised Isolation Forest on SUCCESSFUL logins (account-takeover view).

    Rules already handle failure bursts well. The model's job is different: learn what a normal
    successful login looks like (known IP/device/country, usual hour, plausible travel) and flag
    rare combinations that no single rule covers. No labels are used.
    """
    f = feat.copy()
    ok = f["success"] == 1
    X = f.loc[ok, ML_FEATURES].astype(float)
    model = IsolationForest(n_estimators=300, contamination=cfg["ml_contamination"],
                            random_state=cfg["seed"]).fit(X)
    raw = -model.score_samples(X)
    f["ml_anomaly"] = 0.0
    f.loc[ok, "ml_anomaly"] = 100 * (raw - raw.min()) / (raw.max() - raw.min())
    f["ml_flag"] = False
    f.loc[ok, "ml_flag"] = model.predict(X) == -1
    q = X.quantile(.97)                       # explanation: which features are unusually high?
    f["ml_reason"] = ""
    for i in X.index[f.loc[X.index, "ml_flag"]]:
        odd = [f"{c}={X.at[i, c]:.4g}" for c in ML_FEATURES if X.at[i, c] > q[c] and X.at[i, c] > 0]
        f.at[i, "ml_reason"] = ", ".join(odd) or "rare combination of features"
    return f, model


# ==== SECTION: scoring ====
RULE_COLS = ["R1_brute_force", "R2_password_spraying", "R3_success_after_failures",
             "R3b_unfamiliar_source", "R4_impossible_travel", "R5_new_device_odd_hour", "R6_slow_spray"]


def score_events(feat, cfg=CONFIG, use_ml=True):
    """Apply the explainable rules, add ML evidence, produce risk score, level and reasons."""
    f, p = feat.copy(), cfg["points"]
    failed, ok = f["success"] == 0, f["success"] == 1
    f["R1_brute_force"] = failed & (f["fails_user_10m"] >= cfg["brute_force_fails"])
    f["R2_password_spraying"] = failed & (f["distinct_users_ip_10m"] >= cfg["spray_users"])
    f["R3_success_after_failures"] = ok & (f["fails_user_10m"] >= cfg["success_after_fails"])
    unfamiliar = (f["new_ip"] + f["new_device"] + f["new_country"]) > 0
    f["R3b_unfamiliar_source"] = f["R3_success_after_failures"] & unfamiliar
    f["R4_impossible_travel"] = ok & (f["travel_speed_kmh"] > cfg["max_speed_kmh"]) \
        & (f["countries_seen_before"] < cfg["travel_max_known_countries"])
    f["R5_new_device_odd_hour"] = ok & (f["new_device"] == 1) & (f["hour_dev"] >= cfg["odd_hour_dev"])
    f["R6_slow_spray"] = failed & (f["distinct_users_ip_24h"] >= cfg["slow_spray_users"]) \
        & (f["fails_ip_24h"] >= cfg["slow_spray_users"])
    ml = f["ml_flag"] if (use_ml and "ml_flag" in f) else pd.Series(False, index=f.index)
    f["ML_anomaly"] = ml.astype(bool)

    score = (f["R1_brute_force"] * p["R1"] + (f["R1_brute_force"] & (f["fails_user_10m"] >= 10)) * p["R1_heavy"]
             + f["R2_password_spraying"] * p["R2"]
             + (f["R2_password_spraying"] & (f["distinct_users_ip_10m"] >= 8)) * p["R2_heavy"]
             + f["R3_success_after_failures"] * p["R3"] + f["R3b_unfamiliar_source"] * p["R3b"]
             + f["R4_impossible_travel"] * p["R4"] + f["R5_new_device_odd_hour"] * p["R5"]
             + f["R6_slow_spray"] * p["R6"] + f["ML_anomaly"] * p["ML"])
    f["risk_score"] = score.clip(upper=100).astype(int)
    f["risk_level"] = pd.cut(f["risk_score"], [-1, 0, cfg["level_medium"] - 1, cfg["level_high"] - 1, 100],
                             labels=["None", "Low", "Medium", "High"]).astype(str)
    f["alert"] = f["risk_score"] >= cfg["alert_threshold"]

    def explain(r):
        out, w = [], cfg["window_minutes"]
        if r.R1_brute_force:
            out.append(f"Brute force: {r.fails_user_10m} failed logins for {r.user} within {w} min")
        if r.R2_password_spraying:
            out.append(f"Password spraying: {r.ip} failed against {r.distinct_users_ip_10m} different accounts within {w} min")
        if r.R3_success_after_failures:
            out.append(f"Successful login after {r.fails_user_10m} recent failures")
        if r.R3b_unfamiliar_source:
            new = [n for n, v in (("IP", r.new_ip), ("device", r.new_device), ("country", r.new_country)) if v]
            out.append("...from a source this user has not used before (new " + "/".join(new) + ")")
        if r.R4_impossible_travel:
            out.append(f"Impossible travel: login from {r.country} implies ~{r.travel_speed_kmh:,.0f} km/h since the previous login")
        if r.R5_new_device_odd_hour:
            out.append(f"New device used {r.hour_dev:.1f} h away from this user's usual login time")
        if r.R6_slow_spray:
            out.append(f"Slow password spraying: {r.ip} failed against {r.distinct_users_ip_24h} accounts in {cfg['long_window_hours']} h")
        if r.ML_anomaly:
            out.append(f"ML anomaly (Isolation Forest): {r.ml_reason}")
        return "; ".join(out)

    f["reasons"], f["rules"] = "", ""
    hit = f.index[f["risk_score"] > 0]
    f.loc[hit, "reasons"] = [explain(r) for r in f.loc[hit].itertuples()]
    names = RULE_COLS + ["ML_anomaly"]
    f.loc[hit, "rules"] = f.loc[hit, names].apply(lambda r: ", ".join(c for c in names if r[c]), axis=1)
    return f


# ==== SECTION: incidents ====
def build_incidents(scored, cfg=CONFIG):
    """Group related alerts (same IP or same user, close in time) into incidents so the
    administrator reviews ONE item per attack instead of dozens of raw alerts."""
    a = scored[scored["alert"]].sort_values("timestamp").copy()
    if a.empty:
        return pd.DataFrame(), a
    gap = pd.Timedelta(minutes=cfg["incident_gap_minutes"])
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    last = {"ip": {}, "user": {}}
    for i, r in zip(a.index, a.itertuples()):
        parent[i] = i
        for kind, key in (("ip", r.ip), ("user", r.user)):
            prev = last[kind].get(key)
            if prev is not None and r.timestamp - prev[1] <= gap:
                parent[find(i)] = find(prev[0])
            last[kind][key] = (i, r.timestamp)
    a["incident_id"] = pd.factorize(pd.Series([find(i) for i in a.index], index=a.index))[0] + 1

    rows = []
    for iid, g in a.groupby("incident_id"):
        top = g.sort_values("risk_score", ascending=False).iloc[0]
        rows.append({
            "incident_id": iid, "start": g["timestamp"].min(), "end": g["timestamp"].max(),
            "n_alerts": len(g), "max_score": int(g["risk_score"].max()),
            "risk_level": top["risk_level"],
            "users": ", ".join(sorted(set(g["user"]))), "ips": ", ".join(sorted(set(g["ip"]))),
            "rules": ", ".join(sorted(set(", ".join(g["rules"]).split(", ")))),
            "summary": top["reasons"],
        })
    return pd.DataFrame(rows), a


# ==== SECTION: storage ====
HASH_FIELDS = ["incident_id", "start", "end", "n_alerts", "max_score", "risk_level", "users", "ips", "rules", "summary"]
STATUSES = ("new", "confirmed", "false_positive", "investigating")


def record_hash(row):
    """SHA-256 fingerprint of an incident's immutable fields (tamper evidence)."""
    payload = json.dumps({k: str(row[k]) for k in HASH_FIELDS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def mask_ip(ip, keep_octets=2):
    """Privacy: show only the network part of an address in dashboards and reports."""
    parts = str(ip).split(".")
    return ".".join(parts[:keep_octets] + ["*"] * (len(parts) - keep_octets)) if len(parts) == 4 else "***"


def save_to_db(db_path, alerts, incidents, meta=None, watchlist=None):
    """Persist alerts + incidents in SQLite (parameter-free pandas writer; no user input is ever
    concatenated into SQL). Passwords/tokens are never part of the schema."""
    cols = ["event_id", "timestamp", "user", "ip", "country", "device", "success",
            "risk_score", "risk_level", "rules", "reasons", "incident_id"]
    al = alerts[cols].copy()
    al["timestamp"] = al["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    inc = incidents.copy()
    for c in ("start", "end"):
        inc[c] = inc[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    inc["record_hash"] = inc.apply(record_hash, axis=1)
    inc["status"], inc["reviewed_by"] = "new", ""
    con = sqlite3.connect(db_path)
    try:
        al.to_sql("alerts", con, if_exists="replace", index=False)
        inc.to_sql("incidents", con, if_exists="replace", index=False)
        if watchlist is not None:
            wl = watchlist[cols[:-1]].copy()
            wl["timestamp"] = wl["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            wl.to_sql("watchlist", con, if_exists="replace", index=False)
        pd.DataFrame([meta or {}]).to_sql("run_meta", con, if_exists="replace", index=False)
        con.execute("CREATE TABLE IF NOT EXISTS audit_log (ts TEXT, reviewer TEXT, incident_id INTEGER, old_status TEXT, new_status TEXT)")
        con.commit()
    finally:
        con.close()


def update_status(db_path, incident_id, new_status, reviewer):
    """Administrator review action. Parameterised SQL + allow-listed values + audit trail."""
    if new_status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    con = sqlite3.connect(db_path)
    try:
        old = con.execute("SELECT status FROM incidents WHERE incident_id = ?", (int(incident_id),)).fetchone()
        if old is None:
            raise KeyError(f"no such incident: {incident_id}")
        con.execute("UPDATE incidents SET status = ?, reviewed_by = ? WHERE incident_id = ?",
                    (new_status, reviewer, int(incident_id)))
        con.execute("INSERT INTO audit_log VALUES (?,?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), reviewer, int(incident_id), old[0], new_status))
        con.commit()
    finally:
        con.close()


def verify_integrity(db_path):
    """Return incident ids whose stored fingerprint no longer matches their content."""
    con = sqlite3.connect(db_path)
    try:
        inc = pd.read_sql_query("SELECT * FROM incidents", con)
    finally:
        con.close()
    return [int(r.incident_id) for r in inc.itertuples() if record_hash(r._asdict()) != r.record_hash]


# ==== SECTION: evaluation ====
def confusion_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp, fp = int((y_true & y_pred).sum()), int((~y_true & y_pred).sum())
    fn, tn = int((y_true & ~y_pred).sum()), int((~y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": prec, "recall": rec,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def evaluate_events(feat, cfg=CONFIG):
    """Event-level comparison of the detection strategies against the ground-truth labels."""
    y = feat["gt_label"] == 1
    rules = score_events(feat, cfg, use_ml=False)
    hybrid = score_events(feat, cfg)
    rows = {
        "Rules only (alerts: Medium+)": rules["alert"],
        "ML only (Isolation Forest)": feat["ml_flag"],
        "Hybrid (alerts: Medium+)": hybrid["alert"],
        "Hybrid incl. Low watchlist": hybrid["risk_score"] >= 1,
    }
    return pd.DataFrame({k: confusion_metrics(y, v) for k, v in rows.items()}).T


def evaluate_incidents(scored, incidents, alerts):
    """Incident-level view: was each real attack noticed, how fast, and how many alerts were noise?"""
    gt = scored[scored["gt_attack_id"] > 0]
    alerted = set(scored.loc[scored["alert"], "event_id"])
    watch = set(scored.loc[scored["risk_score"] >= 1, "event_id"])
    rows = []
    for aid, g in gt.groupby("gt_attack_id"):
        hits = g[g["event_id"].isin(alerted)]
        delay = (hits["timestamp"].min() - g["timestamp"].min()).total_seconds() if len(hits) else np.nan
        rows.append({"attack_id": aid, "type": g["gt_attack_type"].iloc[0], "events": len(g),
                     "detected": len(hits) > 0, "on_watchlist": g["event_id"].isin(watch).any(),
                     "seconds_to_first_alert": delay})
    per_attack = pd.DataFrame(rows)
    by_type = per_attack.groupby("type").agg(attacks=("detected", "size"), detected=("detected", "sum"),
                                             on_watchlist=("on_watchlist", "sum"),
                                             median_seconds_to_alert=("seconds_to_first_alert", "median"))
    by_type["detection_rate"] = by_type["detected"] / by_type["attacks"]
    by_type["detection_or_watchlist_rate"] = by_type["on_watchlist"] / by_type["attacks"]
    if len(incidents):
        real = alerts.groupby("incident_id")["gt_label"].max()
        n_inc, n_real = len(real), int((real == 1).sum())
    else:
        n_inc = n_real = 0
    summary = {"attacks_total": len(per_attack), "attacks_detected": int(per_attack["detected"].sum()),
               "attack_detection_rate": per_attack["detected"].mean(),
               "attack_detection_rate_incl_watchlist": per_attack["on_watchlist"].mean(),
               "incidents_raised": n_inc, "incidents_that_are_real_attacks": n_real,
               "incident_precision": n_real / n_inc if n_inc else 0.0,
               "raw_alerts": int(len(alerts)), "alerts_per_incident": len(alerts) / n_inc if n_inc else 0.0}
    return summary, by_type, per_attack


# ==== SECTION: pipeline ====
def run_pipeline(raw=None, cfg=CONFIG, seed=None):
    seed = cfg["seed"] if seed is None else seed
    cfg = {**cfg, "seed": seed}
    raw = generate_logs(seed=seed) if raw is None else raw
    clean, report = preprocess(raw)
    feat = build_features(clean, cfg)
    feat, model = add_ml_scores(feat, cfg)
    scored = score_events(feat, cfg)
    incidents, alerts = build_incidents(scored, cfg)
    return {"raw": raw, "clean": clean, "report": report, "features": feat, "scored": scored,
            "incidents": incidents, "alerts": alerts, "model": model, "cfg": cfg}


if __name__ == "__main__":
    res = run_pipeline()
    res["raw"].to_csv("simulated_login_logs.csv", index=False)
    meta = {"generated_utc": datetime.now(timezone.utc).isoformat(), "events_analysed": len(res["clean"]),
            **res["report"]}
    watch = res["scored"][(res["scored"]["risk_score"] > 0) & ~res["scored"]["alert"]]
    save_to_db("alerts.db", res["alerts"], res["incidents"], meta, watchlist=watch)
    print("Data quality report:", res["report"])
    print(evaluate_events(res["features"], res["cfg"]).round(3).to_string())
    summ, by_type, _ = evaluate_incidents(res["scored"], res["incidents"], res["alerts"])
    print(summ)
    print(by_type.round(2).to_string())
    print("Saved simulated_login_logs.csv and alerts.db -> run: streamlit run app.py")
