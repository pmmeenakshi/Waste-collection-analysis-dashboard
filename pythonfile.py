# pythonfile.py — Interactive Waste Management Dashboard (final tweaks)
import streamlit as st
import pandas as pd
import numpy as np
import re
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="Interactive Waste Management Dashboard", layout="wide")

# ---------- CONFIG ----------
CSV_PATH = r"C:\Users\meena\Downloads\kaggle\INTERNSHIP TASK 1\percent_to_numbers.csv"
META_COLS = [
    "City", "Community", "Community Status", "Pincode", "Type",
    "Inactive Registrations", "Active Registrations", "Lat", "Lon"
]

def fmt_num(x, digits=1):
    try:
        x = float(x)
        if np.isnan(x) or not np.isfinite(x):
            return "N/A"
        return f"{x:.{digits}f}"
    except Exception:
        return "N/A"

# ---------- LOAD & PREP (pairs duplicate date headers: first=Kgs, second=Participation) ----------
@st.cache_data(show_spinner=True)
def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)

    # Drop unnamed stray columns if any
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, na=False)]

    meta_cols = [
        "City","Community","Community Status","Pincode","Type",
        "Inactive Registrations","Active Registrations","Lat","Lon"
    ]

    # Build buckets: each date label (dd-mm-YYYY) may appear twice: [kgs_col, part_col]
    ts_cols = [c for c in df.columns if c not in meta_cols]
    buckets = {}
    order = []
    for c in ts_cols:
        base = re.sub(r"\.\d+$", "", str(c)).strip()   # collapse ".1"
        if not re.fullmatch(r"\d{2}-\d{2}-\d{4}", base):
            continue
        if base not in buckets:
            buckets[base] = []
            order.append(base)
        buckets[base].append(c)

    base_df = df[meta_cols].copy()
    blocks = []
    for label in order:
        cols = buckets[label]
        kgs_col = cols[0] if len(cols) >= 1 else None
        part_col = cols[1] if len(cols) >= 2 else None

        block = base_df.copy()
        block["Date"] = pd.to_datetime(label, format="%d-%m-%Y", errors="coerce")

        # Kgs
        if kgs_col is not None:
            block["Kgs"] = pd.to_numeric(df[kgs_col], errors="coerce")
        else:
            block["Kgs"] = np.nan

        # Participation (0–1 → %)
        if part_col is not None:
            p = pd.to_numeric(df[part_col], errors="coerce")
            med = p.dropna().median()
            if pd.notna(med) and 0 <= med <= 1:
                p = p * 100.0
            block["Participation_Percent"] = p
        else:
            # Fallback from Active/Inactive if needed
            if {"Active Registrations","Inactive Registrations"}.issubset(block.columns):
                a = pd.to_numeric(block["Active Registrations"], errors="coerce")
                i = pd.to_numeric(block["Inactive Registrations"], errors="coerce")
                tot = a + i
                block["Participation_Percent"] = np.where(tot > 0, (a/tot)*100.0, np.nan)
            else:
                block["Participation_Percent"] = np.nan

        blocks.append(block)

    data = pd.concat(blocks, ignore_index=True)

    # Lat/Lon numeric
    data["Lat"] = pd.to_numeric(data["Lat"], errors="coerce")
    data["Lon"] = pd.to_numeric(data["Lon"], errors="coerce")

    # Final safety
    med_all = data["Participation_Percent"].dropna().median()
    if pd.notna(med_all) and 0 <= med_all <= 1:
        data["Participation_Percent"] = data["Participation_Percent"] * 100.0

    return data

data = load_and_prepare(CSV_PATH)

# ---------- SIDEBAR ----------
st.sidebar.title("Filters")

# Date range
min_d = pd.to_datetime(data["Date"].min()).date() if not data.empty else date(2024, 1, 1)
max_d = pd.to_datetime(data["Date"].max()).date() if not data.empty else date.today()
dr = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(dr, tuple) and len(dr) == 2:
    start_d, end_d = dr
else:
    start_d, end_d = min_d, max_d

# City filter (NEW)
cities = sorted([c for c in data["City"].dropna().unique()])
selected_cities = st.sidebar.multiselect("City", options=cities, default=cities)

# Existing controls
metric = st.sidebar.selectbox("Color/Size by", ["Participation %", "Waste (Kgs)"], index=0)
agg_level = st.sidebar.selectbox("Aggregate by", ["Community", "Pincode"], index=0)

# ---------- FILTER & AGG ----------
filtered = data[
    (data["Date"].dt.date >= start_d) &
    (data["Date"].dt.date <= end_d) &
    (data["City"].isin(selected_cities))
].copy()

group_cols = []
if "City" in filtered.columns: group_cols.append("City")
if agg_level == "Community" and "Community" in filtered.columns: group_cols.append("Community")
if agg_level == "Pincode" and "Pincode" in filtered.columns: group_cols.append("Pincode")
if "Lat" in filtered.columns: group_cols.append("Lat")
if "Lon" in filtered.columns: group_cols.append("Lon")
if "Active Registrations" in filtered.columns: group_cols.append("Active Registrations")
if "Inactive Registrations" in filtered.columns: group_cols.append("Inactive Registrations")

agg = filtered.groupby(group_cols, dropna=True).agg({
    "Kgs": "sum",
    "Participation_Percent": "mean"
}).reset_index()

# Safety: fraction → percent if needed
if "Participation_Percent" in agg.columns and not agg["Participation_Percent"].dropna().empty:
    med_agg = agg["Participation_Percent"].dropna().median()
    if 0 <= med_agg <= 1:
        agg["Participation_Percent"] = agg["Participation_Percent"] * 100

for c in ["Active Registrations", "Inactive Registrations"]:
    if c in agg.columns:
        agg[c] = pd.to_numeric(agg[c], errors="coerce").astype("Int64")

# ---------- UI ----------
st.title("Interactive Waste Management Dashboard")
st.caption("Date-filtered mapping with color/size encoding, popups, and bottom charts")

# ===== MAP (FULL WIDTH) =====
st.subheader("Map")
if agg.empty:
    st.info("No data in selected date range / city.")
else:
    center_lat = agg["Lat"].mean() if "Lat" in agg.columns else 20.5937
    center_lon = agg["Lon"].mean() if "Lon" in agg.columns else 78.9629

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    mc = MarkerCluster().add_to(m)

    values = agg["Participation_Percent"] if metric == "Participation %" else agg["Kgs"]
    vmin = float(np.nanmin(values)) if not values.isna().all() else 0.0
    vmax = float(np.nanmax(values)) if not values.isna().all() else 1.0
    if vmax == vmin:
        vmax = vmin + 1.0

    def color_scale(v):
        if pd.isna(v): return "#808080"
        x = (v - vmin) / (vmax - vmin)
        if x >= 2/3: return "#2ECC71"
        if x >= 1/3: return "#F1C40F"
        return "#E74C3C"

    def radius(v):
        if pd.isna(v): return 6
        return 6 + 18 * ((v - vmin) / (vmax - vmin))

    for _, r in agg.iterrows():
        title = r.get("Community", "") if agg_level == "Community" else str(r.get("Pincode", ""))
        p = float(r.get("Participation_Percent", np.nan))
        k = float(r.get("Kgs", np.nan))
        act = r.get("Active Registrations", pd.NA)
        ina = r.get("Inactive Registrations", pd.NA)

        col_value = p if metric == "Participation %" else k
        col = color_scale(col_value)
        rad = radius(k if metric == "Waste (Kgs)" else col_value)

        popup_html = f"""
        <div style="font-size:14px;">
          <b>{agg_level}:</b> {title}<br>
          <b>Active:</b> {act if pd.notna(act) else '—'}<br>
          <b>Inactive:</b> {ina if pd.notna(ina) else '—'}<br>
          <b>Participation %:</b> {fmt_num(p, 1)}<br>
          <b>Total Kgs:</b> {fmt_num(k, 1)}
        </div>
        """
        tooltip_txt = f"{title} — {('Part %' if metric=='Participation %' else 'Kgs')}: {fmt_num(col_value, 1)}"

        folium.CircleMarker(
            location=[r["Lat"], r["Lon"]],
            radius=rad,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip_txt
        ).add_to(mc)

    st_folium(m, width=None, height=560)

# ===== BOTTOM CHARTS (NO PIE) =====
st.subheader("Summary charts")
if filtered.empty:
    st.info("No data to chart for selected range / city.")
else:
    # Prepare time series
    ts_part = (
        filtered.groupby("Date")["Participation_Percent"]
        .mean()
        .reset_index()
        .sort_values("Date")
    )
    ts_kgs  = (
        filtered.groupby("Date")["Kgs"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    # If 0–1, convert to %
    if not ts_part["Participation_Percent"].dropna().empty:
        med_ts = ts_part["Participation_Percent"].dropna().median()
        if 0 <= med_ts <= 1:
            ts_part["Participation_Percent"] = ts_part["Participation_Percent"] * 100

    # Place charts at the bottom in two columns
    c1, c2 = st.columns(2)

    with c1:
        fig1, ax1 = plt.subplots()
        ax1.plot(ts_part["Date"], ts_part["Participation_Percent"], marker="o")
        ax1.set_title("Avg Participation % over time")
        ax1.set_ylabel("Participation (%)")
        ax1.grid(True)
        st.pyplot(fig1, use_container_width=True)

    with c2:
        fig2, ax2 = plt.subplots()
        xlabels = ts_kgs["Date"].dt.strftime("%d-%b-%Y")
        ax2.bar(xlabels, ts_kgs["Kgs"])
        ax2.set_title("Total Kgs over time")
        ax2.set_ylabel("Kgs")
        ax2.tick_params(axis="x", labelrotation=75)
        ax2.grid(axis="y")
        st.pyplot(fig2, use_container_width=True)

# (Optional quick peek)
with st.expander("Data sample (filtered)"):
    st.dataframe(filtered.head(10), use_container_width=True)
