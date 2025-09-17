# app.py
# Streamlit + Supabase tea notes app (matching your latest schema)

import os
from datetime import datetime
import pandas as pd
import streamlit as st
from supabase import create_client, Client

st.set_page_config(page_title="Tea Notes", page_icon="🍵", layout="wide")
st.title("🍵 Tea Notes")

# ---------- Supabase client ----------
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL / SUPABASE_KEY in Streamlit secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------- Helpers ----------
@st.cache_data(ttl=60)
def fetch_teas():
    try:
        res = supabase.table("teas").select("*").order("created_at", desc=True).execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        st.error(f"Error loading teas: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_steeps():
    try:
        res = supabase.table("steeps").select("*").order("session_at", desc=True).execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        st.error(f"Error loading steeps: {e}")
        return pd.DataFrame()

def insert_tea(payload: dict):
    try:
        res = supabase.table("teas").insert(payload).execute()
        return True, res.data
    except Exception as e:
        return False, str(e)

def upsert_tea(payload: dict):
    """Use this if you want idempotent save by (name, pick_year, supplier) uniqueness later."""
    try:
        res = supabase.table("teas").upsert(payload).execute()
        return True, res.data
    except Exception as e:
        return False, str(e)

def insert_steep(payload: dict):
    try:
        res = supabase.table("steeps").insert(payload).execute()
        return True, res.data
    except Exception as e:
        return False, str(e)


# ---------- UI ----------
tab_add_tea, tab_add_steep, tab_browse = st.tabs(["Add Tea", "Add Steep", "Browse"])

with tab_add_tea:
    st.subheader("Add Tea")

    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Name*", placeholder="e.g., Jin Jun Mei")
        type_ = st.text_input("Type", placeholder="Black / Oolong / Green / ...")
        subtype = st.text_input("Subtype", placeholder="e.g., Tong Mu, Rou Gui, Dan Cong")
        processing_notes = st.text_area("Processing notes", placeholder="Brief processing info")

    with col2:
        oxidation = st.text_input("Oxidation", placeholder="e.g., light / medium / heavy")
        roasting = st.text_input("Roasting", placeholder="e.g., none / light / medium / dark")
        cultivar = st.text_input("Cultivar", placeholder="e.g., Jin Guanyin, Tieguanyin")
        region = st.text_input("Region", placeholder="e.g., Wuyi, Fujian, China")

    with col3:
        elevation_m = st.text_input("Elevation (m)", placeholder="e.g., 1200–1500")
        picking_season = st.text_input("Picking season", placeholder="e.g., Spring")
        pick_year = st.number_input("Pick year", min_value=1900, max_value=2100, value=2025, step=1)
        supplier = st.text_input("Supplier", placeholder="e.g., Teavivre")

    if st.button("Save Tea", type="primary", use_container_width=True, disabled=(name.strip() == "")):
        payload = {
            "name": name.strip(),
            "type": type_.strip() or None,
            "subtype": subtype.strip() or None,
            "processing_notes": processing_notes.strip() or None,
            "oxidation": oxidation.strip() or None,
            "roasting": roasting.strip() or None,
            "cultivar": cultivar.strip() or None,
            "region": region.strip() or None,
            "elevation_m": elevation_m.strip() or None,   # text per your schema
            "picking_season": picking_season.strip() or None,
            "pick_year": int(pick_year) if pick_year else None,
            "supplier": supplier.strip() or None,
            # created_at auto
        }
        ok, res = insert_tea(payload)
        if ok:
            st.success("Tea saved ✅")
            fetch_teas.clear()  # refresh cache
        else:
            st.error(f"Failed to save tea: {res}")

with tab_add_steep:
    st.subheader("Add Steep")

    teas_df = fetch_teas()
    if teas_df.empty:
        st.info("No teas yet. Add a tea first.")
    else:
        # Choice label shows both name and supplier/year if present
        teas_df["label"] = teas_df.apply(
            lambda r: f"{r.get('name','(no name)')} — {r.get('supplier','?')} ({r.get('pick_year','?')})",
            axis=1
        )
        choice = st.selectbox("Tea*", options=teas_df["label"].tolist(), index=0)
        chosen_row = teas_df.loc[teas_df["label"] == choice].iloc[0]
        chosen_tea_id = chosen_row["tea_id"]

        col1, col2, col3 = st.columns(3)
        with col1:
            rating = st.number_input("Rating (0–10, 0.1 step)", min_value=0.0, max_value=10.0, value=8.0, step=0.1, format="%.1f")
            steeps_count = st.number_input("Steeps (count in session)", min_value=1, max_value=50, value=1, step=1)
            temperature_c = st.number_input("Temperature °C", min_value=0, max_value=100, value=95, step=1)
        with col2:
            initial_steep_time_sec = st.number_input("Initial steep time (sec)", min_value=0, max_value=3600, value=15, step=1)
            amount_used_g = st.number_input("Amount used (g)", min_value=0.0, max_value=200.0, value=5.0, step=0.1)
            session_at = st.datetime_input("Session time", value=datetime.now())
        with col3:
            steep_time_changes = st.text_input("Steep time changes", placeholder="e.g., +5s each steep, 15/20/25s ...")
            tasting_notes = st.text_area("Tasting notes", placeholder="aroma, taste, aftertaste, texture, energy...")

        if st.button("Save Steep", type="primary", use_container_width=True):
            payload = {
                "tea_id": chosen_tea_id,
                "tasting_notes": tasting_notes.strip() or None,
                "rating": float(rating),                # numeric(2,1)
                "steeps": int(steeps_count),
                "initial_steep_time_sec": int(initial_steep_time_sec),
                "steep_time_changes": steep_time_changes.strip() or None,
                "temperature_c": int(temperature_c),
                "amount_used_g": float(amount_used_g),
                "session_at": session_at.isoformat(),
            }
            ok, res = insert_steep(payload)
            if ok:
                st.success("Steep saved ✅")
                fetch_steeps.clear()
            else:
                st.error(f"Failed to save steep: {res}")

with tab_browse:
    st.subheader("Browse & Analyze")

    teas_df = fetch_teas()
    steeps_df = fetch_steeps()

    # Filters
    with st.expander("Filters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            tea_filter = st.multiselect(
                "Filter by tea",
                options=teas_df["name"].tolist() if not teas_df.empty else [],
                default=[]
            )
        with c2:
            min_year, max_year = (int(teas_df["pick_year"].min()), int(teas_df["pick_year"].max())) if ("pick_year" in teas_df and not teas_df["pick_year"].dropna().empty) else (2000, 2100)
            year_range = st.slider("Pick year range", min_year, max_year, (min_year, max_year))
        with c3:
            supplier_filter = st.multiselect(
                "Supplier",
                options=sorted(teas_df["supplier"].dropna().unique().tolist()) if not teas_df.empty else [],
                default=[]
            )

    # Apply filters to teas
    teas_filtered = teas_df.copy()
    if tea_filter:
        teas_filtered = teas_filtered[teas_filtered["name"].isin(tea_filter)]
    if "pick_year" in teas_filtered:
        teas_filtered = teas_filtered[(teas_filtered["pick_year"].fillna(0) >= year_range[0]) &
                                      (teas_filtered["pick_year"].fillna(9999) <= year_range[1])]
    if supplier_filter:
        teas_filtered = teas_filtered[teas_filtered["supplier"].isin(supplier_filter)]

    st.markdown("### Teas")
    st.dataframe(teas_filtered, use_container_width=True)

    # Join steeps with tea names for display
    st.markdown("### Steeps")
    if not steeps_df.empty:
        if not teas_df.empty and "tea_id" in teas_df.columns:
            merged = steeps_df.merge(
                teas_df[["tea_id", "name", "supplier", "pick_year"]],
                how="left", on="tea_id"
            )
        else:
            merged = steeps_df.copy()

        # If tea filter applied, restrict steeps to those teas
        if tea_filter and "name" in merged.columns:
            merged = merged[merged["name"].isin(tea_filter)]

        st.dataframe(merged, use_container_width=True)

        # Simple charts
        st.markdown("#### Charts")
        left, right = st.columns(2)
        with left:
            if "name" in merged.columns:
                counts = merged.groupby("name")["steep_id"].count().sort_values(ascending=False)
                if not counts.empty:
                    st.caption("Steeps per tea")
                    st.bar_chart(counts)
        with right:
            if "rating" in merged.columns:
                ratings = merged.dropna(subset=["rating"])
                if not ratings.empty:
                    avg_r = ratings.groupby("name")["rating"].mean().sort_values(ascending=False)
                    st.caption("Average rating per tea")
                    st.bar_chart(avg_r)
    else:
        st.info("No steeps logged yet.")
