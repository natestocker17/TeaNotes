import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st
import plotly.express as px

# Optional Supabase
try:
    from supabase import create_client, Client  # type: ignore
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

st.set_page_config(page_title="Tea Notes (Steeps)", page_icon="🍵", layout="wide")
st.title("🍵 Tea Notes — Sessions & Scores")

# -------------------- Config & Data Access --------------------

TEA_TYPES = ["Oolong", "Black", "White", "Green", "Pu-erh", "Dark", "Yellow"]
ROASTING_OPTIONS = ["Unroasted", "Roasted", "Light", "Medium", "Heavy"]

@st.cache_resource
def get_supabase() -> Optional["Client"]:
    if not SUPABASE_AVAILABLE:
        return None
    url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)  # type: ignore
    except Exception:
        return None

SUPABASE = get_supabase()

@st.cache_data(ttl=60)
def load_data() -> Dict[str, pd.DataFrame]:
    if SUPABASE is None:
        return {"teas": pd.DataFrame(), "steeps": pd.DataFrame()}
    teas = pd.DataFrame(SUPABASE.table("teas").select("*").execute().data)  # type: ignore
    steeps = pd.DataFrame(SUPABASE.table("steeps").select("*").execute().data)  # type: ignore
    return {"teas": teas, "steeps": steeps}

db = load_data()
teas_df = db["teas"].copy()
steeps_df = db["steeps"].copy()

# -------------------- Helpers --------------------

def ensure_datetime(col: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(col, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None]*len(col)))

def options_from_column(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    opts = sorted(vals.unique().tolist())
    return opts

def safe_int(text: str) -> Optional[int]:
    text = (text or "").strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except Exception:
        return None

def safe_float(text: str) -> Optional[float]:
    text = (text or "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None

def plot_sessions_with_average(df: pd.DataFrame, title: str = "Session ratings"):
    if df.empty or "rating" not in df.columns:
        st.info("No sessions to chart yet.")
        return
    working = df.copy()
    working["session_at"] = ensure_datetime(working.get("session_at", pd.Series(dtype="datetime64[ns]")))
    working = working.dropna(subset=["rating"])
    if working.empty:
        st.info("No ratings available to chart yet.")
        return
    avg = working["rating"].mean()
    fig = px.line(
        working.sort_values("session_at"),
        x="session_at",
        y="rating",
        color="name",
        markers=True,
        title=title
    )
    fig.update_traces(mode="lines+markers", hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Rating %{y}<extra></extra>")
    fig.add_hline(y=avg, line_dash="dash", annotation_text=f"Average: {avg:.1f}", annotation_position="top left", opacity=0.7)
    fig.update_layout(
        margin=dict(l=12, r=12, t=48, b=12),
        legend=dict(title="Tea", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_xaxes(title_text="Session time")
    fig.update_yaxes(title_text="Rating (0–100)")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Tabs: 1) Add Session (default)  2) Add Tea  3) Browse --------------------

tabs = st.tabs(["📝 Add Session", "➕ Add Tea", "🔎 Browse"])

# ---------- Tab 1: Add Session (default) ----------
with tabs[0]:
    tea_choices = ["(select)"] + teas_df.get("name", pd.Series(dtype=str)).fillna("(unnamed)").tolist()
    tea_selected = st.selectbox("Tea", tea_choices, index=0)
    tea_selected_row = None
    if tea_selected != "(select)" and "name" in teas_df.columns:
        tea_selected_row = teas_df[teas_df["name"] == tea_selected].head(1)
    tea_id = None
    if tea_selected_row is not None and not tea_selected_row.empty:
        tea_id = tea_selected_row.iloc[0].get("tea_id")

    # All optional except Tea
    initial_secs_txt = st.text_input("Initial steep time (seconds)", value="")
    changes_text = st.text_input("Steep time changes", value="")
    temperature_c_txt = st.text_input("Water temperature (°C)", value="")
    amount_used_g_txt = st.text_input("Tea amount used (g)", value="")
    tasting_notes = st.text_area("Tasting notes", value="")
    steep_notes = st.text_area("Steep notes", value="")
    overall_rating_txt = st.text_input("Overall rating (0–5)", value="")

    save_session_btn = st.button("Save Session", type="primary", use_container_width=True)
    if save_session_btn:
        if tea_id is None:
            st.error("Please select a tea first.")
        elif SUPABASE is None:
            st.error("Database is not configured.")
        else:
            row = {
                "tea_id": tea_id,
                "tasting_notes": (tasting_notes or None),
                "steep_notes": (steep_notes or None),
                "rating": safe_float(overall_rating_txt),
                "steeps": None,
                "initial_steep_time_sec": safe_int(initial_secs_txt),
                "steep_time_changes": (changes_text or None),
                "temperature_c": safe_int(temperature_c_txt),
                "amount_used_g": safe_float(amount_used_g_txt),
                "session_at": datetime.utcnow().isoformat()
            }
            try:
                SUPABASE.table("steeps").insert(row).execute()  # type: ignore
                st.success("Saved.")
            except Exception as e:
                st.error(f"Failed to save: {e}")

# ---------- Tab 2: Add Tea ----------
with tabs[1]:
    colA, colB = st.columns(2)

    # Pre-populated options from existing data
    subtype_opts = options_from_column(teas_df, "subtype")
    supplier_opts = options_from_column(teas_df, "supplier")
    cultivar_opts = options_from_column(teas_df, "cultivar")
    region_opts = options_from_column(teas_df, "region")

    with colA:
        tea_name = st.text_input("Tea name (required)")
        tea_type = st.selectbox("Tea type", options=[""] + TEA_TYPES, index=0)
        subtype_sel = st.selectbox("Subtype", options=[""] + subtype_opts, index=0)
        subtype_new = st.text_input("Or add new Subtype")
        supplier_sel = st.selectbox("Supplier", options=[""] + supplier_opts, index=0)
        supplier_new = st.text_input("Or add new Supplier")
        url = st.text_input("URL")
    with colB:
        cultivar_sel = st.selectbox("Cultivar", options=[""] + cultivar_opts, index=0)
        cultivar_new = st.text_input("Or add new Cultivar")
        region_sel = st.selectbox("Region", options=[""] + region_opts, index=0)
        region_new = st.text_input("Or add new Region")
        current_year = datetime.now().year
        pick_year_txt = st.text_input(f"Pick year", value="")
        oxidation = st.text_input("Oxidation")
        roasting = st.selectbox("Roasting", options=[""] + ROASTING_OPTIONS, index=0)

    # Resolve chosen vs new values (all optional)
    subtype = (subtype_new.strip() or subtype_sel.strip() or None)
    supplier = (supplier_new.strip() or supplier_sel.strip() or None)
    cultivar = (cultivar_new.strip() or cultivar_sel.strip() or None)
    region = (region_new.strip() or region_sel.strip() or None)
    pick_year = safe_int(pick_year_txt) if pick_year_txt else None
    roasting_val = roasting.strip() or None
    tea_type_val = tea_type.strip() or None

    add_tea_btn = st.button("Save Tea", type="primary", use_container_width=True)
    if add_tea_btn:
        if not tea_name.strip():
            st.error("Tea name is required.")
        elif SUPABASE is None:
            st.error("Database is not configured.")
        else:
            tea_row = {
                "name": tea_name.strip(),
                "type": tea_type_val,
                "subtype": subtype,
                "supplier": supplier,
                "URL": (url.strip() or None),
                "cultivar": cultivar,
                "region": region,
                "pick_year": pick_year,
                "oxidation": (oxidation.strip() or None),
                "roasting": roasting_val,
                "created_at": datetime.utcnow().isoformat()
            }
            try:
                SUPABASE.table("teas").insert(tea_row).execute()  # type: ignore
                st.success("Saved.")
            except Exception as e:
                st.error(f"Failed to save: {e}")

# ---------- Tab 3: Browse ----------
with tabs[2]:
    col1, col2 = st.columns([1,1])
    with col1:
        tea_type_filter = st.selectbox("Tea type", options=["(all)"] + TEA_TYPES, index=0)
    with col2:
        supplier_filter = st.text_input("Supplier contains")

    if "tea_id" in steeps_df.columns and "tea_id" in teas_df.columns:
        joined = steeps_df.merge(teas_df[["tea_id","name","type","supplier"]], on="tea_id", how="left")
    else:
        joined = steeps_df.copy()
        joined["name"] = None
        joined["type"] = None
        joined["supplier"] = None

    joined["session_at"] = ensure_datetime(joined.get("session_at", pd.Series(dtype="datetime64[ns]")))

    mask = pd.Series([True]*len(joined))
    if tea_type_filter != "(all)":
        mask &= (joined["type"].fillna("").str.lower() == tea_type_filter.lower())
    if supplier_filter:
        mask &= joined["supplier"].fillna("").str.contains(supplier_filter, case=False, na=False)

    joined_filt = joined[mask].copy()

    st.markdown("### Session Ratings")
    plot_sessions_with_average(joined_filt, title="Session Ratings")

    st.markdown("### Data")
    st.dataframe(joined_filt.sort_values("session_at", ascending=False), use_container_width=True)
