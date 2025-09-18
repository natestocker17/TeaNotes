import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit import column_config  # OK even if not strictly needed

# Optional Supabase
try:
    from supabase import create_client, Client  # type: ignore
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

st.set_page_config(page_title="Tea Notes (Steeps)", page_icon="🍵", layout="wide")
st.title("🍵 Tea Notes — Sessions & Scores")

# -------------------- CSS --------------------
st.markdown("""
<style>
/* Make the radio look like tabs */
[data-testid="stHorizontalBlock"] > div:has(> div[data-testid="stRadio"]) { margin-bottom: 0.5rem; }
div[data-testid="stRadio"] > div[role="radiogroup"] {
  display: flex; gap: .25rem; flex-wrap: wrap;
}
div[data-testid="stRadio"] label {
  border: 1px solid var(--secondary-background-color);
  padding: .4rem .8rem; border-radius: .5rem .5rem 0 0;
  background: var(--secondary-background-color); cursor: pointer;
  font-weight: 500;
}
div[data-testid="stRadio"] label[data-checked="true"] {
  background: var(--background-color);
  border-bottom-color: var(--background-color);
  box-shadow: 0 -2px 0 0 var(--primary-color) inset;
}

/* Force text wrapping in DataFrame/Data Editor cells */
[data-testid="stDataFrame"] div[role="gridcell"],
[data-testid="stDataFrame"] div[data-testid="cell-container"],
[data-testid="stDataFrame"] td, 
[data-testid="stDataFrame"] span, 
[data-testid="stDataFrame"] p {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
}
</style>
""", unsafe_allow_html=True)

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
    return sorted(vals.unique().tolist())

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

# -------------------- Box & whisker helper --------------------
def plot_box_by_tea(df, title: str = "Tea ratings — box & whisker"):
    """
    Expects a DataFrame with columns: name, rating (0–5 recommended).
    Shows a box-and-whisker plot per tea and overlays individual points.
    """
    if df is None or len(df) == 0:
        st.info("No data to chart yet.")
        return

    for col in ["name", "rating"]:
        if col not in df.columns:
            st.info("No ratings to chart yet.")
            return

    working = df.copy()
    working["rating"] = pd.to_numeric(working["rating"], errors="coerce")

    for col in ["supplier", "type", "session_at", "tasting_notes", "steep_notes"]:
        if col not in working.columns:
            working[col] = None

    working = working.dropna(subset=["rating"])
    if working.empty:
        st.info("No ratings to chart yet.")
        return

    medians = working.groupby("name")["rating"].median().sort_values(ascending=False)
    category_order = medians.index.tolist()

    fig = px.box(
        working,
        x="name",
        y="rating",
        points="all",
        hover_data=["supplier", "type", "session_at", "tasting_notes", "steep_notes"],
        title=title,
        category_orders={"name": category_order},
    )
    fig.update_layout(
        margin=dict(l=12, r=12, t=48, b=12),
        xaxis_title="Tea",
        yaxis_title="Rating",
        hovermode="closest",
    )
    fig.update_yaxes(range=[0, 5])
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Sticky tab-like nav (no jumping; no extra buttons) --------------------
NAV_ITEMS = ["📝 Add Session", "➕ Add Tea", "📜 Steep history", "📊 Analysis"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = NAV_ITEMS[0]

st.session_state.active_tab = st.radio(
    "Tabs",
    NAV_ITEMS,
    index=NAV_ITEMS.index(st.session_state.active_tab),
    horizontal=True,
    label_visibility="collapsed",
    key="nav_radio",
)

# -------------------- Screens --------------------

if st.session_state.active_tab == "📝 Add Session":
    # ---------- Add Session (unchanged) ----------
    tea_choices = ["(select)"] + teas_df.get("name", pd.Series(dtype=str)).fillna("(unnamed)").tolist()
    tea_selected = st.selectbox("Tea", tea_choices, index=0, key="add_sess_tea")
    tea_selected_row = None
    if tea_selected != "(select)" and "name" in teas_df.columns:
        tea_selected_row = teas_df[teas_df["name"] == tea_selected].head(1)
    tea_id = None
    if tea_selected_row is not None and not tea_selected_row.empty:
        tea_id = tea_selected_row.iloc[0].get("tea_id")

    initial_secs_txt = st.text_input("Initial steep time (seconds)", value="", key="add_sess_initial_secs")
    changes_text = st.text_input("Steep time changes", value="", key="add_sess_changes")
    temperature_c_txt = st.text_input("Water temperature (°C)", value="", key="add_sess_temp")
    amount_used_g_txt = st.text_input("Tea amount used (g)", value="", key="add_sess_amount")
    tasting_notes = st.text_area("Tasting notes", value="", key="add_sess_tnotes")
    steep_notes = st.text_area("Steep notes", value="", key="add_sess_snotes")
    overall_rating_txt = st.text_input("Overall rating (0–5)", value="", key="add_sess_rating")

    save_session_btn = st.button("Save Session", type="primary", use_container_width=True, key="add_sess_save")
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

elif st.session_state.active_tab == "➕ Add Tea":
    # ---------- Add Tea (unchanged) ----------
    colA, colB = st.columns(2)

    subtype_opts = options_from_column(teas_df, "subtype")
    supplier_opts_all = options_from_column(teas_df, "supplier")
    cultivar_opts = options_from_column(teas_df, "cultivar")
    region_opts = options_from_column(teas_df, "region")

    with colA:
        tea_name = st.text_input("Tea name (required)", key="add_tea_name")
        tea_type = st.selectbox("Tea type", options=[""] + TEA_TYPES, index=0, key="add_tea_type")
        subtype_sel = st.selectbox("Subtype", options=[""] + subtype_opts, index=0, key="add_tea_subtype_sel")
        subtype_new = st.text_input("Or add new Subtype", key="add_tea_subtype_new")
        supplier_sel = st.selectbox("Supplier", options=[""] + supplier_opts_all, index=0, key="add_tea_supplier_sel")
        supplier_new = st.text_input("Or add new Supplier", key="add_tea_supplier_new")
        url = st.text_input("URL", key="add_tea_url")
    with colB:
        cultivar_sel = st.selectbox("Cultivar", options=[""] + cultivar_opts, index=0, key="add_tea_cultivar_sel")
        cultivar_new = st.text_input("Or add new Cultivar", key="add_tea_cultivar_new")
        region_sel = st.selectbox("Region", options=[""] + region_opts, index=0, key="add_tea_region_sel")
        region_new = st.text_input("Or add new Region", key="add_tea_region_new")
        pick_year_txt = st.text_input("Pick year", value="", key="add_tea_pick_year")
        oxidation = st.text_input("Oxidation", key="add_tea_oxidation")
        roasting = st.selectbox("Roasting", options=[""] + ROASTING_OPTIONS, index=0, key="add_tea_roasting")

    subtype = (subtype_new.strip() or subtype_sel.strip() or None)
    supplier = (supplier_new.strip() or supplier_sel.strip() or None)
    cultivar = (cultivar_new.strip() or cultivar_sel.strip() or None)
    region = (region_new.strip() or region_sel.strip() or None)
    pick_year = safe_int(pick_year_txt) if pick_year_txt else None
    roasting_val = roasting.strip() or None
    tea_type_val = tea_type.strip() or None

    add_tea_btn = st.button("Save Tea", type="primary", use_container_width=True, key="add_tea_save")
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

elif st.session_state.active_tab == "📜 Steep history":
    # ---------- Steep history ----------
    st.subheader("📜 Steep history")

    # Join steeps to teas for richer display
    if "tea_id" in steeps_df.columns and "tea_id" in teas_df.columns:
        joined = steeps_df.merge(
            teas_df[["tea_id", "name", "type", "supplier", "region", "cultivar", "roasting"]],
            on="tea_id", how="left"
        )
    else:
        joined = steeps_df.copy()
        for col in ["name", "type", "supplier", "region", "cultivar", "roasting"]:
            if col not in joined.columns:
                joined[col] = None

    joined["session_at"] = ensure_datetime(joined.get("session_at", pd.Series(dtype="datetime64[ns]")))
    joined = joined.sort_values("session_at", ascending=False)

    tea_names = (
        teas_df.get("name", pd.Series(dtype=str)).dropna().astype(str).str.strip()
    )
    tea_names = tea_names[tea_names != ""].unique().tolist()
    tea_names_sorted = sorted(tea_names)

    selected_tea = st.selectbox("Find a tea", options=["(select a tea)"] + tea_names_sorted, index=0, key="hist_select_tea")

    st.markdown("### Steeping notes")
    if selected_tea == "(select a tea)":
        st.info("Select a tea above to see all of its steeping notes.")
    else:
        tea_rows = joined[joined["name"] == selected_tea].copy()
        if tea_rows.empty:
            st.warning("No sessions found for this tea yet.")
        else:
            cols = [
                "session_at", "rating",
                "tasting_notes", "steep_notes",
                "initial_steep_time_sec", "steep_time_changes",
                "temperature_c", "amount_used_g",
                "type", "supplier", "region", "cultivar", "roasting",
            ]
            present_cols = [c for c in cols if c in tea_rows.columns]
            tea_rows = tea_rows[present_cols].sort_values("session_at", ascending=False)
            rename_map = {
                "session_at": "Session time",
                "rating": "Rating",
                "tasting_notes": "Tasting notes",
                "steep_notes": "Steep notes",
                "initial_steep_time_sec": "Initial steep (sec)",
                "steep_time_changes": "Steep time changes",
                "temperature_c": "Water temp (°C)",
                "amount_used_g": "Amount (g)",
                "type": "Type",
                "supplier": "Supplier",
                "region": "Region",
                "cultivar": "Cultivar",
                "roasting": "Roasting",
            }
            tea_rows = tea_rows.rename(columns=rename_map)

            # Use read-only data editor; CSS above ensures wrapping
            st.data_editor(
                tea_rows,
                use_container_width=True,
                disabled=True,
                hide_index=False,
                column_config={
                    "Tasting notes": st.column_config.TextColumn("Tasting notes"),
                    "Steep notes": st.column_config.TextColumn("Steep notes"),
                },
            )

elif st.session_state.active_tab == "📊 Analysis":
    # ---------- Analysis ----------
    st.subheader("📊 Analysis")

    # Join for charting
    if "tea_id" in steeps_df.columns and "tea_id" in teas_df.columns:
        joined = steeps_df.merge(
            teas_df[["tea_id", "name", "type", "supplier", "region", "cultivar", "roasting"]],
            on="tea_id", how="left"
        )
    else:
        joined = steeps_df.copy()
        for col in ["name", "type", "supplier", "region", "cultivar", "roasting"]:
            if col not in joined.columns:
                joined[col] = None

    joined["session_at"] = ensure_datetime(joined.get("session_at", pd.Series(dtype="datetime64[ns]")))

    left, right = st.columns([1, 1])
    with left:
        tea_type_filter = st.selectbox("Tea type", options=["(all)"] + TEA_TYPES, index=0, key="analysis_type")
    with right:
        # Supplier dropdown from distinct values
        supplier_opts = sorted(
            teas_df.get("supplier", pd.Series(dtype=str))
                   .dropna().astype(str).str.strip()
                   .replace("", pd.NA).dropna()
                   .unique().tolist()
        )
        supplier_filter = st.selectbox("Supplier", options=["(all)"] + supplier_opts, index=0, key="analysis_supplier")

    scope_mask = pd.Series(True, index=joined.index)
    if tea_type_filter != "(all)":
        scope_mask &= joined["type"].fillna("").str.lower() == tea_type_filter.lower()
    if supplier_filter != "(all)":
        scope_mask &= joined["supplier"].fillna("").str.lower() == supplier_filter.lower()
    scope_df = joined[scope_mask].copy()

    st.markdown("### Box & whisker by tea")
    plot_box_by_tea(scope_df)

    with st.expander("View data used in chart"):
        show_cols = ["name", "rating", "supplier", "type", "session_at", "tasting_notes", "steep_notes"]
        present_cols = [c for c in show_cols if c in scope_df.columns]
        st.data_editor(scope_df[present_cols].sort_values("name"), use_container_width=True, disabled=True)
