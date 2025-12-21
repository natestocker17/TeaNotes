# =========================
# Tea Notes (Steeps) — UI updated for new Supabase schema
# =========================

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit import column_config  # noqa: F401

# -------------------- Supabase availability --------------------
try:
    from supabase import create_client, Client  # type: ignore
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False


@st.cache_resource
def get_supabase() -> Optional["Client"]:
    if not SUPABASE_AVAILABLE:
        return None
    url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = (
        st.secrets.get("SUPABASE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )
    if not url or not key:
        return None
    try:
        return create_client(url, key)  # type: ignore
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_data() -> Dict[str, pd.DataFrame]:
    """Load teas and steeps from Supabase."""
    if SUPABASE is None:
        return {"teas": pd.DataFrame(), "steeps": pd.DataFrame()}

    teas = pd.DataFrame(SUPABASE.table("teas").select("*").execute().data)  # type: ignore
    steeps = pd.DataFrame(SUPABASE.table("steeps").select("*").execute().data)  # type: ignore
    return {"teas": teas, "steeps": steeps}


def ensure_datetime(col: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(col, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None] * len(col)))


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


def safe_index(options: List[str], value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        v = str(value).strip()
    except Exception:
        return default
    return options.index(v) if v in options else default


def get_pk_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _json_sanitize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        try:
            return pd.to_datetime(value, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return str(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def update_supabase_rows(table: str, pk_col: str, rows: List[Dict[str, Any]]) -> List[str]:
    """Update rows by primary key; only send columns that exist in remote table."""
    errs = []
    if SUPABASE is None:
        return ["Database is not configured."]
    existing_cols = set(teas_df.columns) if table == "teas" else set(steeps_df.columns)
    for r in rows:
        pk_val = r.get(pk_col)
        if pk_val is None or (isinstance(pk_val, float) and np.isnan(pk_val)):
            errs.append(f"Missing {pk_col} in row; skipped.")
            continue
        payload = {k: _json_sanitize(v) for k, v in r.items() if k != pk_col and k in existing_cols}
        try:
            SUPABASE.table(table).update(payload).eq(pk_col, pk_val).execute()  # type: ignore
        except Exception as e:
            errs.append(f"{table} update failed for {pk_col}={pk_val}: {e}")
    return errs


def diff_rows(original: pd.DataFrame, edited: pd.DataFrame, pk_col: str, editable_cols: List[str]) -> pd.DataFrame:
    """Return only the edited rows that changed in editable columns."""
    if original.empty or edited.empty:
        return edited.iloc[0:0].copy()
    left = original.set_index(pk_col)
    right = edited.set_index(pk_col)
    cols = [c for c in editable_cols if c in left.columns and c in right.columns]
    if not cols:
        return edited.iloc[0:0].copy()
    l = left[cols].applymap(lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    r = right[cols].applymap(lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    changed_mask = (l != r).any(axis=1)
    changed_idx = changed_mask[changed_mask].index
    return edited[edited[pk_col].isin(changed_idx)].copy()


# --- types for steeps & payload builder ---
INT_COLS_STEEP = ["initial_steep_time_sec", "temperature_c"]
FLOAT_COLS_STEEP = ["rating", "amount_used_g"]


def _to_iso_utc_or_none(value):
    if value is None or (isinstance(value, str) and value.strip() == "") or (isinstance(value, float) and np.isnan(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_steep_payloads(changed_df: pd.DataFrame, pk_col: str) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for _, r in changed_df.iterrows():
        rec: Dict[str, Any] = {pk_col: r[pk_col]}
        cols = [
            "session_at",
            "rating",
            "tasting_notes",
            "steep_notes",
            "initial_steep_time_sec",
            "steep_time_changes",
            "temperature_c",
            "amount_used_g",
        ]
        for c in cols:
            if c not in changed_df.columns:
                continue
            v = r[c]
            if c == "session_at":
                val = _to_iso_utc_or_none(v)
            elif c in INT_COLS_STEEP:
                if v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, float) and np.isnan(v)):
                    val = None
                else:
                    try:
                        val = int(round(float(v)))
                    except Exception:
                        val = None
            elif c in FLOAT_COLS_STEEP:
                if v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, float) and np.isnan(v)):
                    val = None
                else:
                    try:
                        val = float(v)
                    except Exception:
                        val = None
            else:
                if v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, float) and np.isnan(v)):
                    val = None
                else:
                    val = v
            rec[c] = val
        payloads.append(rec)
    return payloads


# -------------------- Supabase + data --------------------
SUPABASE = get_supabase()
_db = load_data()
teas_df = _db["teas"].copy()
steeps_df = _db["steeps"].copy()

# ========================= UI SECTION =========================
st.set_page_config(page_title="Tea Notes (Steeps)", page_icon="🍵", layout="wide")

st.markdown(
    """
<style>
/* "tabs" look for radio */
[data-testid="stHorizontalBlock"] > div:has(> div[data-testid="stRadio"]) { margin-bottom: 0.5rem; }
div[data-testid="stRadio"] > div[role="radiogroup"] { display:flex; gap:.25rem; flex-wrap:wrap; }
div[data-testid="stRadio"] label {
  border:1px solid var(--secondary-background-color); padding:.4rem .8rem; border-radius:.5rem .5rem 0 0;
  background:var(--secondary-background-color); cursor:pointer; font-weight:500;
}
div[data-testid="stRadio"] label[data-checked="true"] {
  background:var(--background-color); border-bottom-color:var(--background-color);
  box-shadow:0 -2px 0 0 var(--primary-color) inset;
}
/* wrap table text */
[data-testid="stDataFrame"] div[role="gridcell"],
[data-testid="stDataFrame"] div[data-testid="cell-container"],
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] span, [data-testid="stDataFrame"] p {
  white-space:normal !important; overflow:visible !important; text-overflow:clip !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🍵 Tea Notes — Sessions & Scores")

TEA_TYPES = ["Oolong", "Black", "White", "Green", "Pu-erh", "Dark", "Yellow"]
TO_BUY_OPTIONS = ["No", "Maybe", "Yes"]

# -------------------- Nav --------------------
NAV_ITEMS = ["📝 Add Session", "➕ Add Tea", "✏️ Edit tea", "📜 Steep history", "📊 Analysis"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = NAV_ITEMS[0]

st.session_state.active_tab = st.radio(
    "Tabs", NAV_ITEMS, index=NAV_ITEMS.index(st.session_state.active_tab),
    horizontal=True, label_visibility="collapsed", key="nav_radio",
)

# -------------------- Reset helpers --------------------
ADD_SESSION_KEYS = [
    "add_sess_tea",
    "add_sess_initial_secs",
    "add_sess_changes",
    "add_sess_temp",
    "add_sess_amount",
    "add_sess_tnotes",
    "add_sess_snotes",
    "add_sess_rating",
]

def reset_add_session_form():
    for k in ADD_SESSION_KEYS:
        if k in st.session_state:
            del st.session_state[k]


# -------------------- Screens --------------------
if st.session_state.active_tab == "📝 Add Session":
    tea_choices = ["(select)"] + teas_df.get("name", pd.Series(dtype=str)).fillna("(unnamed)").tolist()

    with st.form("add_session_form", clear_on_submit=False):
        tea_selected = st.selectbox("Tea", tea_choices, index=0, key="add_sess_tea")

        initial_secs_txt = st.text_input("Initial steep time (seconds)", value="", key="add_sess_initial_secs")
        changes_text = st.text_input("Steep time changes", value="", key="add_sess_changes")
        temperature_c_txt = st.text_input("Water temperature (°C)", value="", key="add_sess_temp")
        amount_used_g_txt = st.text_input("Tea amount used (g)", value="", key="add_sess_amount")
        tasting_notes = st.text_area("Tasting notes", value="", key="add_sess_tnotes")
        steep_notes = st.text_area("Steep notes", value="", key="add_sess_snotes")
        overall_rating_txt = st.text_input("Overall rating (0–5)", value="", key="add_sess_rating")

        save_session_btn = st.form_submit_button("Save Session", type="primary", use_container_width=True)

    tea_selected_row = None
    if tea_selected != "(select)" and "name" in teas_df.columns:
        tea_selected_row = teas_df[teas_df["name"] == tea_selected].head(1)

    tea_id = None
    if tea_selected_row is not None and not tea_selected_row.empty:
        tea_id = tea_selected_row.iloc[0].get("tea_id") or tea_selected_row.iloc[0].get("id")

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
                "session_at": datetime.utcnow().isoformat(),
            }
            try:
                SUPABASE.table("steeps").insert(row).execute()  # type: ignore
                st.success("Saved.")

                # Make the new steep show up elsewhere + reset inputs
                st.cache_data.clear()
                reset_add_session_form()
                st.rerun()

            except Exception as e:
                st.error(f"Failed to save: {e}")

elif st.session_state.active_tab == "➕ Add Tea":
    colA, colB = st.columns(2)

    subtype_opts = options_from_column(teas_df, "subtype")
    supplier_opts_all = options_from_column(teas_df, "supplier")
    cultivar_opts = options_from_column(teas_df, "cultivar")
    region_opts = options_from_column(teas_df, "region")

    with st.form("add_tea_form", clear_on_submit=False):
        with colA:
            tea_name = st.text_input("Tea name (required)", key="add_tea_name")
            tea_type = st.selectbox("Tea type", options=[""] + TEA_TYPES, index=0, key="add_tea_type")
            subtype_sel = st.selectbox("Subtype", options=[""] + subtype_opts, index=0, key="add_tea_subtype_sel")
            subtype_new = st.text_input("Or add new Subtype", key="add_tea_subtype_new")
            supplier_sel = st.selectbox("Supplier", options=[""] + supplier_opts_all, index=0, key="add_tea_supplier_sel")
            supplier_new = st.text_input("Or add new Supplier", key="add_tea_supplier_new")
            url = st.text_input("URL", key="add_tea_url")
            processing_notes = st.text_area("Processing notes", key="add_tea_processing_notes")
            have_already_cb = st.checkbox("Have already", value=False, key="add_tea_have_already")
            to_buy_sel = st.selectbox("To buy", options=TO_BUY_OPTIONS, index=0, key="add_tea_to_buy")

        with colB:
            cultivar_sel = st.selectbox("Cultivar", options=[""] + cultivar_opts, index=0, key="add_tea_cultivar_sel")
            cultivar_new = st.text_input("Or add new Cultivar", key="add_tea_cultivar_new")
            region_sel = st.selectbox("Region", options=[""] + region_opts, index=0, key="add_tea_region_sel")
            region_new = st.text_input("Or add new Region", key="add_tea_region_new")
            pick_year_txt = st.text_input("Pick year", value="", key="add_tea_pick_year")
            picking_season = st.text_input("Picking season", key="add_tea_picking_season")
            st.markdown("**Pricing / weights (NZD & grams)**")
            price_1 = st.text_input("Price 1 (NZD)", value="", key="add_tea_price1")
            weight_1 = st.text_input("Weight 1 (g)", value="", key="add_tea_weight1")
            price_2 = st.text_input("Price 2 (NZD)", value="", key="add_tea_price2")
            weight_2 = st.text_input("Weight 2 (g)", value="", key="add_tea_weight2")
            weight_per_session = st.text_input("Weight per session (g)", value="", key="add_tea_wps")

        add_tea_btn = st.form_submit_button("Save Tea", type="primary", use_container_width=True)

    subtype = (subtype_new.strip() or subtype_sel.strip() or None)
    supplier = (supplier_new.strip() or supplier_sel.strip() or None)
    cultivar = (cultivar_new.strip() or cultivar_sel.strip() or None)
    region = (region_new.strip() or region_sel.strip() or None)
    pick_year = safe_int(pick_year_txt) if pick_year_txt else None
    tea_type_val = tea_type.strip() or None
    processing_notes_val = (processing_notes.strip() or None) if isinstance(processing_notes, str) else None
    picking_season_val = (picking_season.strip() or None) if isinstance(picking_season, str) else None

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
                "processing_notes": processing_notes_val,
                "picking_season": picking_season_val,
                "have_already": bool(have_already_cb),
                "to_buy": to_buy_sel,  # "No", "Maybe", "Yes"
                "price_1_nzd": safe_float(price_1),
                "weight_1_g": safe_float(weight_1),
                "price_2_nzd": safe_float(price_2),
                "weight_2_g": safe_float(weight_2),
                "weight_per_session_g": safe_float(weight_per_session),
                "created_at": datetime.utcnow().isoformat(),
            }
            allowed = set(teas_df.columns) if not teas_df.empty else set(tea_row.keys())
            tea_row = {k: _json_sanitize(v) for k, v in tea_row.items() if k in allowed}
            try:
                SUPABASE.table("teas").insert(tea_row).execute()  # type: ignore
                st.success("Saved.")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")

elif st.session_state.active_tab == "✏️ Edit tea":
    st.subheader("✏️ Edit tea")

    tea_pk = get_pk_column(teas_df, ["tea_id", "id"])
    if tea_pk is None:
        st.warning("Could not determine the tea primary key column (expected 'tea_id' or 'id').")
    elif teas_df.empty:
        st.info("No teas found.")
    else:
        tea_names = teas_df.get("name", pd.Series(dtype=str)).dropna().astype(str).str.strip()
        tea_names = tea_names[tea_names != ""].unique().tolist()
        tea_names_sorted = sorted(tea_names)

        selected_name = st.selectbox(
            "Choose a tea to edit",
            options=["(select)"] + tea_names_sorted,
            index=0,
            key="edit_tea_select"
        )
        if selected_name == "(select)":
            st.info("Select a tea above to edit its details.")
        else:
            row = teas_df[teas_df["name"] == selected_name].head(1)
            if row.empty:
                st.warning("Tea not found.")
            else:
                tea_pk_val = row.iloc[0][tea_pk]

                type_options = [""] + TEA_TYPES
                type_idx = safe_index(type_options, row.iloc[0].get("type", ""))

                raw_to_buy = row.iloc[0].get("to_buy", "No")
                if isinstance(raw_to_buy, bool):
                    current_to_buy = "Yes" if raw_to_buy else "No"
                else:
                    raw_str = str(raw_to_buy).strip().title()
                    current_to_buy = raw_str if raw_str in TO_BUY_OPTIONS else "No"
                to_buy_idx = safe_index(TO_BUY_OPTIONS, current_to_buy, default=0)

                colA, colB = st.columns(2)
                with colA:
                    name_new = st.text_input("Tea name", value=str(row.iloc[0].get("name", "") or ""), key=f"edit_tea_name_{tea_pk_val}")
                    type_new = st.selectbox("Tea type", options=type_options, index=type_idx, key=f"edit_tea_type_{tea_pk_val}")
                    subtype_new = st.text_input("Subtype", value=str(row.iloc[0].get("subtype", "") or ""), key=f"edit_tea_subtype_{tea_pk_val}")
                    supplier_new = st.text_input("Supplier", value=str(row.iloc[0].get("supplier", "") or ""), key=f"edit_tea_supplier_{tea_pk_val}")
                    url_new = st.text_input("URL", value=str(row.iloc[0].get("URL", "") or ""), key=f"edit_tea_url_{tea_pk_val}")
                    processing_notes_new = st.text_area(
                        "Processing notes",
                        value=str(row.iloc[0].get("processing_notes", "") or ""),
                        key=f"edit_tea_processing_notes_{tea_pk_val}",
                    )
                    have_already_new = st.checkbox(
                        "Have already",
                        value=bool(row.iloc[0].get("have_already", False)),
                        key=f"edit_tea_have_{tea_pk_val}",
                    )
                    to_buy_new = st.selectbox(
                        "To buy",
                        options=TO_BUY_OPTIONS,
                        index=to_buy_idx,
                        key=f"edit_tea_tobuy_{tea_pk_val}",
                    )
                with
