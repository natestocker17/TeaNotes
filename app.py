# =========================
# Tea Notes (Steeps) — full app with early raw JSON/CSV endpoint
# =========================

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from streamlit import column_config  # optional, for nicer column configs

# -------------------- Supabase availability --------------------
try:
    from supabase import create_client, Client  # type: ignore
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False


# -------------------- Helpers: query params + DB --------------------
def _get_query_params() -> Dict[str, str]:
    """Return query params as a flat dict[str, str], working on old/new Streamlit."""
    try:
        # New API (1.33+)
        qp = st.query_params
        return {k: v for k, v in qp.items()}
    except Exception:
        # Older API
        qp = st.experimental_get_query_params()
        return {k: (v[0] if isinstance(v, list) and v else v) for k, v in qp.items()}


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
    """Load teas and steeps from Supabase and normalize Buy_again/buy_again in-memory."""
    if SUPABASE is None:
        return {"teas": pd.DataFrame(), "steeps": pd.DataFrame()}
    teas = pd.DataFrame(SUPABASE.table("teas").select("*").execute().data)  # type: ignore
    steeps = pd.DataFrame(SUPABASE.table("steeps").select("*").execute().data)  # type: ignore

    # Normalize Buy_again casing/alias for compatibility
    if "buy_again" not in teas.columns and "Buy_again" in teas.columns:
        teas["buy_again"] = teas["Buy_again"]
    if "Buy_again" not in teas.columns and "buy_again" in teas.columns:
        teas["Buy_again"] = teas["buy_again"]

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
    """Return a valid index for Streamlit selectbox; fall back to default if missing."""
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
    """Convert pandas/NumPy NaN/NaT to None and NumPy scalars to Python scalars for JSON."""
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
    """Update rows by primary key; returns a list of error messages (empty if success)."""
    errs = []
    if SUPABASE is None:
        return ["Database is not configured."]
    for r in rows:
        pk_val = r.get(pk_col)
        if pk_val is None or (isinstance(pk_val, float) and np.isnan(pk_val)):
            errs.append(f"Missing {pk_col} in row; skipped.")
            continue
        payload = {k: _json_sanitize(v) for k, v in r.items() if k != pk_col}
        try:
            SUPABASE.table(table).update(payload).eq(pk_col, pk_val).execute()  # type: ignore
        except Exception as e:
            errs.append(f"{table} update failed for {pk_col}={pk_val}: {e}")
    return errs


def diff_rows(original: pd.DataFrame, edited: pd.DataFrame, pk_col: str, editable_cols: List[str]) -> pd.DataFrame:
    """Return only rows from 'edited' that changed compared to 'original' for the given cols."""
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
    """Coerce edited values to JSON-safe, DB-correct types."""
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


# -------------------- Supabase + data (for raw and UI) --------------------
SUPABASE = get_supabase()
_db = load_data()
teas_df = _db["teas"].copy()
steeps_df = _db["steeps"].copy()


# -------------------- EARLY RAW JSON/CSV ENDPOINT (must run before any UI) --------------------
params = _get_query_params()
if params.get("raw") is not None:
    mode = (params.get("raw") or "steeps").lower()

    # Build join similar to view
    if "tea_id" in steeps_df.columns and ("tea_id" in teas_df.columns or "id" in teas_df.columns):
        teas_key = "tea_id" if "tea_id" in teas_df.columns else "id"
        joined = steeps_df.merge(
            teas_df.rename(columns={teas_key: "tea_id"})[
                [
                    "tea_id",
                    "name",
                    "type",
                    "supplier",
                    "region",
                    "cultivar",
                    "roasting",
                    "buy_again",
                ]
            ],
            on="tea_id",
            how="left",
        )
    else:
        joined = steeps_df.copy()

    # Optional filters
    tea_name_param = params.get("tea_name")
    tea_id_param = params.get("tea_id")
    type_param = params.get("type")
    q_param = params.get("q")  # search tasting/steep notes
    start_param = params.get("from")
    end_param = params.get("to")
    limit_param = params.get("limit")
    order_param = (params.get("order") or "desc").lower()

    if mode in ("steeps", "steeps_with_tea"):
        df_out = joined
    elif mode == "teas":
        df_out = teas_df
    else:
        st.write({"error": "Unknown ?raw parameter. Use ?raw=steeps or ?raw=teas."})
        st.stop()

    # Coerce session_at for filtering/sorting
    if "session_at" in df_out.columns:
        df_out["session_at"] = pd.to_datetime(df_out["session_at"], errors="coerce", utc=True)

    # Apply filters (case-insensitive where sensible)
    if tea_name_param and "name" in df_out.columns:
        df_out = df_out[df_out["name"].astype(str).str.contains(tea_name_param, case=False, na=False)]
    if tea_id_param and "tea_id" in df_out.columns:
        try:
            df_out = df_out[df_out["tea_id"].astype(str) == str(tea_id_param)]
        except Exception:
            pass
    if type_param and "type" in df_out.columns:
        df_out = df_out[df_out["type"].astype(str).str.lower() == str(type_param).lower()]
    if q_param:
        # search across text cols
        for c in ["tasting_notes", "steep_notes"]:
            if c not in df_out.columns:
                df_out[c] = None
        mask = (
            df_out["tasting_notes"].astype(str).str.contains(q_param, case=False, na=False)
            | df_out["steep_notes"].astype(str).str.contains(q_param, case=False, na=False)
        )
        df_out = df_out[mask]
    if start_param and "session_at" in df_out.columns:
        try:
            start_ts = pd.to_datetime(start_param, utc=True)
            df_out = df_out[df_out["session_at"] >= start_ts]
        except Exception:
            pass
    if end_param and "session_at" in df_out.columns:
        try:
            end_ts = pd.to_datetime(end_param, utc=True)
            df_out = df_out[df_out["session_at"] < end_ts]
        except Exception:
            pass

    # Sort newest first if session_at exists
    if "session_at" in df_out.columns:
        df_out = df_out.sort_values("session_at", ascending=(order_param == "asc"))

    # Apply limit
    if limit_param:
        try:
            n = int(limit_param)
            if n > 0:
                df_out = df_out.head(n)
        except Exception:
            pass

    # Output format
    fmt = (params.get("fmt") or "json").lower()
    if fmt == "csv":
        st.write(df_out.to_csv(index=False))
    elif fmt in ("ndjson", "jsonl"):
        # One JSON object per line
        recs = json.loads(df_out.to_json(orient="records"))
        st.write("\n".join(json.dumps(r, ensure_ascii=False) for r in recs))
    else:
        st.write(json.loads(df_out.to_json(orient="records")))
    st.stop()


# ========================= UI SECTION =========================

# -------------------- Page config --------------------
st.set_page_config(page_title="Tea Notes (Steeps)", page_icon="🍵", layout="wide")

# -------------------- CSS --------------------
st.markdown(
    """
<style>
/* Radio looks like tabs */
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

/* Wrap text in grid cells (dataframe / data editor) */
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
""",
    unsafe_allow_html=True,
)

st.title("🍵 Tea Notes — Sessions & Scores")

# -------------------- Config for UI --------------------
TEA_TYPES = ["Oolong", "Black", "White", "Green", "Pu-erh", "Dark", "Yellow"]
ROASTING_OPTIONS = ["Unroasted", "Roasted", "Light", "Medium", "Heavy"]
BUY_AGAIN_OPTIONS = ["Unstated", "Maybe", "No", "Yes", "Definitely"]


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
        title=title,
    )
    fig.update_traces(mode="lines+markers", hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Rating %{y}<extra></extra>")
    fig.add_hline(y=avg, line_dash="dash", annotation_text=f"Average: {avg:.1f}", annotation_position="top left", opacity=0.7)
    fig.update_layout(
        margin=dict(l=12, r=12, t=48, b=12),
        legend=dict(title="Tea", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Session time")
    fig.update_yaxes(title_text="Rating (0–100)")
    st.plotly_chart(fig, use_container_width=True)


def plot_box_by_tea(df, title: str = "Tea ratings — box & whisker"):
    """Box & whisker: Y=rating, X=tea name, show individual points."""
    if df is None or len(df) == 0 or "name" not in df.columns or "rating" not in df.columns:
        st.info("No ratings to chart yet.")
        return
    working = df.copy()
    working["rating"] = pd.to_numeric(working["rating"], errors="coerce")
    for col in ["supplier", "type", "session_at", "tasting_notes", "steep_notes"]:
        if col not in working.columns:
            working[col] = None
    working = working.dropna(subset=["rating"]) if not working.empty else working
    if working.empty:
        st.info("No ratings to chart yet.")
        return
    medians = working.groupby("name")["rating"].median().sort_values(ascending=False)
    fig = px.box(
        working,
        x="name",
        y="rating",
        points="all",
        hover_data=["supplier", "type", "session_at", "tasting_notes", "steep_notes"],
        title=title,
        category_orders={"name": medians.index.tolist()},
    )
    fig.update_layout(
        margin=dict(l=12, r=12, t=48, b=12),
        xaxis_title="Tea",
        yaxis_title="Rating",
        hovermode="closest",
    )
    fig.update_yaxes(range=[0, 5])
    st.plotly_chart(fig, use_container_width=True)


# -------------------- Nav --------------------
NAV_ITEMS = ["📝 Add Session", "➕ Add Tea", "✏️ Edit tea", "📜 Steep history", "📊 Analysis"]
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
    # ---------- Add Session ----------
    tea_choices = ["(select)"] + teas_df.get("name", pd.Series(dtype=str)).fillna("(unnamed)").tolist()
    tea_selected = st.selectbox("Tea", tea_choices, index=0, key="add_sess_tea")
    tea_selected_row = None
    if tea_selected != "(select)" and "name" in teas_df.columns:
        tea_selected_row = teas_df[teas_df["name"] == tea_selected].head(1)
    tea_id = None
    if tea_selected_row is not None and not tea_selected_row.empty:
        tea_id = tea_selected_row.iloc[0].get("tea_id") or tea_selected_row.iloc[0].get("id")

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
                "session_at": datetime.utcnow().isoformat(),
            }
            try:
                SUPABASE.table("steeps").insert(row).execute()  # type: ignore
                st.success("Saved.")
            except Exception as e:
                st.error(f"Failed to save: {e}")

elif st.session_state.active_tab == "➕ Add Tea":
    # ---------- Add Tea ----------
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
        processing_notes = st.text_area("Processing notes", key="add_tea_processing_notes")
    with colB:
        cultivar_sel = st.selectbox("Cultivar", options=[""] + cultivar_opts, index=0, key="add_tea_cultivar_sel")
        cultivar_new = st.text_input("Or add new Cultivar", key="add_tea_cultivar_new")
        region_sel = st.selectbox("Region", options=[""] + region_opts, index=0, key="add_tea_region_sel")
        region_new = st.text_input("Or add new Region", key="add_tea_region_new")
        pick_year_txt = st.text_input("Pick year", value="", key="add_tea_pick_year")
        oxidation = st.text_input("Oxidation", key="add_tea_oxidation")
        roasting = st.selectbox("Roasting", options=[""] + ROASTING_OPTIONS, index=0, key="add_tea_roasting")
        elevation_m_txt = st.text_input("Elevation (m)", value="", key="add_tea_elevation_m")
        picking_season = st.text_input("Picking season", key="add_tea_picking_season")
        # Buy_again dropdown (text)
        buy_again_sel = st.selectbox("Buy again", options=BUY_AGAIN_OPTIONS, index=0, key="add_tea_buy_again")

    # Resolve chosen vs new values (all optional)
    subtype = (subtype_new.strip() or subtype_sel.strip() or None)
    supplier = (supplier_new.strip() or supplier_sel.strip() or None)
    cultivar = (cultivar_new.strip() or cultivar_sel.strip() or None)
    region = (region_new.strip() or region_sel.strip() or None)
    pick_year = safe_int(pick_year_txt) if pick_year_txt else None
    roasting_val = roasting.strip() or None
    tea_type_val = tea_type.strip() or None

    # Extra fields (as text)
    elevation_m = (elevation_m_txt.strip() or None)
    processing_notes_val = (processing_notes.strip() or None) if isinstance(processing_notes, str) else None
    picking_season_val = (picking_season.strip() or None) if isinstance(picking_season, str) else None
    buy_again_val = None if buy_again_sel == "Unstated" else buy_again_sel

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
                "processing_notes": processing_notes_val,
                "elevation_m": elevation_m,  # TEXT
                "picking_season": picking_season_val,
                # Write both keys for compatibility with existing schema
                "buy_again": buy_again_val,
                "Buy_again": buy_again_val,
                "created_at": datetime.utcnow().isoformat(),
            }
            try:
                SUPABASE.table("teas").insert(tea_row).execute()  # type: ignore
                st.success("Saved.")
            except Exception as e:
                st.error(f"Failed to save: {e}")

elif st.session_state.active_tab == "✏️ Edit tea":
    # ---------- Edit Tea Details ----------
    st.subheader("✏️ Edit tea")

    tea_pk = get_pk_column(teas_df, ["tea_id", "id"])
    if tea_pk is None:
        st.warning("Could not determine the tea primary key column (expected 'tea_id' or 'id').")
    else:
        tea_names = (
            teas_df.get("name", pd.Series(dtype=str)).dropna().astype(str).str.strip()
        )
        tea_names = tea_names[tea_names != ""].unique().tolist()
        tea_names_sorted = sorted(tea_names)

        selected_name = st.selectbox("Choose a tea to edit", options=["(select)"] + tea_names_sorted, index=0, key="edit_tea_select")
        if selected_name == "(select)":
            st.info("Select a tea above to edit its details.")
        else:
            row = teas_df[teas_df["name"] == selected_name].head(1)
            if row.empty:
                st.warning("Tea not found.")
            else:
                tea_pk_val = row.iloc[0][tea_pk]

                type_options = [""] + TEA_TYPES
                roast_options = [""] + ROASTING_OPTIONS
                type_idx = safe_index(type_options, row.iloc[0].get("type", ""))
                roast_idx = safe_index(roast_options, row.iloc[0].get("roasting", ""))

                # Normalize current buy_again (accept either column)
                current_buy_again = row.iloc[0].get("buy_again", None)
                if current_buy_again is None:
                    current_buy_again = row.iloc[0].get("Buy_again", None)
                buy_again_idx = safe_index(BUY_AGAIN_OPTIONS, current_buy_again, default=0)

                colA, colB = st.columns(2)
                with colA:
                    name_new = st.text_input("Tea name", value=str(row.iloc[0].get("name", "") or ""))
                    type_new = st.selectbox("Tea type", options=type_options, index=type_idx, key=f"edit_tea_type_{tea_pk_val}")
                    subtype_new = st.text_input("Subtype", value=str(row.iloc[0].get("subtype", "") or ""))
                    supplier_new = st.text_input("Supplier", value=str(row.iloc[0].get("supplier", "") or ""))
                    url_new = st.text_input("URL", value=str(row.iloc[0].get("URL", "") or ""))
                    processing_notes_new = st.text_area(
                        "Processing notes",
                        value=str(row.iloc[0].get("processing_notes", "") or ""),
                        key=f"edit_tea_processing_notes_{tea_pk_val}",
                    )
                with colB:
                    cultivar_new = st.text_input("Cultivar", value=str(row.iloc[0].get("cultivar", "") or ""))
                    region_new = st.text_input("Region", value=str(row.iloc[0].get("region", "") or ""))
                    pick_year_new = st.text_input("Pick year", value=str(row.iloc[0].get("pick_year", "") or ""))
                    oxidation_new = st.text_input("Oxidation", value=str(row.iloc[0].get("oxidation", "") or ""))
                    roasting_new = st.selectbox("Roasting", options=roast_options, index=roast_idx, key=f"edit_tea_roasting_{tea_pk_val}")
                    elevation_m_new = st.text_input(
                        "Elevation (m)",
                        value=str(row.iloc[0].get("elevation_m", "") or ""),
                        key=f"edit_tea_elevation_m_{tea_pk_val}",
                    )
                    picking_season_new = st.text_input(
                        "Picking season",
                        value=str(row.iloc[0].get("picking_season", "") or ""),
                        key=f"edit_tea_picking_season_{tea_pk_val}",
                    )
                    # Buy_again dropdown (key depends on tea id to avoid sticky state)
                    buy_again_new_sel = st.selectbox(
                        "Buy again",
                        options=BUY_AGAIN_OPTIONS,
                        index=buy_again_idx,
                        key=f"edit_tea_buy_again_{tea_pk_val}",
                    )

                save_btn = st.button("Save changes", type="primary", key=f"edit_tea_save_{tea_pk_val}")
                if save_btn:
                    if SUPABASE is None:
                        st.error("Database is not configured.")
                    else:
                        buy_again_to_save = None if buy_again_new_sel == "Unstated" else buy_again_new_sel
                        payload = {
                            "name": name_new.strip() or None,
                            "type": (type_new.strip() or None),
                            "subtype": (subtype_new.strip() or None),
                            "supplier": (supplier_new.strip() or None),
                            "URL": (url_new.strip() or None),
                            "cultivar": (cultivar_new.strip() or None),
                            "region": (region_new.strip() or None),
                            "pick_year": safe_int(pick_year_new),
                            "oxidation": (oxidation_new.strip() or None),
                            "roasting": (roasting_new.strip() or None),
                            "processing_notes": (processing_notes_new.strip() or None)
                            if isinstance(processing_notes_new, str)
                            else None,
                            "elevation_m": (elevation_m_new.strip() or None),
                            "picking_season": (picking_season_new.strip() or None)
                            if isinstance(picking_season_new, str)
                            else None,
                            # Write both keys for compatibility
                            "buy_again": buy_again_to_save,
                            "Buy_again": buy_again_to_save,
                        }
                        payload = {k: _json_sanitize(v) for k, v in payload.items()}
                        try:
                            SUPABASE.table("teas").update(payload).eq(tea_pk, tea_pk_val).execute()  # type: ignore
                            st.success("Tea updated.")
                            st.cache_data.clear()
                        except Exception as e:
                            st.error(f"Failed to update: {e}")

elif st.session_state.active_tab == "📜 Steep history":
    # ---------- Steep history (editable) ----------
    st.subheader("📜 Steep history")

    # Join steeps to teas to show tea meta
    if "tea_id" in steeps_df.columns and ("tea_id" in teas_df.columns or "id" in teas_df.columns):
        teas_key = "tea_id" if "tea_id" in teas_df.columns else "id"
        joined = steeps_df.merge(
            teas_df.rename(columns={teas_key: "tea_id"})[
                ["tea_id", "name", "type", "supplier", "region", "cultivar", "roasting"]
            ],
            on="tea_id",
            how="left",
        )
    else:
        joined = steeps_df.copy()
        for col in ["name", "type", "supplier", "region", "cultivar", "roasting"]:
            if col not in joined.columns:
                joined[col] = None

    joined["session_at"] = ensure_datetime(joined.get("session_at", pd.Series(dtype="datetime64[ns]")))
    joined = joined.sort_values("session_at", ascending=False)

    # ---- Recent steeps (reactive, no buttons) ----
    st.markdown("### Recent steeps")
    recent_left, recent_right = st.columns([1, 1])
    with recent_left:
        recent_n = st.selectbox("Show last N", options=[10, 20, 50, 100], index=1, key="recent_steeps_n")
    with recent_right:
        tea_names_for_sel = (
            teas_df.get("name", pd.Series(dtype=str)).dropna().astype(str).str.strip()
        )
    tea_names = tea_names_for_sel[tea_names_for_sel != ""].unique().tolist() if not teas_df.empty else []
    tea_names_sorted = sorted(tea_names)

    # Put the tea selector after "Recent steeps" controls
    selected_tea = st.selectbox(
        "Find a tea",
        options=["(select a tea)"] + tea_names_sorted,
        index=0,
        key="hist_select_tea",
    )

    only_selected = False
    if selected_tea != "(select a tea)":
        only_selected = st.checkbox(
            "Only show selected tea in recent steeps", value=False, key=f"recent_only_{selected_tea}"
        )

    recent_df = joined.copy()
    if only_selected and selected_tea != "(select a tea)":
        recent_df = recent_df[recent_df["name"] == selected_tea]
    recent_cols = [
        c
        for c in [
            "session_at",
            "name",
            "rating",
            "tasting_notes",
            "steep_notes",
            "initial_steep_time_sec",
            "temperature_c",
            "amount_used_g",
            "supplier",
            "type",
        ]
        if c in recent_df.columns
    ]
    st.dataframe(
        recent_df.sort_values("session_at", ascending=False)[recent_cols].head(recent_n),
        use_container_width=True,
    )

    # ---- Detailed table for selected tea (editable) ----
    if selected_tea == "(select a tea)":
        st.info("Select a tea above to view and edit its steeps.")
    else:
        rows = joined[joined["name"] == selected_tea].copy()
        if rows.empty:
            st.warning("No sessions found for this tea yet.")
        else:
            steep_pk = get_pk_column(rows, ["steep_id", "id"])
            if steep_pk is None:
                st.warning("Could not determine the steep primary key column (expected 'steep_id' or 'id'). Editing is disabled.")
                st.dataframe(rows.sort_values("session_at", ascending=False), use_container_width=True)
            else:
                display_cols = [
                    steep_pk,
                    "session_at",
                    "rating",
                    "tasting_notes",
                    "steep_notes",
                    "initial_steep_time_sec",
                    "steep_time_changes",
                    "temperature_c",
                    "amount_used_g",
                    # meta (read-only)
                    "type",
                    "supplier",
                    "region",
                    "cultivar",
                    "roasting",
                ]
                present_cols = [c for c in display_cols if c in rows.columns]
                rows = rows[present_cols].copy()

                # Remember original for diffing on this tea
                st.session_state["orig_steeps_df"] = rows.copy()

                readonly_cols = ["type", "supplier", "region", "cultivar", "roasting"]
                col_conf = {}
                for c in present_cols:
                    if c in readonly_cols:
                        col_conf[c] = st.column_config.Column(c, disabled=True)
                    elif c in ["tasting_notes", "steep_notes"]:
                        col_conf[c] = st.column_config.TextColumn(c)  # CSS handles wrapping
                    elif c in ["rating", "amount_used_g"]:
                        col_conf[c] = st.column_config.NumberColumn(c, step=0.1, format="%.2f")
                    elif c in ["initial_steep_time_sec", "temperature_c"]:
                        col_conf[c] = st.column_config.NumberColumn(c, step=1, format="%d")
                    elif c == "session_at":
                        col_conf[c] = st.column_config.Column(c)
                    elif c == steep_pk:
                        col_conf[c] = st.column_config.Column(c, disabled=True)

                edited = st.data_editor(
                    rows,
                    key="steep_editor",
                    use_container_width=True,
                    hide_index=True,
                    column_config=col_conf,
                )

                # Save button
                if st.button("Save changes", type="primary", key="save_steeps_btn"):
                    if SUPABASE is None:
                        st.error("Database is not configured.")
                    else:
                        orig = st.session_state.get("orig_steeps_df", rows)
                        editable_cols = [
                            "session_at",
                            "rating",
                            "tasting_notes",
                            "steep_notes",
                            "initial_steep_time_sec",
                            "steep_time_changes",
                            "temperature_c",
                            "amount_used_g",
                        ]
                        changed = diff_rows(orig, edited, steep_pk, editable_cols)
                        if changed.empty:
                            st.info("No changes to save.")
                        else:
                            payloads = build_steep_payloads(changed, steep_pk)
                            errors = update_supabase_rows("steeps", steep_pk, payloads)
                            if errors:
                                for e in errors:
                                    st.error(e)
                            else:
                                st.success(f"Saved {len(payloads)} change(s).")
                                st.cache_data.clear()  # refresh cached data

elif st.session_state.active_tab == "📊 Analysis":
    # ---------- Analysis ----------
    st.subheader("📊 Analysis")

    # Join for charting
    if "tea_id" in steeps_df.columns and ("tea_id" in teas_df.columns or "id" in teas_df.columns):
        teas_key = "tea_id" if "tea_id" in teas_df.columns else "id"
        joined = steeps_df.merge(
            teas_df.rename(columns={teas_key: "tea_id"})[
                ["tea_id", "name", "type", "supplier", "region", "cultivar", "roasting"]
            ],
            on="tea_id",
            how="left",
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
        supplier_opts = sorted(
            teas_df.get("supplier", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
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
