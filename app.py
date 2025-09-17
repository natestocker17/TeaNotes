
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
st.title("🍵 Tea Notes — Steeps, Scores & Sessions")

# -------------------- Config & Data Access --------------------

TEA_TYPES = ["oolong", "black", "white", "green", "pu-erh", "dark", "yellow"]

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
DEMO_MODE = SUPABASE is None

@st.cache_data(ttl=60)
def load_data() -> Dict[str, pd.DataFrame]:
    if DEMO_MODE:
        # Demo mode: load sample CSVs if present
        teas_path = "/mnt/data/teas_rows.csv"
        steeps_path = "/mnt/data/steeps_rows.csv"
        if os.path.exists(teas_path) and os.path.exists(steeps_path):
            teas = pd.read_csv(teas_path)
            steeps = pd.read_csv(steeps_path)
        else:
            teas = pd.DataFrame(columns=[
                "tea_id","name","type","subtype","processing_notes","oxidation","roasting",
                "cultivar","region","elevation_m","picking_season","pick_year","supplier",
                "created_at","URL","url"
            ])
            steeps = pd.DataFrame(columns=[
                "steep_id","tea_id","tasting_notes","rating","steeps","initial_steep_time_sec",
                "steep_time_changes","temperature_c","amount_used_g","session_at","steep_notes","steep_scores_json"
            ])
        # Ensure expected columns
        if "steep_scores_json" not in steeps.columns:
            steeps["steep_scores_json"] = None
        return {"teas": teas, "steeps": steeps}

    # Live mode: fetch from Supabase
    teas = pd.DataFrame(SUPABASE.table("teas").select("*").execute().data)  # type: ignore
    steeps = pd.DataFrame(SUPABASE.table("steeps").select("*").execute().data)  # type: ignore
    # Normalize expected columns
    if "steep_scores_json" not in steeps.columns:
        steeps["steep_scores_json"] = None
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

def compute_average_score(steep_scores: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in steep_scores if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))

def parse_time_changes(raw: Any) -> List[int]:
    """Accepts list/str of comma-separated ints, returns list of ints."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw if pd.notna(x)]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                pass
        return out
    return []

def expand_steep_scores_row(row: pd.Series) -> Optional[pd.DataFrame]:
    """
    Turn one steep-session row into long-form rows: one per steep, with score.
    Column 'steep_scores_json' should contain a JSON list of per-steep scores (0-5 or 0-10, your choice).
    """
    scores_raw = row.get("steep_scores_json")
    if pd.isna(scores_raw) or scores_raw in (None, "", "null"):
        return None
    try:
        if isinstance(scores_raw, str):
            scores = json.loads(scores_raw)
        else:
            scores = scores_raw
        if not isinstance(scores, list):
            return None
        data = []
        for i, s in enumerate(scores, start=1):
            if s is None:
                continue
            data.append({
                "steep_number": i,
                "steep_score": float(s),
                "tea_id": row.get("tea_id"),
                "steep_id": row.get("steep_id"),
                "session_at": row.get("session_at"),
            })
        if not data:
            return None
        return pd.DataFrame(data)
    except Exception:
        return None

def build_long_scores(steeps: pd.DataFrame) -> pd.DataFrame:
    longs = []
    for _, r in steeps.iterrows():
        df = expand_steep_scores_row(r)
        if df is not None:
            longs.append(df)
    if not longs:
        return pd.DataFrame(columns=["steep_number","steep_score","tea_id","steep_id","session_at"])
    out = pd.concat(longs, ignore_index=True)
    out["session_at"] = ensure_datetime(out["session_at"])
    return out

def plot_steeps_with_average(long_df: pd.DataFrame, title: str = "Steep scores"):
    if long_df.empty:
        st.info("No per-steep scores yet. Add a session with steep scores in the first tab.")
        return
    # Average across visible data
    avg = long_df["steep_score"].mean()
    fig = px.line(
        long_df.sort_values(["session_at","steep_number"]),
        x="steep_number",
        y="steep_score",
        color="session_at",
        markers=True,
        title=title
    )
    fig.update_traces(mode="lines+markers", hovertemplate="Steep %{x}<br>Score %{y}<extra></extra>")
    fig.add_hline(y=avg, line_dash="dash", annotation_text=f"Average: {avg:.2f}", annotation_position="top left", opacity=0.7)
    fig.update_layout(
        margin=dict(l=12, r=12, t=48, b=12),
        legend=dict(title="Session", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_xaxes(title_text="Steep #", tickmode="linear", dtick=1)
    fig.update_yaxes(title_text="Score")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Tabs: 1) Add Session (default)  2) Add Tea  3) Browse --------------------

tabs = st.tabs(["📝 Add Session", "➕ Add Tea", "🔎 Browse"])

# ---------- Tab 1: Add Session (default) ----------
with tabs[0]:
    st.caption("Record a tasting session with per-steep scores.")
    # Select Tea
    tea_choices = ["(select)"] + teas_df.get("name", pd.Series(dtype=str)).fillna("(unnamed)").tolist()
    tea_selected = st.selectbox("Tea", tea_choices, index=0)
    tea_selected_row = None
    if tea_selected != "(select)" and "name" in teas_df.columns:
        tea_selected_row = teas_df[teas_df["name"] == tea_selected].head(1)
    tea_id = None
    if tea_selected_row is not None and not tea_selected_row.empty:
        tea_id = tea_selected_row.iloc[0].get("tea_id")

    # --- Steep timing fields (order matters for UX) ---
    initial_secs = st.number_input("Initial steep time (seconds)", min_value=0, step=5, value=15, help="First infusion length.")
    num_steeps = st.number_input("Number of steeps", min_value=1, max_value=15, step=1, value=6)

    with st.expander("Steep time changes (after initial steep)", expanded=False):
        time_changes: List[int] = []
        for i in range(2, int(num_steeps)+1):
            val = st.number_input(f"Δ time for steep {i} (±seconds)", value=5, step=5, key=f"time_change_{i}")
            time_changes.append(int(val))

    temperature_c = st.number_input("Water temperature (°C)", min_value=0, max_value=100, value=95)
    amount_used_g = st.number_input("Tea amount used (g)", min_value=0.0, step=0.5, value=5.0)
    tasting_notes = st.text_area("Tasting notes", placeholder="Aroma, texture, aftertaste, etc.")

    # Dynamic per-steep scores
    st.subheader("Per-steep scores")
    st.caption("Give each steep a quick score (0–5). These appear as markers on the chart; an average line is shown.")
    per_steep_scores: List[Optional[float]] = []
    cols = st.columns(5)
    for i in range(1, int(num_steeps)+1):
        with cols[(i-1) % 5]:
            s = st.number_input(f"Steep {i}", min_value=0.0, max_value=5.0, step=0.1, value=4.0, key=f"score_{i}")
            per_steep_scores.append(float(s))

    # Rating at the end (overall)
    overall_rating = st.slider("Overall rating (0–100)", min_value=0, max_value=100, value=80)

    save_session_btn = st.button("Save Session", type="primary", use_container_width=True)
    if save_session_btn:
        if tea_id is None:
            st.error("Please select a tea first (use the 'Add Tea' tab if needed).")
        else:
            row = {
                "tea_id": tea_id,
                "tasting_notes": tasting_notes or None,
                "rating": float(overall_rating),
                "steeps": int(num_steeps),
                "initial_steep_time_sec": int(initial_secs),
                "steep_time_changes": ",".join(str(x) for x in time_changes) if time_changes else None,
                "temperature_c": int(temperature_c),
                "amount_used_g": float(amount_used_g),
                "session_at": datetime.utcnow().isoformat(),
                "steep_scores_json": json.dumps(per_steep_scores),
            }
            if DEMO_MODE:
                st.success("Demo mode: session captured locally (not persisted).")
                # Update in-memory data for the chart in the Browse tab
                steeps_df.loc[len(steeps_df)] = row
            else:
                try:
                    SUPABASE.table("steeps").insert(row).execute()  # type: ignore
                    st.success("Session saved to Supabase.")
                except Exception as e:
                    st.error(f"Failed to save session: {e}")

# ---------- Tab 2: Add Tea ----------
with tabs[1]:
    st.caption("Add a tea to your catalogue.")
    colA, colB = st.columns(2)
    with colA:
        tea_name = st.text_input("Tea name", placeholder="e.g., Lao Cong Mi Lan Xiang")
        tea_type = st.selectbox("Tea type", options=TEA_TYPES, index=0, help="Choose the main style of the tea.")
        subtype = st.text_input("Subtype (optional)", placeholder="Dan Cong, Rou Gui, etc.")
        supplier = st.text_input("Supplier (optional)")
        url = st.text_input("URL (optional)")
    with colB:
        cultivar = st.text_input("Cultivar (optional)")
        region = st.text_input("Region (optional)")
        pick_year = st.number_input("Pick year (optional)", min_value=1900, max_value=datetime.now().year, step=1)
        oxidation = st.text_input("Oxidation (notes)")
        roasting = st.text_input("Roasting (notes)")

    add_tea_btn = st.button("Save Tea", type="primary", use_container_width=True)
    if add_tea_btn:
        if not tea_name:
            st.error("Tea name is required.")
        else:
            tea_row = {
                "name": tea_name,
                "type": tea_type,
                "subtype": subtype or None,
                "supplier": supplier or None,
                "URL": url or None,
                "cultivar": cultivar or None,
                "region": region or None,
                "pick_year": int(pick_year) if pick_year else None,
                "oxidation": oxidation or None,
                "roasting": roasting or None,
                "created_at": datetime.utcnow().isoformat()
            }
            if DEMO_MODE:
                st.success("Demo mode: tea captured locally (not persisted).")
                # Update in-memory list for the Add Session tab
                teas_df.loc[len(teas_df)] = tea_row
            else:
                try:
                    SUPABASE.table("teas").insert(tea_row).execute()  # type: ignore
                    st.success("Tea saved to Supabase.")
                except Exception as e:
                    st.error(f"Failed to save tea: {e}")

# ---------- Tab 3: Browse (charts + table) ----------
with tabs[2]:
    st.caption("Explore your sessions and teas. Use filters to narrow down, then see per-steep markers and the average line.")
    # Filters
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        tea_type_filter = st.selectbox("Tea type", options=["(all)"] + TEA_TYPES, index=0)
    with col2:
        supplier_filter = st.text_input("Supplier contains", value="")
    with col3:
        date_from = st.date_input("From date", value=None)
    with col4:
        date_to = st.date_input("To date", value=None)

    # Join steeps with teas for filtering/context
    if "tea_id" in steeps_df.columns and "tea_id" in teas_df.columns:
        joined = steeps_df.merge(teas_df[["tea_id","name","type","supplier"]], on="tea_id", how="left")
    else:
        joined = steeps_df.copy()
        joined["name"] = None
        joined["type"] = None
        joined["supplier"] = None

    joined["session_at"] = ensure_datetime(joined.get("session_at", pd.Series(dtype="datetime64[ns]")))

    # Apply filters
    mask = pd.Series([True]*len(joined))
    if tea_type_filter != "(all)":
        mask &= (joined["type"].str.lower() == tea_type_filter)
    if supplier_filter:
        mask &= joined["supplier"].fillna("").str.contains(supplier_filter, case=False, na=False)
    if date_from:
        mask &= (joined["session_at"].dt.date >= date_from)
    if date_to:
        mask &= (joined["session_at"].dt.date <= date_to)

    joined_filt = joined[mask].copy()

    # Build long-form per-steep scores for chart
    long_scores = []
    for _, r in joined_filt.iterrows():
        df = expand_steep_scores_row(r)
        if df is not None:
            long_scores.append(df)
    if long_scores:
        long_scores = pd.concat(long_scores, ignore_index=True)
        long_scores["session_at"] = ensure_datetime(long_scores["session_at"])
    else:
        long_scores = pd.DataFrame(columns=["steep_number","steep_score","session_at"])

    st.markdown("### 📈 Steep Scores (markers per steep + average line)")
    plot_steeps_with_average(long_scores, title="Steep Scores (touch-friendly, responsive)")

    st.markdown("### 📋 Data preview")
    st.dataframe(joined_filt.sort_values("session_at", ascending=False), use_container_width=True)
