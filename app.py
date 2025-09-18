# -------------------- New helper: box & whisker across teas --------------------

def plot_box_by_tea(df: pd.DataFrame, title: str = "Tea ratings — box & whisker"):
    """
    Expects a DataFrame with columns: name, rating (0–5 recommended).
    Shows a box-and-whisker plot per tea and overlays individual points.
    """
    if df.empty or "rating" not in df.columns or "name" not in df.columns:
        st.info("No data to chart yet.")
        return

    working = df.copy()
    # Coerce and drop rows without numeric ratings
    working["rating"] = pd.to_numeric(working["rating"], errors="coerce")
    working = working.dropna(subset=["rating"])
    if working.empty:
        st.info("No ratings to chart yet.")
        return

    # Order teas by median rating (desc) for a more useful X axis
    medians = working.groupby("name")["rating"].median().sort_values(ascending=False)
    category_order = medians.index.tolist()

    fig = px.box(
        working,
        x="name",
        y="rating",
        points="all",  # show individual sessions as points
        hover_data=["supplier", "type", "session_at", "tasting_notes", "steep_notes"],
        title=title,
        category_orders={"name": category_order},
    )
    fig.update_layout(
        margin=dict(l=12, r=12, t=48, b=12),
        xaxis_title="Tea",
        yaxis_title="Rating",
        hovermode="closest"
    )

    # If you always score 0–5, lock the axis; otherwise comment this out.
    fig.update_yaxes(range=[0, 5])

    st.plotly_chart(fig, use_container_width=True)


# ---------- Tab 3: Browse (re-hauled) ----------
with tabs[2]:
    st.subheader("🔎 Browse & Compare")

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

    # ---- Controls
    left, right = st.columns([2, 1])

    with left:
        tea_names = (
            teas_df.get("name", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        tea_names_sorted = sorted(tea_names)
        selected_tea = st.selectbox("Find a tea", options=["(all teas)"] + tea_names_sorted, index=0)

    with right:
        # Optional quick filters (kept minimal). You can remove this block if you want *only* name search.
        with st.expander("Optional filters"):
            tea_type_filter = st.selectbox("Tea type", options=["(all)"] + TEA_TYPES, index=0)
            supplier_filter = st.text_input("Supplier contains", value="")

    # ---- Apply filters
    mask = pd.Series(True, index=joined.index)

    if selected_tea != "(all teas)":
        mask &= (joined["name"].fillna("") == selected_tea)

    if tea_type_filter != "(all)":
        mask &= (joined["type"].fillna("").str.lower() == tea_type_filter.lower())

    if supplier_filter:
        mask &= joined["supplier"].fillna("").str.contains(supplier_filter, case=False, na=False)

    filtered = joined[mask].copy()

    # ---- 1) Box & whisker across teas (based on current filter scope)
    st.markdown("### Box & whisker by tea")
    plot_box_by_tea(
        # Use the *scope* of teas after optional filters, but don't force a single-tea view for the chart:
        joined[(joined["type"].fillna("").str.lower() == tea_type_filter.lower()) if tea_type_filter != "(all)" else slice(None)]
        if selected_tea == "(all teas)" else
        joined  # When a tea is picked, still show global context by default; change to 'filtered' to show just that tea.
    )

    # ---- 2) Steeping notes for the selected tea
    st.markdown("### Steeping notes")
    if selected_tea == "(all teas)":
        st.info("Select a tea above to see all of its steeping notes.")
    else:
        tea_rows = joined[joined["name"] == selected_tea].copy()
        if tea_rows.empty:
            st.warning("No sessions found for this tea yet.")
        else:
            # Friendly columns & ordering
            cols = [
                "session_at", "rating",
                "tasting_notes", "steep_notes",
                "initial_steep_time_sec", "steep_time_changes",
                "temperature_c", "amount_used_g",
                "type", "supplier", "region", "cultivar", "roasting",
            ]
            present_cols = [c for c in cols if c in tea_rows.columns]
            tea_rows = tea_rows[present_cols].sort_values("session_at", ascending=False)

            # Tidy labels
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

            st.dataframe(tea_rows, use_container_width=True)
