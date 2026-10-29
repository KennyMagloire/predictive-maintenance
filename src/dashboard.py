"""
dashboard.py — Design component 4: Maintenance Dashboard

Presents predictions to maintenance engineers: fleet overview, machine detail,
alert list and remaining-useful-life tracking.

Run from the project root with:
    python -m streamlit run src/dashboard.py
"""

import sys
from pathlib import Path

# Streamlit puts this file's folder on the path, not the project root.
# Adding the root lets us write "from src.models import ...".
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.data_pipeline import SENSORS, ENGINEERED, load_dataset, prepare
from src.models import load_classifier, load_rul_model, predict_rul
from src.prediction_engine import assess

DEFAULT_DATA = ROOT / "data" / "ai4i2020.csv"
CMAPSS_DATA = ROOT / "data" / "train_FD001.txt"
STATUS_ICON = {"Critical": "🔴 Critical", "Warning": "🟠 Warning", "Normal": "🟢 Normal"}
CMAPSS_COLUMNS = (["engine", "cycle", "op1", "op2", "op3"]
                  + [f"s{i}" for i in range(1, 22)])

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")


# ---------- Loading (cached so models load once, not on every click) ----------

@st.cache_resource
def get_classifier():
    return load_classifier()


@st.cache_resource
def get_rul_model():
    return load_rul_model()


@st.cache_data
def read_default():
    return load_dataset(DEFAULT_DATA)


@st.cache_data
def read_cmapss():
    return pd.read_csv(CMAPSS_DATA, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)


# ---------- Scoring ----------

def build_table(df):
    """Clean, validate, engineer features, score. Returns (table, rejected, problems)."""
    model, features, thresholds = get_classifier()
    X, rejected, problems = prepare(df, features)
    if X is None:
        return None, rejected, problems
    results = assess(model, X, thresholds)
    table = df.loc[X.index, ["Product ID", "Type"]]
    table = table.join(X[SENSORS + ENGINEERED]).join(results)
    return table, rejected, problems


def for_display(frame):
    """Replace plain status text with a coloured icon version for readability."""
    shown = frame.copy()
    shown["Status"] = shown["Status"].map(STATUS_ICON)
    return shown


# ---------- Page ----------

st.title("🛠️ Predictive Maintenance Dashboard")
st.caption("Failure prediction and remaining useful life for industrial equipment")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload a CSV of machine readings", type="csv")
    st.caption("Without an upload, the AI4I 2020 dataset is used. "
               "Uploaded rows with missing or impossible values are rejected "
               "and reported, not scored.")

df = load_dataset(uploaded) if uploaded else read_default()

required = SENSORS + ["Type"]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    st.error(f"The file is missing required column(s): {', '.join(missing_cols)}")
    st.stop()

if "Product ID" not in df.columns:
    df = df.copy()
    df["Product ID"] = "Row " + df.index.astype(str)

table, rejected, problems = build_table(df)
if table is None:
    st.error("No usable records: every row failed validation.")
    st.write(problems)
    st.stop()

tab_overview, tab_detail, tab_alerts, tab_rul = st.tabs(
    ["Overview", "Machine detail", "Alerts", "Remaining useful life"])

# ----- Overview -----
with tab_overview:
    counts = table["Status"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Critical", int(counts.get("Critical", 0)))
    c2.metric("🟠 Warning", int(counts.get("Warning", 0)))
    c3.metric("🟢 Normal", int(counts.get("Normal", 0)))
    c4.metric("⛔ Rejected (invalid data)", len(rejected))
    loaded, scored = len(df), len(table)
    removed = loaded - scored - len(rejected)
    st.caption(f"Records loaded: {loaded:,} · scored: {scored:,} · "
               f"rejected: {len(rejected):,} · duplicates removed: {removed:,}")

    for message in problems:
        st.warning(message)
    if len(rejected) > 0:
        with st.expander("Show rejected records"):
            st.dataframe(rejected)

    st.subheader("Machines needing attention")
    attention = (table[table["Status"] != "Normal"]
                 .sort_values("Failure probability", ascending=False))
    st.dataframe(for_display(attention.head(10)))

    st.subheader("Fleet status")
    st.bar_chart(counts.reindex(["Critical", "Warning", "Normal"]).fillna(0))

# ----- Machine detail -----
with tab_detail:
    show_all = st.checkbox("Include Normal machines in the list")
    ranked = table.sort_values("Failure probability", ascending=False)
    pool = ranked if show_all else ranked[ranked["Status"] != "Normal"]

    if pool.empty:
        st.info("No machines to show.")
    else:
        machine = st.selectbox("Select a machine", pool["Product ID"])
        row = pool[pool["Product ID"] == machine].iloc[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Health score", f"{row['Health score']:.1f} / 100")
        m2.metric("Failure probability", f"{row['Failure probability']:.1%}")
        m3.metric("Status", STATUS_ICON[row["Status"]])
        st.info(f"**Recommended action:** {row['Action']}")

        st.subheader("Why this machine is flagged")
        features = SENSORS + ENGINEERED
        values = table.loc[row.name, features].astype(float)
        compare = pd.DataFrame({
            "This machine": values,
            "Fleet median": table[features].median(),
        })
        compare["Difference vs fleet (%)"] = (
            (compare["This machine"] / compare["Fleet median"] - 1) * 100).round(1)
        st.dataframe(compare.round(2))

        model, model_features, _ = get_classifier()
        importance = pd.Series(model.feature_importances_, index=model_features)
        st.caption("What the model relies on overall (global feature importance). "
                   "Compare it with the table above to see which of this machine's "
                   "readings stand out on the features that matter most.")
        st.bar_chart(importance.sort_values())

# ----- Alerts -----
with tab_alerts:
    chosen = st.multiselect("Show statuses", ["Critical", "Warning", "Normal"],
                            default=["Critical", "Warning"])
    alerts = (table[table["Status"].isin(chosen)]
              .sort_values("Failure probability", ascending=False))
    st.write(f"{len(alerts)} machine(s)")
    st.dataframe(for_display(alerts))
    st.download_button("Download as CSV", alerts.to_csv(index=False),
                       file_name="alerts.csv", mime="text/csv")

# ----- Remaining useful life -----
with tab_rul:
    st.write("Remaining useful life (RUL) is predicted for turbofan engines from the "
             "NASA C-MAPSS dataset, a separate model from the failure classifier.")
    if not CMAPSS_DATA.exists():
        st.error(f"Dataset not found: {CMAPSS_DATA}")
    else:
        rul_model, rul_features, rul_config = get_rul_model()
        cmapss = read_cmapss()

        engine_id = st.selectbox("Engine", sorted(cmapss["engine"].unique()))
        eng = cmapss[cmapss["engine"] == engine_id].copy()
        eng["Predicted RUL"] = predict_rul(rul_model, eng[rul_features])
        eng["Actual RUL"] = (eng["cycle"].max() - eng["cycle"]).clip(
            upper=rul_config["rul_cap"])

        first, last = int(eng["cycle"].min()), int(eng["cycle"].max())
        cycle = st.slider("Current cycle", first, last, int(last * 0.8))
        now = eng[eng["cycle"] == cycle].iloc[0]

        r1, r2 = st.columns(2)
        r1.metric("Predicted RUL (cycles)", f"{now['Predicted RUL']:.0f}")
        r2.metric("Actual RUL (cycles)", f"{now['Actual RUL']:.0f}")

        st.line_chart(eng.set_index("cycle")[["Actual RUL", "Predicted RUL"]])
        st.caption(f"RUL is capped at {rul_config['rul_cap']} cycles. These engines "
                   "were part of the model's training data, so this chart illustrates "
                   "behaviour; the held-out accuracy (RMSE 18.6 cycles, cross-validated "
                   "R² 0.81) is the figure to quote.")