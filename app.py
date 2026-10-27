import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="SmartPrice", page_icon="📱", layout="wide")
BASE = Path(__file__).resolve().parent

@st.cache_resource
def load_saved():
    model = joblib.load(BASE / "best_smartphone_model.pkl")
    features = joblib.load(BASE / "feature_columns.pkl")
    name = joblib.load(BASE / "best_model_name.pkl")
    scaler_path = BASE / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model, list(features), str(name), scaler

@st.cache_data
def load_prices():
    df = pd.read_csv(BASE / "smartphone_prices.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Old Price (₹)": "Old Price (Rs)", "New Price (₹)": "New Price (Rs)"})
    for col in ("Model", "Old Price (Rs)"):
        if col not in df.columns:
            raise ValueError(f"Missing CSV column: {col}")
    df["Model"] = df["Model"].astype(str).str.strip()
    df["Old Price (Rs)"] = pd.to_numeric(df["Old Price (Rs)"].astype(str).str.replace(r"[₹,\s]", "", regex=True), errors="coerce")
    return df.dropna(subset=["Model", "Old Price (Rs)"])

try:
    model, features, model_name, scaler = load_saved()
    df = load_prices()
except Exception as e:
    st.error(f"Unable to load files: {e}")
    st.stop()

# The training code one-hot encodes Model and retains Old Price.
model_features = [c for c in features if c.startswith("Model_")]
if "Old Price (Rs)" not in features or not model_features:
    st.error("These saved files were not trained with both Model and Old Price. Retrain with both inputs and replace the .pkl files.")
    st.stop()

if getattr(model, "n_features_in_", len(features)) != len(features):
    st.error("The saved model and feature_columns.pkl do not match. Save them from the same training run.")
    st.stop()

st.title("📱 SmartPrice")

models = sorted(df["Model"].unique().tolist())
selected_model = st.selectbox("Smartphone Model", models)
example_price = float(df.loc[df["Model"] == selected_model, "Old Price (Rs)"].iloc[0])
old_price = st.number_input("Old Price (₹)", min_value=1.0, value=example_price, step=500.0)

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Predict New Price", type="primary"):
    row = pd.DataFrame(0.0, index=[0], columns=features)
    row.loc[0, "Old Price (Rs)"] = old_price
    encoded_model = "Model_" + selected_model
    if encoded_model in features:
        row.loc[0, encoded_model] = 1.0
    # A model absent from the dummy columns can be the dropped baseline.

    linear_names = {"Linear Regression", "Ridge Regression", "Lasso Regression"}
    if model_name in linear_names:
        if scaler is None:
            st.error("scaler.pkl is required for this trained model. Save it from your training notebook.")
            st.stop()
        model_input = scaler.transform(row)
    else:
        model_input = row

    try:
        predicted = float(model.predict(model_input)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    difference = predicted - old_price
    change_pct = difference / old_price * 100
    result = {
        "Model": selected_model,
        "Old Price (Rs)": round(old_price, 2),
        "Predicted New Price (Rs)": round(predicted, 2),
        "Difference (Rs)": round(difference, 2),
        "Change (%)": round(change_pct, 2),
    }
    st.session_state.history.append(result)
    st.session_state.last_result = result

if "last_result" in st.session_state:
    r = st.session_state.last_result
    old, new, diff, pct = r["Old Price (Rs)"], r["Predicted New Price (Rs)"], r["Difference (Rs)"], r["Change (%)"]
    st.success(f"Predicted New Price: ₹{new:,.2f}")
    a, b, c = st.columns(3)
    a.metric("Old Price", f"₹{old:,.2f}")
    b.metric("Predicted New Price", f"₹{new:,.2f}")
    c.metric("Price Difference", f"₹{diff:+,.2f}", delta=f"{pct:+.2f}%")

    st.subheader("Price Comparison")
    st.bar_chart(pd.DataFrame({"Price (₹)": [old, new]}, index=["Old Price", "Predicted New Price"]))

    st.subheader("Prediction Report")
    report = pd.DataFrame([r])
    st.dataframe(report, hide_index=True, use_container_width=True)
    st.download_button("Download CSV Report", report.to_csv(index=False), "smartphone_prediction_report.csv", "text/csv")

st.subheader("Prediction History")
if st.session_state.history:
    history = pd.DataFrame(st.session_state.history)
    st.dataframe(history, hide_index=True, use_container_width=True)
    st.download_button("Download History CSV", history.to_csv(index=False), "smartphone_prediction_history.csv", "text/csv")
    if st.button("Clear History"):
        st.session_state.history = []
        st.session_state.pop("last_result", None)
        st.rerun()
else:
    st.write("No predictions yet.")
