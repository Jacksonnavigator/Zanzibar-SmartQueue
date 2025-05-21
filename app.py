import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.set_page_config(page_title="Zanzibar SmartQueue", layout="wide")

st.title("🛫 Zanzibar SmartQueue – Airport Queue Management")

DATA_PATH = "queue_data.csv"

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

def save_data(new_entry):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)
    return df

# Sidebar input
st.sidebar.header("📥 Add Queue Data")
location = st.sidebar.selectbox("Location", ["Check-In", "Security", "Immigration"])
queue_len = st.sidebar.slider("Queue Length", 0, 100, 10)
if st.sidebar.button("Submit Entry"):
    entry = {
        "timestamp": datetime.now(),
        "location": location,
        "queue_length": queue_len
    }
    df = save_data(entry)
    st.sidebar.success("✅ Data added!")
else:
    df = load_data()

# Display current data
st.subheader("📊 Current Queue Data")
st.dataframe(df.tail(10))

# AI Prediction
st.subheader("📈 AI Prediction and Staff Suggestion")
for loc in df["location"].unique():
    loc_df = df[df["location"] == loc]
    loc_df["hour"] = loc_df["timestamp"].dt.hour
    X = loc_df[["hour"]]
    y = loc_df["queue_length"]
    model = LinearRegression().fit(X, y)

    future_hours = np.arange(24).reshape(-1, 1)
    preds = model.predict(future_hours)

    st.markdown(f"**🔹 Location:** {loc}")
    fig, ax = plt.subplots()
    ax.plot(future_hours, preds, marker='o', label='Predicted Queue Length')
    ax.set_title(f"{loc} – Predicted Queue per Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Queue Length")
    st.pyplot(fig)

    peak_hour = future_hours[np.argmax(preds)][0]
    peak_queue = int(max(preds))
    suggested_staff = int(np.ceil(peak_queue / 15))
    st.info(f"⏰ **Peak at {peak_hour}:00** – Estimated Queue: {peak_queue} → **Staff Needed: {suggested_staff}**")

st.caption("Built with ❤️ for Zanzibar Airport")
