

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.data.labels import load_labels
from src.analysis.dataset_statistics import generate_dataset_statistics
from src.analysis.class_distribution import (
    analyze_class_distribution,
    recommend_class_weighting,
)
from src.utils.config import TRAIN_LABELS

# CACHE
@st.cache_data
def load_dataset():

    labels,_ = load_labels(TRAIN_LABELS)
    stats_df = generate_dataset_statistics(labels)
    class_df = analyze_class_distribution(labels)
    recommendation = recommend_class_weighting(class_df)

    return labels, stats_df, class_df, recommendation

labels, stats_df, class_df, recommendation = load_dataset()

# PAGE TITLE
st.title("📊 Dataset Statistics")

st.markdown(
    """
    Explore the overall characteristics of the
    **RSNA Pneumonia Detection Challenge** dataset.
    """
)
st.divider()

# KEY METRICS
st.subheader("📌 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

stats = dict(zip(stats_df["Statistic"], stats_df["Value"]))

with col1:
    st.metric(
        "Patients",
        int(stats["Total Patients"])
    )

with col2:
    st.metric(
        "Images",
        int(stats["Total Images"])
    )

with col3:
    st.metric(
        "Positive",
        int(stats["Positive Cases"])
    )

with col4:
    st.metric(
        "Negative",
        int(stats["Negative cases"])
    )

st.divider()

# DATASET SUMMARY
st.subheader("📋 Dataset Summary")

st.dataframe(
    stats_df,
    width="stretch",
    hide_index=True,
)

st.divider()

# CLASS DISTRIBUTION TABLE
st.subheader("📈 Class Distribution")

st.dataframe(
    class_df,
    width="stretch",    
    hide_index=True, 
)

st.divider()

# BAR CHART
st.subheader("📊 Class Count")

fig, ax = plt.subplots(figsize=(6,4))

bars = ax.bar(class_df["Class"], class_df["Count"])

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha = "center",
        va = "bottom",
    )

ax.set_xlabel("Class")
ax.set_ylabel("Patients")
st.pyplot(fig)
st.divider()

# PIE CHART
st.subheader("🥧 Class Percentage")

fig, ax = plt.subplots(figsize=(6,6))

ax.pie(
    class_df["Count"],
    labels = class_df["Class"],
    autopct = "%1.1f%%",
    startangle = 90,
)

ax.axis("equal")
st.pyplot(fig)
st.divider()

# ENGINEERING INSIGHT
st.subheader("🧠 Engineering Insight")

st.info(recommendation)

st.markdown(
    """
    **How this influences the model**

    - Check whether class imbalance exists.
    - Decide if weighted loss is required.
    - Decide if focal loss should be explored.
    - Plan evaluation metrics beyond accuracy.
    """
)

st.divider()

# RAW DATA
with st.expander("🔍 View Raw Statistics"):
    st.write(stats_df)
    st.write(class_df)