


import streamlit as st

from src.analysis.metadata_analysis import extract_metadata
from src.utils.config import TRAIN_IMAGES


# CACHE
@st.cache_data
def load_metadata():
    return extract_metadata(TRAIN_IMAGES)


metadata_df = load_metadata()


# PAGE TITLE
st.title("📁 Metadata Explorer")

st.markdown(
"""
Explore patient and image metadata from the RSNA dataset.
"""
)

st.divider()


# SIDE BAR
st.sidebar.header("Filters")

sex = st.sidebar.selectbox(
    "Sex",
    ["All"] + sorted(metadata_df["sex"].dropna().unique().tolist())
)

view = st.sidebar.selectbox(
    "View Position",
    ["All"] + sorted(metadata_df["view_position"].dropna().unique().tolist())
)


# PATIENT SEARCH
search = st.text_input(
    "🔍 Search Patient ID"
)


# FILTERING
filtered = metadata_df.copy()

if sex != "All":
    filtered = filtered[
        filtered["sex"] == sex
    ]

if view != "All":
    filtered = filtered[
        filtered["view_position"] == view
    ]

if search:

    filtered = filtered[
        filtered["patient_id"]
        .str.contains(search, case=False)
    ]


# KPI'S
st.subheader("Dataset Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Patients",
        len(filtered)
    )

with col2:
    st.metric(
        "Average Age",
        round(filtered["age"].mean(), 1)
    )

with col3:
    st.metric(
        "Manufacturers",
        filtered["manufacturer"].nunique()
    )


# METADATA TABLE
st.divider()

st.subheader("Metadata Table")

st.dataframe(
    filtered,
    width="stretch",
    hide_index=True,
)


# DOWNLOAD
st.download_button(
    "📥 Download Filtered Metadata",
    filtered.to_csv(index=False),
    file_name="metadata.csv",
    mime="text/csv",
)


# OPTIONAL STATS
with st.expander("Dataset Statistics"):

    st.write(filtered.describe(include="all"))