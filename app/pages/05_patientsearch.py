
import pandas as pd
import streamlit as st

from src.data.dicom_loader import load_dicom
from src.data.labels import load_labels
from src.utils.config import TRAIN_IMAGES, TRAIN_LABELS


# CACHE
@st.cache_data
def load_dataset():
    labels, _ = load_labels(TRAIN_LABELS)
    return labels


labels = load_dataset()

patient_list = sorted(labels.keys())


# PAGE TITLE
st.title("🔍 Patient Search")

st.write(
    """
Search and explore patient records from the
RSNA Pneumonia Detection Challenge dataset.
"""
)

st.divider()


# PATIENT SELECTION
patient_id = st.selectbox(
    "Search Patient",
    options=patient_list,
    index=None,
    placeholder="Type a Patient ID..."
)

if patient_id is None:
    st.info("👆 Select a patient to generate the report.")
    st.stop()


# LOAD IMAGE
image_path = TRAIN_IMAGES / f"{patient_id}.dcm"

image, metadata = load_dicom(image_path)


# DISPLAY
col1, col2 = st.columns([2.2, 1])

with col1:

    st.image(
        image,
        width="stretch",
        caption=f"Patient ID : {patient_id}",
        clamp=True,
    )

with col2:

    st.subheader("Patient Summary")

    target = labels[patient_id]["target"]

    if target == 1:
        st.error("🦠 Pneumonia")
    else:
        st.success("✅ Normal")

    st.metric(
        "Bounding Boxes",
        len(labels[patient_id]["bounding_boxes"])
    )


# METADATA
st.divider()

st.subheader("Metadata")

metadata_df = pd.DataFrame(
    metadata.items(),
    columns=["Field", "Value"]
)

st.dataframe(
    metadata_df,
    width="stretch",
    hide_index=True,
)


# BOUNDING BOXES
if target == 1:

    st.divider()

    st.subheader("Bounding Boxes")

    bbox_df = pd.DataFrame(
        labels[patient_id]["bounding_boxes"]
    )

    st.dataframe(
        bbox_df,
        width="stretch",
        hide_index=True,
    )


# PATIENT REPORT
st.divider()

st.subheader("Patient Report")

report = {
    "Patient ID": patient_id,
    "Diagnosis": "Pneumonia" if target == 1 else "Normal",
    "Number of Bounding Boxes": len(
        labels[patient_id]["bounding_boxes"]
    ),
}

st.json(report)


# DOWNLOAD REPORT
report_df = pd.DataFrame([report])

st.download_button(
    label="📥 Download Report",
    data=report_df.to_csv(index=False),
    file_name=f"{patient_id}_report.csv",
    mime="text/csv",
)