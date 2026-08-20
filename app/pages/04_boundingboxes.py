
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

from src.data.dicom_loader import load_dicom
from src.data.labels import load_labels
from src.utils.config import TRAIN_IMAGES, TRAIN_LABELS

# CACHE
@st.cache_data
def load_dataset():

    labels, _ = load_labels(TRAIN_LABELS)

    patient_ids = sorted(labels.keys())

    return labels, patient_ids


labels, patient_ids = load_dataset()

# TITLE
st.title("📦 Bounding Box Viewer")

st.write(
    "Visualize annotated pneumonia regions."
)

st.divider()

# RANDOM BUTTON
if "bbox_index" not in st.session_state:
    st.session_state.bbox_index = 0

if st.button("🎲 Random Positive Patient"):

    positive = [
        pid
        for pid in patient_ids
        if labels[pid]["target"] == 1
    ]

    random_patient = random.choice(positive)

    st.session_state.bbox_index = patient_ids.index(
        random_patient
    )


# SLIDER
selected = st.slider(

    "Patient",

    0,

    len(patient_ids)-1,

    st.session_state.bbox_index

)

patient_id = patient_ids[selected]


# LOAD IMAGE
image_path = TRAIN_IMAGES / f"{patient_id}.dcm"

image, _ = load_dicom(image_path)


# DRAW BOUNDING BOXES
fig, ax = plt.subplots(figsize=(8,8))

ax.imshow(image, cmap="gray")

if labels[patient_id]["target"] == 1:

    for bbox in labels[patient_id]["bounding_boxes"]:

        rect = patches.Rectangle(

            (bbox["x"], bbox["y"]),

            bbox["width"],

            bbox["height"],

            linewidth=2,

            edgecolor="red",

            facecolor="none"

        )

        ax.add_patch(rect)

ax.axis("off")

st.pyplot(fig)


# METADATA
target = labels[patient_id]["target"]

if target == 1:

    st.error("🦠 Pneumonia")

else:

    st.success("✅ Normal")

st.write(f"Patient ID : {patient_id}")

st.write(
    f"Bounding Boxes : {len(labels[patient_id]['bounding_boxes'])}"
)



# COORDINATES
if target == 1:

    st.subheader("Coordinates")

    for i, bbox in enumerate(labels[patient_id]["bounding_boxes"], start=1):

        st.write(f"### Region {i}")

        st.write(
            f"""
            x : {bbox['x']}

            y : {bbox['y']}

            width : {bbox['width']}

            height : {bbox['height']}
            """
        )


