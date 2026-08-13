## Sprint 2 - Milestone 1 -------------------

**Status:** ✅ Completed

### Goal
Build a reusable DICOM reader.

### Deliverables
- [x] Read DICOM
- [x] Extract pixel array
- [x] Convert to NumPy
- [x] Preserve metadata

### Files Created
- src/data/dicom_loader.py

### Lessons Learned
- DICOM stores both image data and rich medical metadata.
- `pixel_array` extracts the image as a NumPy array.
- Relative paths in notebooks depend on the current working directory.
- Keeping `load_dicom()` focused makes the pipeline reusable.

## Sprint 2 - Milestone 2 ------------------------
Image Preprocessing

## preprocess_image()

Purpose:
Prepare DICOM images for machine learning.

Steps:
1. Resize to 512×512
2. Convert to float32
3. Normalize to [0,1]

Returns:
NumPy ndarray


## Sprint 2 - Milestone 3 ------------------------

## Label Loader

Purpose:
Load RSNA labels and organize them by patient.

Output Structure:

patient_id
    ├── target
    └── bounding_boxes

Statistics Returned:
- Number of patients
- Positive cases
- Negative cases
- Total bounding boxes