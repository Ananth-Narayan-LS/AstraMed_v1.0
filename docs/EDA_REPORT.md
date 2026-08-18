
# Dataset Statistics

- Total Patients: 26684
- Total Images: 26684
- Positive Cases: 6012
- Negative Cases: 20672
- Average Images per Patient: 1.0
- Total Bounding Boxes: 9555
- Images with Multiple Bounding Boxes: 3398

# Class Distribution

Positive Cases: 6012
Negative Cases: 20672
Positive Percentage: 22.53
Negative Percentage: 77.47
Observation:
Decision:

# Patient Analysis

- Duplicate Patients 
- Images per Patient 
- Patients with Multiple Bounding Boxes
- Observation
- Engineering Decision

# Image Analysis

- Width Distribution
- Height Distribution
- Aspect Ratio
- Pixel Spacing
- View Position
- Observation
- Engineering Decision

# Bounding Box Analysis

- Bounding Box Width

Observation:

- Bounding Box Height

Observation:

- Bounding Box Area

Observation:

- Bounding Box Aspect Ratio

Observation:

- Number of Bounding Boxes per Patient

Observation:

- Spatial Distribution

Observation:

- Outliers

Observation:

- Engineering Decision

Observation:

# Outlier Detection

## Image Dimension Outliers

Number of width outliers:

Number of height outliers:

Observation:

-

## Bounding Box Outliers

Number of unusually large bounding boxes:

Observation:

-

## Missing Metadata

Observation:

-

## DICOM Integrity

Invalid DICOMs detected in sample:

Observation:

-

## Blank Images

Potential blank images detected:

Observation:

-

## Outlier Gallery

Visual inspection findings:

-

## Engineering Decision

Outliers to investigate further:

-

Outliers to retain:

-

Outliers to remove:

-

# Correlation Study

## Objective

Investigate relationships between patient,
image, metadata, pixel, and label characteristics.

## Age vs Pneumonia

Observation:

-

## Sex vs Pneumonia

Observation:

-

## Brightness vs Pneumonia

Observation:

-

## Bounding Box Area vs Pneumonia

Observation:

-

Important:

Bounding-box area is inherently related to the positive
label because bounding boxes are provided for pneumonia
annotations. Therefore this relationship should not be
interpreted as an independent predictive relationship.

## Images per Patient

Observation:

-

## Correlation Matrix

Strongest observed relationships:

-

## Important Caveats

Correlation does not imply causation.

Some relationships may arise from the way the dataset
was constructed rather than from genuine clinical
relationships.

## Engineering Decisions

-