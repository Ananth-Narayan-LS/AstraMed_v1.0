I actually like this idea a lot.

As AstraMed grows, you'll have **100+ interview questions**. Without grouping them by file/module, it'll become difficult to revise.

From Sprint 3 onwards, let's adopt this format.

---

# 📂 File: `src/data/split_data.py`

### Interview Questions

**1.** Why did you split by patient instead of by image?

**2.** What is data leakage? Can you explain it with an example from the RSNA dataset?

**3.** Why did you use `random.seed(42)`? What happens if you don't set a seed?

**4.** Why do the train, validation, and test ratios need to sum to 1? What would happen if they didn't?

**5.** Why convert `labels.keys()` into a list? Why not shuffle the dictionary directly?

**6.** Why do we verify that the three sets have no overlap? What would happen if one patient appeared in both train and validation?

**7.** How would you modify this function if the dataset were highly imbalanced? (Stratified splitting.)

**8.** Is random splitting always the best choice? When would you split based on time, hospital, or geographic location?

**9.** Why did we create `split_data.py` instead of putting the function inside `labels.py`?

**10.** If a new patient is added tomorrow, do we need to regenerate the split? How would you make the split reproducible?

---

# 📂 File: `src/data/dicom_loader.py`

### Interview Questions

**11.** Why does `pydicom` need external plugins to read some DICOM files?

Expected Answer:

> The DICOM standard supports multiple transfer syntaxes. While `pydicom` parses the DICOM structure, compressed pixel data (JPEG Baseline, JPEG-LS, JPEG2000, etc.) requires external decoders such as `pylibjpeg` or `GDCM`.

---

# 📂 File: `src/data/dataset.py`

### Interview Questions

**12.** What is a Dataset in PyTorch?

**13.** Why inherit from `torch.utils.data.Dataset`?

**14.** Why implement `__len__()`?

**15.** Why implement `__getitem__()`?

**16.** Why didn't you load all images inside `__init__()`?

**17.** Why does the Dataset call `load_dicom()` instead of reading DICOM directly?

**18.** Why is `preprocess_image()` called inside the Dataset?

**19.** Why does `preprocess_image()` return a NumPy array instead of a Tensor?

**20.** Why convert to a Tensor inside the Dataset instead of inside preprocessing?

**21.** Why do we call `unsqueeze(0)`?

**22.** Why does each sample return a dictionary instead of a tuple?

**23.** Why do we include `patient_id` in every sample?

**24.** Initially we returned `bounding_boxes`. Why was that a problem?

**25.** What happens internally every time `dataset[100]` is executed?

**26.** Isn't reading DICOM every time slow?

**27.** Why is the Dataset based on patient IDs instead of image filenames?

**28.** Why doesn't the Dataset perform train/test splitting itself?

**29.** We encountered a JPEG-compressed DICOM error. Why?

**30.** How would you make the Dataset more robust?

**31.** If the dataset contained two million images, what improvements would you make?

**32.** Why not preprocess every DICOM into PNG before training?

**33.** Suppose tomorrow the dataset includes patient age and gender. How would you extend the Dataset?

**34.** Why separate:

- `load_dicom()`
- `preprocess_image()`
- `RSNADataset`

instead of putting everything into one file?

**35.** *(Reserved for future Dataset enhancements.)*

**36.** If tomorrow you switch from PyTorch to TensorFlow, what changes?

---

# 📂 File: `src/data/dataloader.py`

### Interview Questions

**37.** What is the difference between a Dataset and a DataLoader?

**38.** Why create `dataloader.py` instead of creating the DataLoader inside the training script?

**39.** Explain the complete AstraMed data pipeline.

**40.** Why did we separate every stage instead of writing one large function?

**41.** What exactly happens when you execute:

```python
batch = next(iter(train_loader))
```

**42.** Why do all images inside one batch need identical dimensions?

**43.** We got:

```text
RuntimeError:
each element in list of batch should be of equal size
```

Why?

**44.** Why did we remove `bounding_boxes` from the Dataset?

**45.** If AstraMed later becomes an object detector, what changes?

**46.** What is a custom `collate_fn`?

**47.** Why don't we shuffle validation and test DataLoaders?

**48.** What does `batch_size` affect?

**49.** What does `num_workers` do?

**50.** Why is the Dataset performing lazy loading?

**51.** Suppose AstraMed scales to one million DICOM images. What would you improve first?

**52.** What software engineering principles can you identify in AstraMed's data pipeline?

**53.** Which component changes if the dataset switches from DICOM to PNG?

**54.** Why is this architecture better than writing everything inside a notebook?

**55.** Walk me through every function call that happens when:

```python
batch = next(iter(train_loader))
```

# 📂 File: src/analysis/dataset_statistics.py

56. Why did you create a separate analysis module instead of putting this inside the notebook?
57. Why compute statistics before training a model?
58. Why is counting positive and negative cases important?
(Hint: Class imbalance.)
59. What does "Average Images per Patient" tell us?
60. Why count images with multiple bounding boxes?
(Hint: Disease severity and localization complexity.)
61. What would happen if the number of images was greater than the number of patients?
(Hint: Multiple scans per patient and potential data leakage.)
62. If tomorrow another hospital contributes new data, which statistics would you compare first before combining the datasets?
(Expected: Class distribution, patient demographics, image properties, annotation quality.)

# 📂 File: src/analysis/class_distribution.py

63. Why is checking class distribution one of the first steps in EDA?
64. What is class imbalance?
Can you explain it using the RSNA dataset?
65. How does class imbalance affect model training?
66. Why is accuracy a poor metric on highly imbalanced datasets?
67. Suppose 95% of the dataset belongs to one class.
What would happen if the model always predicted the majority class?
68. How can we handle class imbalance?
Expected answers:
Class weights
Weighted Cross-Entropy Loss
Focal Loss
Oversampling
Undersampling
Data augmentation
69. Why did we create class_distribution.py instead of writing the calculations directly in the notebook?
70. Why is a bar chart generally better than a pie chart for comparing class counts?
(Expected answer: Bar charts make it easier to compare magnitudes accurately, while pie charts are useful for showing proportions but less precise for comparison.)
71. Why did we include percentages as well as counts?
72. If you discover severe class imbalance, what changes later in the pipeline?
(Hint: Loss function, sampling strategy, evaluation metrics, threshold tuning.)


# 📂 File: src/analysis/patient_analysis.py

73. Why is patient-level analysis more important than image-level analysis in medical AI?
74. Why did we count images from the filesystem instead of inferring them from the labels?
75. Why check for duplicate patient IDs?
76. What does it mean if one patient has multiple images?
77. What does it mean if one patient has multiple bounding boxes?
78. Why analyze only positive patients when studying bounding boxes?
79. Suppose another dataset has five images per patient. How would that affect your preprocessing and splitting strategy?
80. Why is a histogram appropriate for image counts?
81. Why is a bar chart appropriate for bounding box frequencies?
82. How can patient-level analysis influence model design?
Expected discussion points:
Patient-level data leakage prevention.
Multi-instance learning if multiple images exist.
Severity estimation using bounding box counts.
Patient-centric evaluation rather than image-centric evaluation.


# 📂 File: src/analysis/image_analysis.py

83. Why did we use stop_before_pixels=True when reading DICOM files?
(Expected: To read only metadata, avoiding the overhead of loading image pixel data.)
84. Why analyze image dimensions before training?
85. Why is aspect ratio important in medical imaging?
86. What is PixelSpacing? Why does it matter?
(Expected: It represents the physical distance between adjacent pixels, usually in millimeters. It's important because two images can have the same pixel dimensions but represent different physical sizes.)
87. What is the difference between image resolution and pixel spacing?
88. What are AP and PA chest X-rays?
(Expected: AP = Anterior-to-Posterior, PA = Posterior-to-Anterior. They differ in acquisition geometry and can affect the appearance of anatomical structures.)
89. Why check image size consistency?
90. Suppose half the dataset contains 1024×1024 images and the other half 2048×2048. How would you handle this?
91. Why use histograms for width and height distributions?
92. Why use boxplots in addition to histograms?
(Expected: Histograms show the distribution shape, while boxplots highlight medians, quartiles, and potential outliers.)
93. If you find images with very different pixel spacing, how could that affect model performance?
(Expected: The same pathology could occupy different numbers of pixels depending on spacing, potentially confusing the model. Physical-size normalization or spacing-aware preprocessing may be needed.)

# 📂 File: src/analysis/pixel_analysis.py

94. Why analyze pixel intensities before training a CNN?
95. What does mean pixel intensity represent?
96. Why can standard deviation be used as a rough measure of contrast?
97. What is dynamic range?
98. Why manually inspect the darkest and brightest images?
99. Why analyze raw DICOMs instead of normalized images?
100. What could cause unusually dark or bright X-rays?
101. Why use a sample instead of processing every image during exploratory analysis?
102. Why use a fixed random seed for sampling?
103. What is the difference between histogram and density plot?
104. What does a boxplot tell you that a histogram might not?
105. If you discover substantial brightness variation, what preprocessing or augmentation techniques could you consider?
106. Would you automatically apply histogram equalization if brightness variation exists?
Expected answer:
No. I would first determine whether the variation represents meaningful acquisition differences or undesirable artifacts. Applying aggressive contrast enhancement without validation could remove clinically useful information or introduce artificial patterns.

107. Why shouldn't we calculate pixel statistics on normalized images when investigating the original dataset?
108. Why did we use random sampling instead of simply taking the first 1,000 images?
109. How would you determine whether 1,000 images are enough for this EDA?
110. If the brightness distribution of the sample is very different from the full dataset, what would you do?
Expected answer:
Increase the sample size or use a more representative sampling strategy, potentially stratifying by relevant characteristics such as class or acquisition/view type.


# 📂 File: src/analysis/bounding_box_analysis.py

111. What does a bounding box represent in the RSNA dataset?
112. Why does one patient/image sometimes have multiple bounding boxes?
113. How do you calculate bounding box area?
area = width × height
114. How do you calculate bounding box aspect ratio?
aspect ratio = width / height
115. Why might bounding box area be useful for medical-image analysis?
116. What does a bounding box center represent?
117. Why did we calculate:
center_x = x + width / 2
center_y = y + height / 2
118. Why did we use a heatmap for bounding-box centers?
119. Can the heatmap tell us which anatomical lung lobe is affected?
Expected answer:
No. It shows the spatial distribution of annotation centers in image coordinates. Anatomical localization would require additional anatomical information or segmentation.

120. What does it mean if bounding boxes are very small?
Possible discussion:
small opacity regions
annotation granularity
potential localization difficulty
potential label noise
121. What does a very large bounding box indicate?
It could represent a large region of lung opacity, but we shouldn't automatically interpret it as disease severity without clinical validation.
122. Why might multiple bounding boxes exist in one image?
Because pneumonia opacity may occur in multiple distinct regions.
123. Why did we create one row per bounding box in bbox_df?
Because bounding boxes are variable-length annotations. A patient/image can have zero, one, or multiple boxes.
124. Why didn't we simply keep the nested bounding-box dictionaries?
Because a tabular representation makes statistical analysis, visualization, aggregation, and plotting much easier.
125. If AstraMed becomes an object-detection model later, how would these annotations be used?
Expected discussion:
Image
 ↓
Object Detection Model
 ↓
Bounding Boxes
 ↓
Class + Confidence
The existing annotations could become ground-truth targets for detector training/evaluation.
126. What problems could exist with bounding-box annotations?
Possible answers:
inaccurate localization
inconsistent box sizes
overlapping boxes
missing annotations
annotation noise
ambiguous disease boundaries
127. Why should we inspect bounding-box outliers visually?
Because a statistical outlier may be a legitimate medical case rather than bad data.


# 📂 File: src/analysis/metadata_analysis.py

128. What is DICOM metadata?
129. Why is metadata useful in medical imaging?
130. Why did we use stop_before_pixels=True?
131. Why do we convert PatientAge from "030Y" into 30?
132. What does PixelSpacing represent?
133. What is the difference between pixel dimensions and physical dimensions?
134. What are AP and PA X-ray views?
135. Why might AP and PA images have different appearances?
136. Why is patient sex useful to analyze even if we don't use it as a model feature?
137. Why did we analyze the manufacturer?
This is the research-oriented question.
A strong answer:
Different manufacturers or acquisition systems can introduce systematic differences in image appearance. If manufacturer information is correlated with the target, a model could learn scanner-specific shortcuts instead of clinically meaningful features.

138. What is shortcut learning?
139. How can metadata cause data leakage?
140. Suppose 90% of pneumonia-positive cases came from Hospital A and 90% of negative cases came from Hospital B. What's the danger?
The model could learn:
Hospital A → pneumonia
Hospital B → normal
instead of:
X-ray features → pneumonia
That's a classic dataset bias / shortcut learning problem.
141. Why shouldn't we automatically remove all metadata?
Because some metadata can be clinically useful.
We should determine:
whether it improves generalization
whether it introduces leakage
whether it represents legitimate clinical information
whether it will be available during real-world inference
142. Why analyze missing metadata?
Missingness itself can contain information.
Also, a feature that's missing for a large portion of the dataset may not be reliable enough to use directly.
143. Why are we sampling 1,000 images instead of processing all DICOMs?
Same reasoning as Milestone 5:
Metadata extraction is cheaper than pixel decoding, but sampling still allows faster exploratory iteration. Once the analysis is finalized, we can run a full-dataset verification.

144. If manufacturer is highly correlated with pneumonia, what would you investigate?
Expected discussion:
Manufacturer
      ↓
Hospital/site?
      ↓
Acquisition protocol?
      ↓
Patient population?
      ↓
Class imbalance?
      ↓
Potential shortcut?
This is the sort of question that can turn a basic ML interview into a research conversation.

