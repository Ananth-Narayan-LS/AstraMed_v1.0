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


# 📂 File: src/analysis/outlier_detection.py
145. What is an outlier?
146. Why shouldn't we automatically remove outliers?
147. What is the IQR method?
Expected:
IQR = Q3 - Q1

Lower bound = Q1 - 1.5 × IQR

Upper bound = Q3 + 1.5 × IQR
148. Why did we use IQR instead of simply using mean ± standard deviation?
Because IQR is more robust to extreme values and does not assume a normally distributed variable.
149. How did we detect unusually large bounding boxes?
150. How can you detect a corrupted DICOM?
Possible checks:
- DICOM can be opened.
- Required metadata exists.
- Pixel data can be decoded.
- Pixel array has valid dimensions.
- Pixel values are not unexpectedly invalid.
151. Why does pydicom.dcmread() succeeding not necessarily mean the DICOM is completely valid?
Because parsing the DICOM structure and successfully decoding its pixel data are separate operations.
That's particularly relevant because we already encountered compressed DICOMs in AstraMed.
152. How did we define a blank image?
A strong answer:
We used extremely low pixel-intensity variation as a screening heuristic. It is not a clinically validated definition of a blank X-ray, so flagged samples require visual inspection.

153. Why is std < 5 not necessarily a universal threshold?
Because pixel distributions depend on the dataset, acquisition system, bit depth, preprocessing, and modality.
154. Why do we need an outlier gallery?
Because numerical statistics can tell us that something is unusual, but visual inspection helps determine why it is unusual.
155. What would you do if an outlier is actually a valid severe pneumonia case?
Keep it.
An unusual medical case isn't bad data merely because it's statistically unusual.
156. What would you do if an image is corrupted?
Depending on the situation:
- replace/re-download it
- exclude it from training
- document the exclusion
- preserve the original raw file
157. Why do we sample 1,000 images for corruption and blank-image checks?
Because these checks require pixel decoding and are expensive.
For exploratory analysis, sampling enables fast iteration.
158. How would you perform final dataset validation?
Run the integrity checks over all images, not just the sample, and save the results as a reproducible validation report.
159. What's the difference between an outlier and a data error?
This is a good one.
An outlier is unusual relative to the distribution. A data error is incorrect, corrupted, or invalid data. An outlier can be perfectly valid.


# 📂 File: src/analysis/correlation_analysis.py
160. What is correlation?
161. What is the difference between correlation and causation?
162. Why did we use Spearman correlation?
163. What does a correlation of +1 mean?
164. What does a correlation of 0 mean?
165. What does a negative correlation mean?
166. Why can't we conclude that age causes pneumonia just because they correlate?
167. Why did we convert sex from categorical values to numeric values?
168. Why did we aggregate bounding boxes before correlation analysis?
Expected:
Because one image can have multiple bounding boxes. Correlation analysis requires observations at a consistent unit of analysis, so we aggregate bounding-box information to the image/patient-ID level.

169. Why would bounding-box area show a strong correlation with the target?
Expected:
Because the bounding boxes are pneumonia annotations. Positive samples have bounding boxes while negative samples don't, so the relationship is partly built into the labeling process.

170. Why shouldn't bounding-box area be used as an input feature for our classification model?
Because it would require the ground-truth pneumonia annotation to exist at inference time.
That's label leakage.
171. Why did we sample 1,000 images?
Because pixel-level statistics are computationally expensive and we're still in exploratory analysis.
172. What happens if the sampled metadata and pixel datasets don't contain the same patient IDs?
The merge produces missing values.
That's why we explicitly inspect:
correlation_df.isna().sum()
173. Why is correlation analysis useful before modeling?
It can reveal:
- potential relationships
- redundant variables
- dataset bias
- suspicious shortcuts
- possible leakage
- variables worth investigating further
174. Is a high correlation always useful for machine learning?
No.
A high correlation may be:
- caused by leakage
- caused by dataset construction
- unstable
- non-causal
- unavailable during inference
This is especially important in medical ML.

# 📂 File: pages/01_Dataset.py
109. Why cache the dataset using @st.cache_data?
110. Why separate data loading from UI rendering?
111. Why reuse analysis functions instead of recalculating everything inside the Streamlit page?
112. Why use st.metric() for KPIs?
113. Why present both tables and charts instead of only charts?
114. Why include an Engineering Insight section?
(Expected answer: The dashboard should not only visualize the data but also communicate actionable decisions that influence preprocessing, modeling, and evaluation.)


# 📂 File: pages/02_Image_Explorer.py
115. Why cache the labels instead of reading the CSV every time?
116. Why use st.session_state?
Expected Answer
To preserve the selected image across Streamlit reruns. Without it, interactions like clicking the random button would reset the page state.
117. Why store only the selected index in session_state instead of the image itself?
Expected Answer
Images are large objects. Storing only the index keeps the session lightweight and allows images to be loaded on demand.
118. Why separate image loading (load_dicom) from UI rendering?
119. Why show metadata alongside the image?
Expected Answer
Metadata provides context such as acquisition parameters and image dimensions, making the viewer more useful for debugging and analysis.
120. Why use st.image() instead of Matplotlib for displaying a single image?
121. If the dataset contained 500,000 images, what improvements would you make?
Possible answers:
- Search by patient ID instead of only a slider.
- Pagination.
- Thumbnail previews.
- Cached metadata.
- Database indexing.
- Lazy loading.


# 📂 File: pages/03_Metadata.py
122. Why cache the metadata DataFrame using @st.cache_data?
Expected Answer
Metadata extraction from thousands of DICOM files is expensive. Caching prevents repeated reads and significantly improves dashboard responsiveness.
123. Why perform filtering on a copy (filtered = metadata_df.copy()) instead of modifying the original DataFrame?
Expected Answer
To preserve the cached dataset and avoid side effects. The original data remains unchanged while users interact with filters.
124. Why use sidebar filters instead of embedding them in the main page?
Expected Answer
Filters are controls rather than content. Placing them in the sidebar keeps the main page focused on results and improves usability.
125. Why support both dropdown filters and free-text patient search?
Expected Answer
Dropdowns are useful for categorical exploration, while text search allows quick lookup of a known patient. Together they support both exploratory analysis and targeted inspection.
126. Why provide CSV export from the dashboard?
Expected Answer
It allows filtered subsets of the metadata to be shared, analyzed further, or used in downstream workflows without rerunning code.



# 📂 File: app/pages/04_Bounding_Box.py
127. Why are bounding boxes displayed only for positive patients?
Expected Answer
Negative patients have no pneumonia regions, so there are no bounding box annotations. Displaying boxes only for positive cases reflects the dataset correctly.
128. Why use matplotlib.patches.Rectangle instead of modifying the image array directly?
Expected Answer
Drawing overlays keeps the original image unchanged. It's a non-destructive visualization approach that's easier to customize and debug.
129. Why is the bounding box represented by (x, y, width, height) instead of two corner points?
Expected Answer
The RSNA dataset stores annotations in COCO-style format using the top-left corner and dimensions. This is a common annotation format and is convenient for drawing and converting to other formats if needed.
130. Why doesn't AstraMed v1 use bounding boxes during training?
Expected Answer
AstraMed v1 is a classification model that predicts whether pneumonia is present. Bounding boxes are useful for visualization and future localization or detection models, but they are not required as training inputs for classification.
131. If AstraMed evolves into an object detection model, what changes?
Expected Answer
The dataset would return images along with bounding boxes and class labels. The training pipeline would switch to an object detection framework (e.g., Faster R-CNN, YOLO), and the DataLoader would use a custom collate_fn to handle variable numbers of boxes per image.


# 📂 File: app/pages/05_Patient_Search.py
132. Why validate the patient ID before loading the image?
Expected Answer
To prevent unnecessary file access and provide immediate feedback if the patient does not exist. It avoids runtime errors and improves user experience.
133. Why generate the report dynamically instead of storing reports for every patient?
Expected Answer
The report is derived from the latest data. Generating it on demand avoids duplication, reduces storage, and ensures consistency if the underlying data changes.
134. Why iterate over metadata.items() instead of referencing individual fields?
Expected Answer
It makes the UI resilient to schema changes. New metadata fields automatically appear without modifying the rendering code.
135. Why provide a downloadable report?
Expected Answer
It demonstrates how processed results can be exported for sharing, documentation, or integration with downstream workflows, making the application more practical.
136. How would you extend this page if AstraMed included model predictions?
Expected Answer
The report could include predicted probabilities, confidence scores, Grad-CAM visualizations, and comparisons between ground-truth labels and model predictions. This would transform the page from a dataset explorer into a clinical decision-support interface.


# 📂 File: src/data/dataloader.py
56. What is the purpose of verifying the DataLoader before training a model?
Expected Answer
To ensure the entire data pipeline works correctly before introducing model-related complexity. This helps isolate bugs early in the development process.
57. What does next(iter(train_loader)) do internally?
Expected Answer
It creates an iterator over the DataLoader and retrieves the first batch. Internally, the DataLoader requests sample indices, calls Dataset.__getitem__() for each sample, and collates them into a batch.
58. Why is the image tensor shape (batch_size, 1, 256, 256)?
Expected Answer
- batch_size → Number of samples in the batch.
- 1 → Single grayscale channel.
- 256 × 256 → Image dimensions after preprocessing.
This is the standard input format expected by CNNs.
59. Why is the image datatype torch.float32?
Expected Answer
Neural networks perform floating-point computations. Converting images to float32 ensures compatibility with PyTorch layers and efficient GPU computation.
60. Why are labels stored as integers instead of floats?
Expected Answer
Classification labels represent discrete classes (0 or 1). Loss functions like CrossEntropyLoss expect integer class indices rather than floating-point values.
61. Why do we check for NaN and Inf values before training?
Expected Answer
NaN or Inf values can propagate through the network, causing unstable gradients, exploding losses, or complete training failure. Detecting them early prevents difficult debugging later.
62. Why do all images in a batch need identical shapes?
Expected Answer
The DataLoader stacks individual tensors into a single batch tensor. Tensor stacking requires every sample to have exactly the same dimensions.
63. Why do we visualize a batch before training?
Expected Answer
Visual inspection confirms that preprocessing, normalization, labels, and image orientation are correct. It is often the fastest way to detect pipeline errors.
64. Why do we verify class distribution in a batch?
Expected Answer
To ensure that both classes are being loaded correctly and to identify severe class imbalance, which may require weighted loss functions or sampling strategies later.


# 📂src/models/baseline_cnn.py
66. Why do we build a baseline CNN before using pretrained models?
Expected Answer
A baseline provides a reference point. It helps measure whether more advanced architectures genuinely improve performance rather than assuming they do.
67. Why does the first convolution use in_channels=1?
Expected Answer
Chest X-rays are grayscale images, so each image has a single channel.
68. Why does the second convolution use in_channels=32?
Expected Answer
The second convolution receives the 32 feature maps produced by the first convolution as its input.
69. What is the purpose of the ReLU activation?
Expected Answer
ReLU introduces non-linearity, allowing the network to learn complex patterns. It also helps mitigate the vanishing gradient problem compared to sigmoid or tanh in hidden layers.
70. Why do we use MaxPooling?
Expected Answer
MaxPooling reduces spatial dimensions, decreases computation, increases the receptive field, and retains the most salient features.
71. Why do we flatten the feature maps?
Expected Answer
Fully connected layers operate on vectors, so the 3D feature maps must be converted into a 1D representation.
72. Why is the final layer Linear(128, 1)?
Expected Answer
The task is binary classification, so the network outputs a single logit (or probability after sigmoid) representing the likelihood of pneumonia.
73. Why is the output shape (batch_size, 1)?
Expected Answer
Each image receives one prediction corresponding to the probability of belonging to the positive class (pneumonia).
74. Why don't we use Batch Normalization or Dropout yet?
Expected Answer
The objective of a baseline is simplicity. Additional components introduce more variables, making it harder to evaluate the impact of future improvements.
75. How would you improve this model in later experiments?
Expected Answer
Possible improvements include:
- Adding Batch Normalization.
- Introducing Dropout for regularization.
- Increasing network depth.
- Using residual connections.
- Applying data augmentation.
- Leveraging pretrained architectures such as ResNet or EfficientNet.
- Fine-tuning hyperparameters.


# 📂src/training/losses.py
76. Why did you create a separate losses.py module?
Expected Answer
To separate training configuration from the training loop. This makes it easy to experiment with different loss functions without modifying the core training logic.
77. Why are we using BCELoss for the baseline model?
Expected Answer
The model performs binary classification and its final layer applies a Sigmoid, producing probabilities between 0 and 1. BCELoss is designed to compare these probabilities against binary labels.
78. What is the difference between BCELoss and BCEWithLogitsLoss?
Expected Answer
- BCELoss expects probabilities, so the model must include a Sigmoid.
- BCEWithLogitsLoss expects raw logits and internally applies a numerically stable sigmoid operation. It is generally preferred for production models.
79. Why don't we use CrossEntropyLoss?
Expected Answer
CrossEntropyLoss is intended for multi-class classification and expects class logits for two or more classes. Our baseline predicts a single probability for binary classification.
(Note: You could also model binary classification with two output neurons and CrossEntropyLoss, but for this architecture a single-output binary formulation is simpler and more common.)


# 📂src/training/optimizers.py
80. Why did you create a separate optimizers.py module?
Expected Answer
To decouple optimizer selection from the training loop. This makes experimentation and configuration changes much easier.
81. Why are we using Adam for the baseline?
Expected Answer
Adam combines momentum with adaptive learning rates, converges quickly on many problems, and requires less manual tuning than SGD, making it a strong default choice for an initial baseline.
82. When would you choose SGD instead of Adam?
Expected Answer
SGD with momentum is often preferred for large-scale vision models because it can generalize better after careful tuning, although it may converge more slowly.
83. Why is the learning rate set to 1e-3?
Expected Answer
1e-3 is a widely used default for Adam that provides a good balance between convergence speed and stability. It serves as a sensible starting point before hyperparameter tuning.
84. Why does the optimizer receive model.parameters()?
Expected Answer
The optimizer needs access to all trainable parameters so it can update their values using the gradients computed during backpropagation.
85. How would you extend this module for future experiments?
Expected Answer
You could add support for:
- AdamW
- RMSprop
- Focal Loss
- Weighted BCE for class imbalance
- Learning rate schedulers
- Optimizer configuration via YAML or JSON


# 📂src/training/train.py
86. Why do we call model.train() before training?
Expected Answer
It switches the model to training mode. Layers such as Dropout and Batch Normalization behave differently during training and evaluation. Calling model.train() ensures they operate correctly.
87. Why do we move images and targets to the device?
Expected Answer
The model and data must reside on the same device (CPU or GPU). Otherwise, PyTorch raises a device mismatch error.
88. Why do we convert targets to float()?
Expected Answer
BCELoss expects floating-point targets with values in the range [0, 1]. Our labels are integers, so we convert them to float.
89. Why do we call unsqueeze(1) on the targets?
Expected Answer
The model outputs tensors of shape (batch_size, 1), while labels are initially (batch_size,). Adding a dimension makes their shapes compatible for loss computation.
90. Why do we call optimizer.zero_grad() before backward()?
Expected Answer
PyTorch accumulates gradients by default. Calling zero_grad() clears gradients from the previous iteration so each update is based only on the current batch.
91. What happens during the forward pass?
Expected Answer
The input images pass through the CNN, producing predicted probabilities for each sample.
92. What does loss.backward() compute?
Expected Answer
It computes the gradients of the loss with respect to every trainable parameter using backpropagation.
93. What does optimizer.step() do?
Expected Answer
It updates the model's parameters using the computed gradients according to the chosen optimization algorithm (e.g., Adam).
94. Why do we accumulate loss.item() instead of loss?
Expected Answer
loss.item() converts the tensor to a Python scalar, avoiding unnecessary computation graph retention and reducing memory usage.
95. Why do we return the average epoch loss instead of the last batch loss?
Expected Answer
The average loss reflects the model's performance across the entire dataset and is a more reliable metric for tracking training progress than the loss from a single batch.

# 📂src/experiments/config.py
1. Why use a dataclass for experiment configuration?
A dataclass provides a structured container for configuration values while automatically generating methods such as __init__ and providing type annotations.
2. Why centralize experiment configuration?
To prevent hyperparameters from being hardcoded across multiple files and to ensure experiments are reproducible and comparable.
3. Why generate experiment names automatically?
An experiment name provides an identifiable record of the configuration used to produce a particular result.
4. Why include the seed in the experiment name?
Because the seed affects randomness and therefore can affect the resulting model.
5. Why use Path instead of strings for directories?
pathlib.Path provides an OS-independent and safer interface for filesystem operations.
6. Why shouldn't we hardcode outputs/experiments in multiple files?
Because duplicated path definitions create configuration drift and make the project harder to maintain.
7. Why reuse src/utils/config.py?
It already defines the project's root and major directory paths. Reusing it maintains a single source of truth for project paths.
8. Why separate experiment configuration from training code?
Configuration describes what experiment to run, while training code describes how training is performed. This is separation of concerns.
src/utils/reproducibility.py
9. Why do we need random seeds in machine learning?
Machine-learning pipelines often contain stochastic operations. Setting seeds makes experiments more repeatable.
10. Why set seeds for Python, NumPy and PyTorch separately?
Each library maintains its own random-number generator.
11. What does torch.manual_seed() control?
It sets the random seed used by PyTorch's random-number generation.
12. Why call torch.cuda.manual_seed_all()?
It sets the CUDA random seed across available CUDA devices.
13. What is the purpose of cudnn.deterministic = True?
It asks cuDNN to use deterministic algorithms where supported, improving reproducibility.
14. What is the disadvantage of deterministic execution?
Deterministic algorithms can sometimes be slower than optimized nondeterministic algorithms.
src/experiments/tracker.py
15. Why do we need an experiment tracker?
To persist configuration, metadata, training history and experiment status rather than relying only on terminal output.
16. Why store training history separately from metadata?
Metadata describes the experiment itself, while history describes the results produced during training.
17. Why save experiment results as JSON?
JSON is lightweight, human-readable and easy for Python and other tools to consume.
18. Why track model metadata?
It allows us to compare model performance together with computational characteristics such as parameter count and device.
19. Why record experiment status?
It lets us distinguish between experiments that completed successfully and experiments that failed during execution.


# 📂 🎤 Interview questions — Milestone 2
src/data/advanced_input.py
Q1. Why does an ImageNet-pretrained model require 3 channels?
Because the original ImageNet models were trained on RGB images, so their first convolutional layer expects 3 input channels.
Q2. Why not modify RSNADataset to return 3 channels?
Because the dataset is a shared data-layer component and the existing Sprint 4 baseline depends on its [B,1,H,W] contract. Modifying it would unnecessarily couple the dataset to Sprint 5 model requirements.
Q3. Why replicate grayscale instead of learning a grayscale-to-RGB conversion?
Replication preserves the original X-ray information without introducing another trainable transformation or changing the experimental comparison.
Q4. Does channel replication create actual RGB information?
No. It creates three copies of the same grayscale information. It only satisfies the expected three-channel tensor interface.
Q5. Why use ImageNet normalization?
Because pretrained ImageNet weights were learned under that preprocessing convention, so using the corresponding normalization helps maintain compatibility with the pretrained feature representations.
Q6. Why resize to 224 instead of keeping 512?
224 is the standard input resolution for many ImageNet-pretrained architectures and significantly reduces memory and computation, which is important for controlled comparisons on the available GPU.
Q7. Why have both dataset_image_size and model_image_size?
Because the dataset layer and model-input layer have different responsibilities. The dataset maintains a canonical representation, while the adapter prepares that representation for the advanced model.
src/experiments/config.py
Q8. Why should image size be part of experiment configuration?
Because preprocessing is part of an experiment's definition. Recording it makes experiments reproducible and comparable.
Q9. What happens if every model handles preprocessing differently?
The architecture comparison becomes confounded because differences in performance could come from preprocessing rather than the model architecture.
Q10. Why is this adapter better than putting repeat() inside ResNet code?
Because the adapter separates input preparation from model architecture. Every advanced model can therefore consume the same standardized tensor interface.


# 📂 src/models/resnet.py
Q1. Why use pretrained ResNet-50?
It starts with features learned from ImageNet, potentially reducing the amount of medical-image training needed compared with learning every feature from scratch.
Q2. Why replace the final classification layer?
The original head predicts 1,000 ImageNet classes. Our task requires one binary-classification logit.
Q3. Why return logits instead of probabilities?
The advanced training pipeline uses BCEWithLogitsLoss, which combines sigmoid and binary cross-entropy in a numerically stable operation.
Q4. Why use the model factory from this file?
It avoids duplicating architecture-construction logic and keeps all model creation consistent.
scripts/test_resnet50.py
Q5. Why run model.eval()?
It switches layers such as BatchNorm into evaluation behavior instead of updating their training statistics.
Q6. Why use torch.no_grad()?
It disables gradient tracking during inference, reducing memory consumption.
Q7. Why count total and trainable parameters separately?
Total parameters describe model size. Trainable parameters show how many parameters currently have requires_grad=True.
Q8. Does a successful forward pass prove that ResNet-50 detects pneumonia accurately?
No. It proves that the input can pass through the model and produce a valid output. Accuracy requires training or evaluation on labeled data.

# 📂 src/models/densenet.py
Q1. What is DenseNet?
DenseNet is a convolutional neural network architecture in which each layer receives feature maps from earlier layers within a dense block.
Q2. What is feature reuse?
Instead of discarding earlier representations, subsequent layers can use features extracted by preceding layers.
Q3. Why replace the classifier?
The original classifier predicts 1,000 ImageNet classes. Our task requires one binary-classification logit.
Q4. Why use the model factory?
It centralizes model construction and keeps architecture-specific code out of the training pipeline.
scripts/test_densenet121.py
Q5. Why use pretrained=True?
To initialize DenseNet-121 with weights learned on ImageNet instead of random initialization.
Q6. Why use model.eval() and torch.no_grad()?
model.eval() switches layers into evaluation behavior, while torch.no_grad() avoids storing gradients during inference.
Q7. Why count parameters?
Parameter count helps compare model size and computational requirements, although it does not directly measure detection quality.
Q8. Does the smoke test establish that DenseNet is better than ResNet?
No. It establishes that the model loads and runs correctly. Comparing performance requires controlled training and validation.


# 📂 src/models/efficientnet.py
Q1. Why choose EfficientNet-B0?
It is the baseline variant of the EfficientNet family, designed to balance model size and computational cost. It is a sensible starting point for a GPU with limited memory.
Q2. What is compound scaling?
EfficientNet scales network depth, width, and input resolution together using a coordinated scaling strategy.
Q3. Why replace the classification head?
The original model predicts 1,000 ImageNet classes; AstraMed needs a single binary-classification logit.
Q4. Why reuse the model factory?
It centralizes architecture construction and avoids duplicating model-specific code.
scripts/test_efficientnet_b0.py
Q5. Why test with a batch size of one?
It is a lightweight way to verify model construction and inference before committing to larger batches or training.
Q6. Does low parameter count guarantee low GPU memory usage?
No. Activations, intermediate feature maps, gradients, optimizer states, and batch size also affect memory consumption.
Q7. Does this smoke test establish EfficientNet-B0's accuracy?
No. It verifies that the model runs correctly. Accuracy must be measured on labeled validation data after training or appropriate evaluation.

# 📂src/models/convnext.py
Q1. What is ConvNeXt?
A convolutional neural network architecture that modernizes conventional CNN design using ideas inspired by newer deep-learning architectures.
Q2. Why ConvNeXt-Tiny instead of a larger variant?
It is a reasonable starting point for limited GPU memory while still providing a modern architecture to evaluate.
Q3. Why replace the classifier?
The original head predicts 1,000 ImageNet classes. AstraMed requires one binary-classification logit.
Q4. Why reuse model_factory.py?
It centralizes model creation, prevents duplicated implementations, and keeps training code independent of architecture-specific details.
scripts/test_convnext_tiny.py
Q5. Why test a forward pass before training?
It catches model construction, tensor-shape, device-placement, and output-shape errors before spending time on training.
Q6. Does a successful CUDA forward pass guarantee training will fit in GPU memory?
No. Training requires additional memory for gradients, optimizer states, and saved activations.
Q7. Why record parameter count and GPU memory?
They help compare model complexity and inference resource requirements alongside validation performance.


# 📂 src/models/vit.py
Q1. How does Vision Transformer process an image?
It divides the image into patches, converts each patch into an embedding, adds positional information, and processes the resulting token sequence using transformer layers.
Q2. What does /16 mean in ViT-B/16?
The image is divided into patches of \(16 \times 16\) pixels. For a 224 × 224 image, that produces 196 image patches.
Q3. Why does ViT use positional embeddings?
Self-attention does not inherently encode the spatial order of tokens. Positional embeddings provide information about where patches occur in the image.
Q4. Why replace the classification head?
The pretrained head predicts 1,000 ImageNet classes. AstraMed needs one binary-classification logit.
scripts/test_vit_b16.py
Q5. Why test at batch size one?
It minimizes memory requirements while verifying that the model can process an image successfully.
Q6. Why is input resolution important for ViT?
Changing resolution changes the patch grid and the number of tokens. The pretrained model's positional embeddings and input configuration must support the chosen resolution.
Q7. Does a successful forward pass prove that ViT is good at detecting pneumonia?
No. It verifies the implementation and tensor flow. Detection performance must be measured on labeled data.


# 📂 7. Interview questions — Milestone 9
src/experiments/runner.py
Q1. Why build a unified experiment runner?
It standardizes training and evaluation across architectures, reducing duplicated code and making experiments easier to reproduce.
Q2. Why use a wrapper around advanced models?
It inserts preprocessing into the forward path without modifying the existing dataset or baseline implementation.
Q3. Why use different loss functions for the baseline and advanced models?
The baseline outputs sigmoid probabilities and uses BCELoss. The advanced models output logits and use BCEWithLogitsLoss.
Q4. Why track PR-AUC as well as ROC-AUC?
PR-AUC emphasizes the precision-recall tradeoff and is particularly informative when positive cases are relatively uncommon.
Q5. Why save the best checkpoint using validation loss?
It provides a consistent checkpoint-selection rule. We should still inspect validation metrics such as ROC-AUC and recall before deciding which model is best overall.
Q6. Why not use the test set during architecture selection?
Repeatedly tuning against test results leaks information from the test set into model selection, producing an overly optimistic estimate of generalization.
scripts/run_experiment.py
Q7. Why have a separate script for launching experiments?
It separates command-line execution from the reusable runner, allowing the same runner to be called by scripts, notebooks, or other orchestration tools.
Q8. Why start with a tiny smoke test?
It detects integration errors before committing significant compute time to a full training run.


# 📂 