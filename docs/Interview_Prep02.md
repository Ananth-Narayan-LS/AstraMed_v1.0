

changes to be made :
1. answer in 1 line for all questions
2. maintain the question numbering sequence
3. for each file name in 1 line tell what it contributes to the project pipeleine and where all it is used .
make sure its easy to read

---

# 📂 File: `src/data/split_data.py`

**Contribution:** Splits the dataset into train, validation, and test sets at the **patient level** to prevent data leakage; 
used by `dataset.py`, notebooks, and the training pipeline.

### Interview Questions

1. Why did you split by patient instead of by image?**  
**Answer:** Splitting by patient prevents data leakage by ensuring the same patient's images never appear in multiple sets.

2. What is data leakage? Can you explain it with an example from the RSNA dataset?**  
**Answer:** Data leakage occurs when training data indirectly appears in validation or test data, leading to overly optimistic performance.

3. Why did you use `random.seed(42)`? What happens if you don't set a seed?**  
**Answer:** Setting a seed makes the data split reproducible across multiple runs.
random.seed(42) ensures you get the exact same "random" results every time you run your code.

4. Why do the train, validation, and test ratios need to sum to 1? What would happen if they didn't?**  
**Answer:** The ratios must sum to 1 so every sample is assigned to exactly one split.

5. Why convert `labels.keys()` into a list? Why not shuffle the dictionary directly?**  
**Answer:** Dictionaries cannot be shuffled directly, so their keys are converted to a list before randomization.

6. Why do we verify that the three sets have no overlap? What would happen if one patient appeared in both train and validation?**  
**Answer:** Overlapping patients cause data leakage and produce misleading evaluation results.

**7. How would you modify this function if the dataset were highly imbalanced? (Stratified splitting.)**  
**Answer:** I would perform stratified patient-level splitting to preserve class proportions across all sets.

**8. Is random splitting always the best choice? When would you split based on time, hospital, or geographic location?**  
**Answer:** Random splitting is not always appropriate; temporal or site-based splits better reflect real-world deployment.

**9. Why did we create `split_data.py` instead of putting the function inside `labels.py`?**  
**Answer:** Splitting and label parsing are separate responsibilities, improving modularity and maintainability.

**10. If a new patient is added tomorrow, do we need to regenerate the split? How would you make the split reproducible?**  
**Answer:** Yes, the split should be regenerated using the same random seed or by saving the split IDs.

---

# 📂 File: `src/data/dicom_loader.py`

**Contribution:** Reads raw DICOM files, extracts pixel data and metadata, and provides a standardized interface for preprocessing, EDA, Streamlit, and model training.

### Interview Questions

**11. Why does `pydicom` need external plugins to read some DICOM files?**  
**Answer:** `pydicom` parses DICOM files, but compressed pixel data requires external decoders such as `pylibjpeg` or `GDCM`.

---
# 📂 File: `src/data/dataset.py`

**Contribution:** Implements the custom PyTorch Dataset that loads, preprocesses, and returns one patient sample at a time; used by `dataloader.py`, notebooks, and the training pipeline.

### Interview Questions

**12. What is a Dataset in PyTorch?**  
**Answer:** A Dataset defines how individual samples are loaded and accessed by index.

**13. Why inherit from `torch.utils.data.Dataset`?**  
**Answer:** It follows PyTorch's standard interface, making it compatible with DataLoader.

**14. Why implement `__len__()`?**  
**Answer:** It tells PyTorch the total number of samples in the dataset.

**15. Why implement `__getitem__()`?**  
**Answer:** It defines how to load and return a single sample when requested.

**16. Why didn't you load all images inside `__init__()`?**  
**Answer:** Loading all images at once would consume excessive memory, so lazy loading is used.

**17. Why does the Dataset call `load_dicom()` instead of reading DICOM directly?**  
**Answer:** It reuses a dedicated loader, keeping the code modular and maintainable.

**18. Why is `preprocess_image()` called inside the Dataset?**  
**Answer:** It ensures every image is consistently preprocessed before training.

**19. Why does `preprocess_image()` return a NumPy array instead of a Tensor?**  
**Answer:** Preprocessing is framework-independent, while tensor conversion is PyTorch-specific.

**20. Why convert to a Tensor inside the Dataset instead of inside preprocessing?**  
**Answer:** Tensor conversion is the Dataset's responsibility because it interfaces directly with PyTorch.

**21. Why do we call `unsqueeze(0)`?**  
**Answer:** It adds the channel dimension required by CNNs for grayscale images.

**22. Why does each sample return a dictionary instead of a tuple?**  
**Answer:** Dictionaries provide named fields, making the code more readable and extensible.

**23. Why do we include `patient_id` in every sample?**  
**Answer:** It enables patient tracking, debugging, and result mapping.

**24. Initially we returned `bounding_boxes`. Why was that a problem?**  
**Answer:** Bounding boxes have variable lengths, preventing the DataLoader from batching samples automatically.

**25. What happens internally every time `dataset[100]` is executed?**  
**Answer:** The Dataset loads the DICOM, preprocesses it, converts it to a tensor, and returns the sample.

**26. Isn't reading DICOM every time slow?**  
**Answer:** Yes, but lazy loading is memory-efficient and scales better for large datasets.

**27. Why is the Dataset based on patient IDs instead of image filenames?**  
**Answer:** Patient IDs support patient-level splitting and prevent data leakage.

**28. Why doesn't the Dataset perform train/test splitting itself?**  
**Answer:** Splitting is a separate responsibility handled before dataset creation.

**29. We encountered a JPEG-compressed DICOM error. Why?**  
**Answer:** Some DICOM files use compressed transfer syntaxes that require external decoding plugins.

**30. How would you make the Dataset more robust?**  
**Answer:** I would add error handling, caching, logging, and support for data augmentation.

**31. If the dataset contained two million images, what improvements would you make?**  
**Answer:** I would optimize loading with caching, multiprocessing, efficient storage formats, and faster I/O.

**32. Why not preprocess every DICOM into PNG before training?**  
**Answer:** Keeping raw DICOM preserves medical information and allows reproducible preprocessing.

**33. Suppose tomorrow the dataset includes patient age and gender. How would you extend the Dataset?**  
**Answer:** I would include the additional metadata as extra fields in the returned sample dictionary.

**34. Why separate `load_dicom()`, `preprocess_image()`, and `RSNADataset` instead of putting everything into one file?**  
**Answer:** Separation of concerns improves modularity, testing, and code reuse.

**35. (Reserved for future Dataset enhancements.)**

**36. If tomorrow you switch from PyTorch to TensorFlow, what changes?**  
**Answer:** Only the Dataset implementation changes while the preprocessing pipeline remains largely reusable.

---

# 📂 File: `src/data/dataloader.py`

**Contribution:** Wraps the Dataset into batches with configurable loading options for efficient model training; used by notebooks, `train.py`, and future validation/inference pipelines.

### Interview Questions

**37. What is the difference between a Dataset and a DataLoader?**  
**Answer:** A Dataset loads individual samples, while a DataLoader batches and iterates over them.

**38. Why create `dataloader.py` instead of creating the DataLoader inside the training script?**  
**Answer:** It separates data loading configuration from training logic for better reusability.

**39. Explain the complete AstraMed data pipeline.**  
**Answer:** DICOM → `load_dicom()` → `preprocess_image()` → `RSNADataset` → `DataLoader` → CNN.

**40. Why did we separate every stage instead of writing one large function?**  
**Answer:** Modular stages improve readability, debugging, testing, and maintenance.

**41. What exactly happens when you execute `batch = next(iter(train_loader))`?**  
**Answer:** The DataLoader fetches samples from the Dataset, batches them, and returns the first batch.

**42. Why do all images inside one batch need identical dimensions?**  
**Answer:** PyTorch stacks tensors into one batch, which requires identical shapes.

**43. We got `RuntimeError: each element in list of batch should be of equal size`. Why?**  
**Answer:** Some returned samples had different sizes or structures, preventing automatic batching.

**44. Why did we remove `bounding_boxes` from the Dataset?**  
**Answer:** Variable-length bounding boxes caused batching failures in the default DataLoader.

**45. If AstraMed later becomes an object detector, what changes?**  
**Answer:** The Dataset would return bounding boxes and the DataLoader would use a custom `collate_fn`.

**46. What is a custom `collate_fn`?**  
**Answer:** It defines how variable-sized samples are combined into a batch.

**47. Why don't we shuffle validation and test DataLoaders?**  
**Answer:** Their purpose is evaluation, so sample order should remain consistent.

**48. What does `batch_size` affect?**  
**Answer:** It determines memory usage, training speed, and gradient stability.

**49. What does `num_workers` do?**  
**Answer:** It controls how many worker processes load data in parallel.

**50. Why is the Dataset performing lazy loading?**  
**Answer:** Lazy loading reduces memory usage by loading images only when needed.

**51. Suppose AstraMed scales to one million DICOM images. What would you improve first?**  
**Answer:** I would optimize the data loading pipeline using caching, parallel loading, and faster storage.

**52. What software engineering principles can you identify in AstraMed's data pipeline?**  
**Answer:** It follows modularity, separation of concerns, reusability, and single responsibility.

**53. Which component changes if the dataset switches from DICOM to PNG?**  
**Answer:** Only the image loading component (`load_dicom`) needs to be replaced.

**54. Why is this architecture better than writing everything inside a notebook?**  
**Answer:** It creates reusable, maintainable, and production-ready code.

**55. Walk me through every function call that happens when `batch = next(iter(train_loader))`.**  
**Answer:** The DataLoader calls the Dataset, which loads the DICOM, preprocesses it, converts it to a tensor, and returns a batched output.
# 📂 File: `src/analysis/dataset_statistics.py`

**Contribution:** Computes high-level dataset statistics to understand the dataset before modeling; used by EDA notebooks, Streamlit Dataset page, and project reports.

### Interview Questions

**56. Why did you create a separate analysis module instead of putting this inside the notebook?**  
**Answer:** It makes the analysis reusable, modular, and easier to maintain.

**57. Why compute statistics before training a model?**  
**Answer:** Statistics help identify dataset characteristics and potential issues before modeling.

**58. Why is counting positive and negative cases important?**  
**Answer:** It reveals class imbalance, which affects training and evaluation.

**59. What does "Average Images per Patient" tell us?**  
**Answer:** It shows whether patients have multiple scans that could cause data leakage.

**60. Why count images with multiple bounding boxes?**  
**Answer:** It indicates cases with multiple pneumonia regions and greater localization complexity.

**61. What would happen if the number of images was greater than the number of patients?**  
**Answer:** It suggests multiple scans per patient, requiring patient-level splitting to avoid leakage.

**62. If tomorrow another hospital contributes new data, which statistics would you compare first before combining the datasets?**  
**Answer:** I would compare class distribution, patient demographics, image properties, and annotation quality.

---

# 📂 File: `src/analysis/class_distribution.py`

**Contribution:** Analyzes the balance between positive and negative classes to guide loss functions, sampling strategies, and evaluation metrics; used in EDA reports and the Streamlit dashboard.

### Interview Questions

**63. Why is checking class distribution one of the first steps in EDA?**  
**Answer:** It helps identify imbalance that can significantly impact model performance.

**64. What is class imbalance? Can you explain it using the RSNA dataset?**  
**Answer:** Class imbalance occurs when one class has significantly more samples than the other, as in the RSNA dataset.

**65. How does class imbalance affect model training?**  
**Answer:** It biases the model toward the majority class and reduces minority-class performance.

**66. Why is accuracy a poor metric on highly imbalanced datasets?**  
**Answer:** A model can achieve high accuracy by predicting only the majority class.

**67. Suppose 95% of the dataset belongs to one class. What would happen if the model always predicted the majority class?**  
**Answer:** It would achieve high accuracy but fail to detect the minority class.

**68. How can we handle class imbalance?**  
**Answer:** We can use class weights, weighted loss, focal loss, resampling, or data augmentation.

**69. Why did we create `class_distribution.py` instead of writing the calculations directly in the notebook?**  
**Answer:** It keeps analysis reusable and separates computation from visualization.

**70. Why is a bar chart generally better than a pie chart for comparing class counts?**  
**Answer:** Bar charts make numerical comparisons more accurate than pie charts.

**71. Why did we include percentages as well as counts?**  
**Answer:** Percentages provide a normalized view that is independent of dataset size.

**72. If you discover severe class imbalance, what changes later in the pipeline?**  
**Answer:** I would modify the loss function, sampling strategy, evaluation metrics, and decision threshold.

---

# 📂 File: `src/analysis/patient_analysis.py`

**Contribution:** Examines patient-level characteristics to detect leakage risks and understand dataset organization; used in EDA and data validation.

### Interview Questions

**73. Why is patient-level analysis more important than image-level analysis in medical AI?**  
**Answer:** Medical models should generalize across patients rather than individual images.

**74. Why did we count images from the filesystem instead of inferring them from the labels?**  
**Answer:** The filesystem reflects the actual available images regardless of label files.

**75. Why check for duplicate patient IDs?**  
**Answer:** Duplicate IDs may indicate data quality issues or leakage risks.

**76. What does it mean if one patient has multiple images?**  
**Answer:** The patient has multiple scans that require careful splitting and evaluation.

**77. What does it mean if one patient has multiple bounding boxes?**  
**Answer:** The patient has multiple annotated pneumonia regions.

**78. Why analyze only positive patients when studying bounding boxes?**  
**Answer:** Only positive cases contain bounding box annotations.

**79. Suppose another dataset has five images per patient. How would that affect your preprocessing and splitting strategy?**  
**Answer:** I would ensure all images from one patient remain in the same dataset split.

**80. Why is a histogram appropriate for image counts?**  
**Answer:** Histograms effectively show the frequency distribution of numerical data.

**81. Why is a bar chart appropriate for bounding box frequencies?**  
**Answer:** Bar charts clearly compare counts across discrete categories.

**82. How can patient-level analysis influence model design?**  
**Answer:** It guides leakage prevention, evaluation strategy, and patient-centric modeling.

---

# 📂 File: `src/analysis/image_analysis.py`

**Contribution:** Analyzes image properties such as dimensions, aspect ratio, and acquisition metadata to ensure consistency before training; used in EDA and preprocessing decisions.

### Interview Questions

**83. Why did we use `stop_before_pixels=True` when reading DICOM files?**  
**Answer:** It loads only metadata, making analysis much faster.

**84. Why analyze image dimensions before training?**  
**Answer:** It ensures all images can be processed consistently by the model.

**85. Why is aspect ratio important in medical imaging?**  
**Answer:** Preserving aspect ratio prevents anatomical distortion during preprocessing.

**86. What is PixelSpacing? Why does it matter?**  
**Answer:** PixelSpacing defines the physical distance between pixels and affects real-world measurements.

**87. What is the difference between image resolution and pixel spacing?**  
**Answer:** Resolution measures pixel count, while PixelSpacing measures physical pixel size.

**88. What are AP and PA chest X-rays?**  
**Answer:** AP and PA are different acquisition views that can change image appearance.

**89. Why check image size consistency?**  
**Answer:** Consistent image sizes simplify preprocessing and model training.

**90. Suppose half the dataset contains 1024×1024 images and the other half 2048×2048. How would you handle this?**  
**Answer:** I would resize them to a common input size while preserving important information.

**91. Why use histograms for width and height distributions?**  
**Answer:** Histograms show how image dimensions are distributed across the dataset.

**92. Why use boxplots in addition to histograms?**  
**Answer:** Boxplots highlight medians, spread, and outliers more clearly.

**93. If you find images with very different pixel spacing, how could that affect model performance?**  
**Answer:** Different physical scales may confuse the model unless spacing-aware preprocessing is applied.

---

# 📂 File: `src/analysis/pixel_analysis.py`

**Contribution:** Studies pixel intensity distributions to identify brightness, contrast, and image-quality variations before preprocessing; used in EDA and preprocessing design.

### Interview Questions

**94. Why analyze pixel intensities before training a CNN?**  
**Answer:** It helps identify image quality issues and guides preprocessing decisions.

**95. What does mean pixel intensity represent?**  
**Answer:** It represents the average brightness of an image.

**96. Why can standard deviation be used as a rough measure of contrast?**  
**Answer:** Higher intensity variation generally indicates greater image contrast.

**97. What is dynamic range?**  
**Answer:** Dynamic range is the difference between the minimum and maximum pixel values.

**98. Why manually inspect the darkest and brightest images?**  
**Answer:** Visual inspection helps distinguish valid cases from data quality issues.

**99. Why analyze raw DICOMs instead of normalized images?**  
**Answer:** Raw images preserve the original characteristics needed for unbiased analysis.

**100. What could cause unusually dark or bright X-rays?**  
**Answer:** Acquisition settings, scanner differences, or corrupted images can affect brightness.

**101. Why use a sample instead of processing every image during exploratory analysis?**  
**Answer:** Sampling provides representative insights while reducing computation time.

**102. Why use a fixed random seed for sampling?**  
**Answer:** A fixed seed makes the sampled subset reproducible.

**103. What is the difference between histogram and density plot?**  
**Answer:** A histogram shows counts, while a density plot estimates the underlying distribution.

**104. What does a boxplot tell you that a histogram might not?**  
**Answer:** A boxplot clearly highlights medians, quartiles, and outliers.

**105. If you discover substantial brightness variation, what preprocessing or augmentation techniques could you consider?**  
**Answer:** I would consider normalization, histogram equalization, CLAHE, or brightness augmentation after validation.

**106. Would you automatically apply histogram equalization if brightness variation exists?**  
**Answer:** No, I would first verify that it improves image quality without removing clinically important information.

**107. Why shouldn't we calculate pixel statistics on normalized images when investigating the original dataset?**  
**Answer:** Normalization changes the original pixel distribution and can hide dataset characteristics.

**108. Why did we use random sampling instead of simply taking the first 1,000 images?**  
**Answer:** Random sampling produces a more representative subset of the dataset.

**109. How would you determine whether 1,000 images are enough for this EDA?**  
**Answer:** I would verify that the sample statistics remain stable as the sample size increases.

**110. If the brightness distribution of the sample is very different from the full dataset, what would you do?**  
**Answer:** I would increase the sample size or use a more representative sampling strategy.
# 📂 File: `src/analysis/bounding_box_analysis.py`

**Contribution:** Analyzes pneumonia bounding box annotations to understand lesion size, location, and distribution; used in EDA reports, Streamlit Bounding Box page, and future object detection models.

### Interview Questions

**111. What does a bounding box represent in the RSNA dataset?**  
**Answer:** A bounding box marks the annotated region containing pneumonia.

**112. Why does one patient/image sometimes have multiple bounding boxes?**  
**Answer:** Pneumonia can appear in multiple separate regions within the same image.

**113. How do you calculate bounding box area?**  
**Answer:** Bounding box area is calculated as `width × height`.

**114. How do you calculate bounding box aspect ratio?**  
**Answer:** Aspect ratio is calculated as `width ÷ height`.

**115. Why might bounding box area be useful for medical-image analysis?**  
**Answer:** It helps quantify the size of annotated pneumonia regions.

**116. What does a bounding box center represent?**  
**Answer:** It represents the central location of the annotated lesion.

**117. Why did we calculate `center_x` and `center_y`?**  
**Answer:** They simplify spatial analysis and visualization of lesion locations.

**118. Why did we use a heatmap for bounding-box centers?**  
**Answer:** A heatmap reveals common spatial patterns of pneumonia annotations.

**119. Can the heatmap tell us which anatomical lung lobe is affected?**  
**Answer:** No, it only shows image coordinates, not anatomical structures.

**120. What does it mean if bounding boxes are very small?**  
**Answer:** Small boxes may indicate localized disease, annotation detail, or possible label noise.

**121. What does a very large bounding box indicate?**  
**Answer:** It represents a large annotated opacity but does not necessarily imply severe disease.

**122. Why might multiple bounding boxes exist in one image?**  
**Answer:** Pneumonia may affect multiple distinct regions of the lungs.

**123. Why did we create one row per bounding box in `bbox_df`?**  
**Answer:** A tabular format makes statistical analysis and visualization much easier.

**124. Why didn't we simply keep the nested bounding-box dictionaries?**  
**Answer:** Nested structures are harder to aggregate, analyze, and plot.

**125. If AstraMed becomes an object-detection model later, how would these annotations be used?**  
**Answer:** They would serve as ground-truth bounding boxes for detector training and evaluation.

**126. What problems could exist with bounding-box annotations?**  
**Answer:** Annotations may contain localization errors, missing boxes, inconsistent sizes, or noise.

**127. Why should we inspect bounding-box outliers visually?**  
**Answer:** Statistical outliers may represent genuine medical cases rather than bad data.

---

# 📂 File: `src/analysis/metadata_analysis.py`

**Contribution:** Extracts and analyzes DICOM metadata to identify acquisition patterns, dataset bias, and potential data leakage; used in EDA, Streamlit Metadata page, and preprocessing decisions.

### Interview Questions

**128. What is DICOM metadata?**  
**Answer:** DICOM metadata contains information about the patient, scan, and image acquisition.

**129. Why is metadata useful in medical imaging?**  
**Answer:** It provides contextual information that can improve data understanding and quality assessment.

**130. Why did we use `stop_before_pixels=True`?**  
**Answer:** It reads only metadata, making extraction much faster.

**131. Why do we convert PatientAge from `"030Y"` into `30`?**  
**Answer:** Numeric ages are easier to analyze statistically.

**132. What does PixelSpacing represent?**  
**Answer:** PixelSpacing specifies the real-world distance between adjacent pixels.

**133. What is the difference between pixel dimensions and physical dimensions?**  
**Answer:** Pixel dimensions count pixels, while physical dimensions measure actual size.

**134. What are AP and PA X-ray views?**  
**Answer:** AP and PA are different chest X-ray acquisition directions.

**135. Why might AP and PA images have different appearances?**  
**Answer:** Different acquisition geometries affect the appearance of anatomical structures.

**136. Why is patient sex useful to analyze even if we don't use it as a model feature?**  
**Answer:** It helps identify dataset bias and population characteristics.

**137. Why did we analyze the manufacturer?**  
**Answer:** Manufacturer differences may introduce scanner-specific shortcuts that bias the model.

**138. What is shortcut learning?**  
**Answer:** Shortcut learning occurs when a model learns dataset-specific cues instead of true clinical features.

**139. How can metadata cause data leakage?**  
**Answer:** Metadata may indirectly reveal the target label instead of actual image information.

**140. Suppose 90% of pneumonia-positive cases came from Hospital A and 90% of negative cases came from Hospital B. What's the danger?**  
**Answer:** The model may learn hospital identity instead of detecting pneumonia.

**141. Why shouldn't we automatically remove all metadata?**  
**Answer:** Some metadata is clinically useful and should be evaluated rather than discarded.

**142. Why analyze missing metadata?**  
**Answer:** Missing values may reveal dataset quality issues or hidden patterns.

**143. Why are we sampling 1,000 images instead of processing all DICOMs?**  
**Answer:** Sampling enables faster exploratory analysis while remaining representative.

**144. If manufacturer is highly correlated with pneumonia, what would you investigate?**  
**Answer:** I would investigate hospital differences, acquisition protocols, class imbalance, and potential shortcut learning.

---

# 📂 File: `src/analysis/outlier_detection.py`

**Contribution:** Detects unusual samples, corrupted DICOMs, and annotation anomalies to improve dataset quality before training; used in EDA, preprocessing validation, and quality assurance.

### Interview Questions

**145. What is an outlier?**  
**Answer:** An outlier is a data point that differs significantly from the rest of the dataset.

**146. Why shouldn't we automatically remove outliers?**  
**Answer:** Some outliers represent valid and clinically important cases.

**147. What is the IQR method?**  
**Answer:** The IQR method identifies outliers using values beyond `Q1 − 1.5×IQR` and `Q3 + 1.5×IQR`.

**148. Why did we use IQR instead of simply using mean ± standard deviation?**  
**Answer:** IQR is more robust to extreme values and does not assume a normal distribution.

**149. How did we detect unusually large bounding boxes?**  
**Answer:** We flagged bounding boxes whose areas exceeded the IQR-based outlier limits.

**150. How can you detect a corrupted DICOM?**  
**Answer:** By verifying successful reading, metadata integrity, pixel decoding, and valid image dimensions.

**151. Why does `pydicom.dcmread()` succeeding not necessarily mean the DICOM is completely valid?**  
**Answer:** Reading metadata does not guarantee that pixel data can be decoded successfully.

**152. How did we define a blank image?**  
**Answer:** We used extremely low pixel intensity variation as a screening heuristic.

**153. Why is `std < 5` not necessarily a universal threshold?**  
**Answer:** Pixel statistics vary across datasets, scanners, and imaging protocols.

**154. Why do we need an outlier gallery?**  
**Answer:** Visual inspection helps determine whether statistical outliers are valid or erroneous.

**155. What would you do if an outlier is actually a valid severe pneumonia case?**  
**Answer:** I would keep it because unusual medical cases are still valid training data.

**156. What would you do if an image is corrupted?**  
**Answer:** I would exclude or replace it while documenting the decision.

**157. Why do we sample 1,000 images for corruption and blank-image checks?**  
**Answer:** Sampling greatly reduces computation while enabling rapid exploratory analysis.

**158. How would you perform final dataset validation?**  
**Answer:** I would run integrity checks on the entire dataset and save a reproducible validation report.

**159. What's the difference between an outlier and a data error?**  
**Answer:** An outlier is unusual but can be valid, whereas a data error is incorrect or corrupted.
# 📂 File: `src/analysis/correlation_analysis.py`

**Contribution:** Computes relationships between metadata, pixel statistics, and annotations to identify useful features, dataset bias, and potential data leakage; used in EDA reports and feature engineering decisions.

### Interview Questions

**160. What is correlation?**  
**Answer:** Correlation measures the strength and direction of the relationship between two variables.

**161. What is the difference between correlation and causation?**  
**Answer:** Correlation indicates association, whereas causation means one variable directly affects the other.

**162. Why did we use Spearman correlation?**  
**Answer:** Spearman measures monotonic relationships and is robust to non-normal data and outliers.

**163. What does a correlation of +1 mean?**  
**Answer:** It indicates a perfect positive relationship between two variables.

**164. What does a correlation of 0 mean?**  
**Answer:** It indicates no linear or monotonic relationship between the variables.

**165. What does a negative correlation mean?**  
**Answer:** It means one variable decreases as the other increases.

**166. Why can't we conclude that age causes pneumonia just because they correlate?**  
**Answer:** Correlation alone cannot establish a cause-and-effect relationship.

**167. Why did we convert sex from categorical values to numeric values?**  
**Answer:** Correlation algorithms require numerical inputs.

**168. Why did we aggregate bounding boxes before correlation analysis?**  
**Answer:** Correlation requires one observation per patient/image, so multiple boxes must be aggregated.

**169. Why would bounding-box area show a strong correlation with the target?**  
**Answer:** Bounding boxes exist only for positive pneumonia cases, creating an inherent correlation.

**170. Why shouldn't bounding-box area be used as an input feature for our classification model?**  
**Answer:** It requires ground-truth annotations at inference time, causing label leakage.

**171. Why did we sample 1,000 images?**  
**Answer:** Sampling reduces computation while providing representative exploratory analysis.

**172. What happens if the sampled metadata and pixel datasets don't contain the same patient IDs?**  
**Answer:** The merged dataset contains missing values that must be handled.

**173. Why is correlation analysis useful before modeling?**  
**Answer:** It helps identify useful features, redundancy, bias, shortcuts, and leakage.

**174. Is a high correlation always useful for machine learning?**  
**Answer:** No, high correlation may result from leakage, bias, or dataset artifacts.

---

# 📂 File: `app/pages/01_Dataset.py`

**Contribution:** Serves as the dashboard homepage by presenting overall dataset statistics and EDA summaries; uses all analysis modules from `src/analysis`.

### Interview Questions

**175. Why cache the dataset using `@st.cache_data`?**  
**Answer:** Caching avoids repeated computations and speeds up the dashboard.

**176. Why separate data loading from UI rendering?**  
**Answer:** It improves modularity, readability, and maintainability.

**177. Why reuse analysis functions instead of recalculating everything inside the Streamlit page?**  
**Answer:** Reusing functions avoids code duplication and ensures consistent results.

**178. Why use `st.metric()` for KPIs?**  
**Answer:** It presents important statistics in a clean and easily readable format.

**179. Why present both tables and charts instead of only charts?**  
**Answer:** Tables provide exact values while charts reveal overall patterns.

**180. Why include an Engineering Insight section?**  
**Answer:** It connects EDA findings with practical modeling decisions.

---

# 📂 File: `app/pages/02_Image_Explorer.py`

**Contribution:** Provides an interactive DICOM image viewer with metadata and random browsing capabilities; uses `load_labels()` and `load_dicom()` from the data pipeline.

### Interview Questions

**181. Why cache the labels instead of reading the CSV every time?**  
**Answer:** Caching reduces repeated disk access and improves responsiveness.

**182. Why use `st.session_state`?**  
**Answer:** It preserves the selected image across Streamlit reruns.

**183. Why store only the selected index in `session_state` instead of the image itself?**  
**Answer:** Storing an index consumes far less memory than storing images.

**184. Why separate image loading (`load_dicom`) from UI rendering?**  
**Answer:** It keeps the data pipeline independent of the interface.

**185. Why show metadata alongside the image?**  
**Answer:** Metadata provides useful context for debugging and analysis.

**186. Why use `st.image()` instead of Matplotlib for displaying a single image?**  
**Answer:** `st.image()` is simpler, faster, and integrates naturally with Streamlit.

**187. If the dataset contained 500,000 images, what improvements would you make?**  
**Answer:** I would add search, pagination, indexing, cached metadata, and lazy loading.

---

# 📂 File: `app/pages/03_Metadata.py`

**Contribution:** Enables interactive exploration and filtering of DICOM metadata extracted from the dataset; uses `metadata_analysis.py` and Streamlit widgets.

### Interview Questions

**188. Why cache the metadata DataFrame using `@st.cache_data`?**  
**Answer:** Metadata extraction is expensive, so caching greatly improves performance.

**189. Why perform filtering on a copy (`filtered = metadata_df.copy()`) instead of modifying the original DataFrame?**  
**Answer:** It preserves the cached dataset and prevents unintended side effects.

**190. Why use sidebar filters instead of embedding them in the main page?**  
**Answer:** Sidebar filters separate controls from content and improve usability.

**191. Why support both dropdown filters and free-text patient search?**  
**Answer:** Together they support both exploration and targeted lookup.

**192. Why provide CSV export from the dashboard?**  
**Answer:** It allows filtered results to be shared and analyzed externally.

---

# 📂 File: `app/pages/04_Bounding_Box.py`

**Contribution:** Visualizes pneumonia bounding boxes on DICOM images for annotation inspection; uses `load_labels()`, `load_dicom()`, and Matplotlib overlays.

### Interview Questions

**193. Why are bounding boxes displayed only for positive patients?**  
**Answer:** Only positive cases contain pneumonia annotations.

**194. Why use `matplotlib.patches.Rectangle` instead of modifying the image array directly?**  
**Answer:** It creates non-destructive overlays while preserving the original image.

**195. Why is the bounding box represented by `(x, y, width, height)` instead of two corner points?**  
**Answer:** The RSNA dataset stores annotations in this standard format.

**196. Why doesn't AstraMed v1 use bounding boxes during training?**  
**Answer:** AstraMed v1 performs image classification, not object detection.

**197. If AstraMed evolves into an object detection model, what changes?**  
**Answer:** The dataset, DataLoader, loss functions, and training pipeline would support bounding-box targets.

---

# 📂 File: `app/pages/05_Patient_Search.py`

**Contribution:** Provides patient-level search, metadata display, image visualization, and downloadable reports; integrates `load_labels()`, `load_dicom()`, and Streamlit components.

### Interview Questions

**198. Why validate the patient ID before loading the image?**  
**Answer:** Validation prevents unnecessary file access and runtime errors.

**199. Why generate the report dynamically instead of storing reports for every patient?**  
**Answer:** Dynamic reports always reflect the latest available data.

**200. Why iterate over `metadata.items()` instead of referencing individual fields?**  
**Answer:** It automatically adapts to future metadata changes without code modification.

**201. Why provide a downloadable report?**  
**Answer:** It enables users to save and share patient information easily.

**202. How would you extend this page if AstraMed included model predictions?**  
**Answer:** I would add prediction probabilities, Grad-CAM visualizations, confidence scores, and ground-truth comparisons.

You're right. Since the previous section ended at **202**, this section should continue from **203**.

Also, there was another issue: I had accidentally shifted the numbering in `baseline_cnn.py` and `losses.py`. Here's the corrected numbering.

---

# 📂 File: `src/data/dataloader.py` *(DataLoader Verification Notebook)*

**Contribution:** Verifies the complete data pipeline (Dataset → DataLoader → Batch) before model training by checking tensors, shapes, datatypes, labels, NaNs, and class balance; used in `notebooks/16_dataloader_verification.ipynb`.

### Interview Questions

**203. What is the purpose of verifying the DataLoader before training a model?**  
**Answer:** It ensures the complete data pipeline works correctly before introducing model complexity.

**204. What does `next(iter(train_loader))` do internally?**  
**Answer:** It creates a DataLoader iterator and returns the first batch of samples.

**205. Why is the image tensor shape `(batch_size, 1, 256, 256)`?**  
**Answer:** It represents a batch of grayscale images resized to the model's input dimensions.

**206. Why is the image datatype `torch.float32`?**  
**Answer:** Neural networks perform floating-point computations using `float32`.

**207. Why are labels stored as integers instead of floats?**  
**Answer:** Labels represent discrete class indices before being converted if required by the loss function.

**208. Why do we check for NaN and Inf values before training?**  
**Answer:** They can propagate through the network and make training unstable.

**209. Why do all images in a batch need identical shapes?**  
**Answer:** Tensor batching requires every sample to have identical dimensions.

**210. Why do we visualize a batch before training?**  
**Answer:** It verifies preprocessing, labels, orientation, and overall pipeline correctness.

**211. Why do we verify class distribution in a batch?**  
**Answer:** It confirms both classes are loading correctly and reveals any imbalance.

---

# 📂 File: `src/models/baseline_cnn.py`

**Contribution:** Implements AstraMed's first CNN baseline for binary pneumonia classification; used by the training pipeline in `train.py` and future evaluation scripts.

### Interview Questions

**212. Why do we build a baseline CNN before using pretrained models?**  
**Answer:** A baseline provides a reference point for measuring future improvements.

**213. Why does the first convolution use `in_channels=1`?**  
**Answer:** Chest X-rays are grayscale images with one input channel.

**214. Why does the second convolution use `in_channels=32`?**  
**Answer:** It processes the 32 feature maps produced by the first convolution.

**215. What is the purpose of the ReLU activation?**  
**Answer:** ReLU introduces non-linearity and enables the network to learn complex features.

**216. Why do we use MaxPooling?**  
**Answer:** MaxPooling reduces spatial dimensions while preserving important features.

**217. Why do we flatten the feature maps?**  
**Answer:** Fully connected layers require one-dimensional input vectors.

**218. Why is the final layer `Linear(128, 1)`?**  
**Answer:** Binary classification requires a single output representing pneumonia probability.

**219. Why is the output shape `(batch_size, 1)`?**  
**Answer:** The model predicts one probability for each image in the batch.

**220. Why don't we use Batch Normalization or Dropout yet?**  
**Answer:** The baseline is intentionally kept simple for fair future comparisons.

**221. How would you improve this model in later experiments?**  
**Answer:** I would add BatchNorm, Dropout, deeper layers, augmentation, and pretrained models.

---

# 📂 File: `src/training/losses.py`

**Contribution:** Centralizes loss-function selection for the training pipeline, allowing different losses to be swapped without modifying the training loop; used by notebooks and `train.py`.

### Interview Questions

**222. Why did you create a separate `losses.py` module?**  
**Answer:** It separates loss configuration from training logic for easier experimentation.

**223. Why are we using `BCELoss` for the baseline model?**  
**Answer:** The model outputs sigmoid probabilities for binary classification.

**224. What is the difference between `BCELoss` and `BCEWithLogitsLoss`?**  
**Answer:** `BCELoss` expects probabilities, whereas `BCEWithLogitsLoss` expects raw logits.

**225. Why don't we use `CrossEntropyLoss`?**  
**Answer:** Our model predicts a single probability instead of multiple class logits.

---

# 📂 File: `src/training/optimizers.py`

**Contribution:** Provides a centralized interface for selecting optimizers and hyperparameters, making optimizer experimentation easy; used by notebooks and the training pipeline.

### Interview Questions

**226. Why did you create a separate `optimizers.py` module?**  
**Answer:** It separates optimizer configuration from the training loop.

**227. Why are we using Adam for the baseline?**  
**Answer:** Adam converges quickly and requires minimal hyperparameter tuning.

**228. When would you choose SGD instead of Adam?**  
**Answer:** SGD is preferred when better final generalization is desired after careful tuning.

**229. Why is the learning rate set to `1e-3`?**  
**Answer:** `1e-3` is the standard starting learning rate for Adam.

**230. Why does the optimizer receive `model.parameters()`?**  
**Answer:** It needs access to all trainable parameters to update the model.

**231. How would you extend this module for future experiments?**  
**Answer:** I would add AdamW, RMSprop, schedulers, weighted losses, and configuration-file support.

# 📂 File: `src/training/train.py`

**Contribution:** Implements the training loop for one epoch by performing forward propagation, loss computation, backpropagation, and parameter updates; used by the training notebook and future training scripts.

### Interview Questions

**232. Why do we call `model.train()` before training?**  
**Answer:** It switches the model into training mode so training-specific layers behave correctly.

**233. Why do we move images and targets to the device?**  
**Answer:** The model and data must be on the same CPU or GPU for computation.

**234. Why do we convert targets to `float()`?**  
**Answer:** `BCELoss` expects floating-point target values between 0 and 1.

**235. Why do we call `unsqueeze(1)` on the targets?**  
**Answer:** It matches the target shape with the model output shape `(batch_size, 1)`.

**236. Why do we call `optimizer.zero_grad()` before `backward()`?**  
**Answer:** It clears previously accumulated gradients before computing new ones.

**237. What happens during the forward pass?**  
**Answer:** The input images pass through the CNN to produce predictions.

**238. What does `loss.backward()` compute?**  
**Answer:** It computes gradients of the loss with respect to every trainable parameter.

**239. What does `optimizer.step()` do?**  
**Answer:** It updates the model parameters using the computed gradients.

**240. Why do we accumulate `loss.item()` instead of `loss`?**  
**Answer:** `loss.item()` stores only the scalar value without retaining the computation graph.

**241. Why do we return the average epoch loss instead of the last batch loss?**  
**Answer:** The average epoch loss better represents overall training performance across the entire dataset.