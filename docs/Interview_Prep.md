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

