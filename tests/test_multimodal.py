"""
Sprint 8 unit tests. Run:  python tests/test_multimodal.py

Everything here uses SYNTHETIC inputs that exist only to exercise the code (marked
'synthetic'); they are never used for training and are not clinical data. Tests that need
local data/artifacts (cache, clinical table, baseline checkpoint) are reported as SKIP
when those are absent.
"""
import hashlib
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multimodal.clinical_data import (
    ClinicalFeatureEncoder, missingness_report, parse_age, validate_join,
)
from src.multimodal.dataset import MultimodalDataset, collate_batch
from src.multimodal.model import (
    MultimodalPneumoniaNet, load_multimodal_checkpoint, save_multimodal_checkpoint,
)
from src.multimodal.text import ClinicalTextEncoder, SimpleTokenizer
from src.utils.config import OUTPUT_DIR

BASELINE = (OUTPUT_DIR / "experiments" / "resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42"
            / "checkpoints" / "best_model.pth")
CLINICAL_CSV = OUTPUT_DIR / "multimodal" / "clinical_metadata.csv"
SKIPPED = []


def synthetic_table(n=8):
    return pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "age": [20.0, 40.0, 60.0, math.nan, 30.0, 50.0, 70.0, 110.0][:n],
        "sex": ["M", "F", "M", None, "F", "M", "F", "M"][:n],
        "view_position": ["AP", "PA", "AP", "PA", None, "AP", "PA", "AP"][:n],
    })


class FakeStore:
    def get(self, patient_id):
        generator = torch.Generator().manual_seed(abs(hash(patient_id)) % 1000)
        return torch.rand(1, 224, 224, generator=generator)


def synthetic_labels(table):
    return {p: {"target": i % 2, "bounding_boxes": []} for i, p in enumerate(table.patient_id)}


def small_batch(model_kwargs=None, with_struct=True, mask=None, n=3):
    batch = {
        "image": torch.rand(n, 1, 224, 224),
        "clinical_features": torch.randn(n, 5),
        "modality_mask": torch.tensor(mask if mask is not None else [[1.0, 1.0, 0.0, 0.0]] * n),
    }
    return batch


# ------------------------------------------------------------------ clinical encoder & join
def test_age_parsing_and_malformed_values():
    assert parse_age("058Y") == 58.0
    assert math.isnan(parse_age(None)) and math.isnan(parse_age("")) and math.isnan(parse_age("abc"))
    assert math.isnan(parse_age("999Y"))            # implausible
    assert abs(parse_age("006M") - 0.5) < 1e-9


def test_encoder_fits_on_training_rows_only():
    table = synthetic_table()
    train, other = table.iloc[:4], table.iloc[4:]
    encoder = ClinicalFeatureEncoder().fit(train)
    expected_mean = train["age"].dropna().mean()
    assert abs(encoder.age_mean - expected_mean) < 1e-9
    # transforming other rows must not change the fitted statistics
    encoder.transform(other)
    assert abs(encoder.age_mean - expected_mean) < 1e-9
    refit = ClinicalFeatureEncoder().fit(table)
    assert refit.age_mean != encoder.age_mean


def test_missing_values_use_indicators_not_invented_values():
    table = synthetic_table()
    encoder = ClinicalFeatureEncoder().fit(table)
    features = encoder.transform(table)
    names = encoder.feature_names
    row = features[3]                                  # age and sex both missing
    assert row[names.index("age_missing")] == 1.0 and row[names.index("age_z")] == 0.0
    assert row[names.index("sex_unknown")] == 1.0
    assert row[names.index("sex_M")] == 0.0 and row[names.index("sex_F")] == 0.0
    assert encoder.structured_mask(table)[3] == 0.0   # nothing genuinely present
    assert encoder.structured_mask(table)[0] == 1.0
    assert np.isfinite(features).all()


def test_unfitted_encoder_refuses_to_transform():
    try:
        ClinicalFeatureEncoder().transform(synthetic_table())
    except RuntimeError:
        return
    raise AssertionError("unfitted encoder must raise")


def test_encoder_is_deterministic_and_roundtrips():
    table = synthetic_table()
    a = ClinicalFeatureEncoder().fit(table)
    b = ClinicalFeatureEncoder.from_dict(a.to_dict())
    assert np.array_equal(a.transform(table), b.transform(table))


def test_view_position_is_opt_in():
    table = synthetic_table()
    assert ClinicalFeatureEncoder(False).fit(table).dim == 5
    assert ClinicalFeatureEncoder(True).fit(table).dim == 8


def test_join_rejects_duplicates_orphans_and_missing_columns():
    table = synthetic_table()
    ids = list(table.patient_id)
    validate_join(table, ids)
    for bad, ids_arg in (
        (pd.concat([table, table.iloc[[0]]]), ids),                 # duplicate id
        (table.iloc[:-1], ids),                                      # labelled patient without row
        (table, ids[:-1]),                                           # row without label
        (table.drop(columns=["age"]), ids),                          # malformed table
    ):
        try:
            validate_join(bad, ids_arg)
        except ValueError:
            continue
        raise AssertionError("invalid join was accepted")
    assert missingness_report(table)["age_missing"] == 1


# ------------------------------------------------------------------ dataset
def test_dataset_schema_alignment_and_split_guards():
    table = synthetic_table()
    labels = synthetic_labels(table)
    encoder = ClinicalFeatureEncoder().fit(table)
    dataset = MultimodalDataset(list(table.patient_id[:6]), labels, table, encoder, FakeStore(),
                                forbidden_ids=["p6", "p7"])
    sample = dataset[3]
    assert sample["patient_id"] == "p3" and sample["target"] == labels["p3"]["target"]
    assert math.isnan(sample["age"]) and sample["sex"] == ""
    assert sample["symptoms_text"] == "" and sample["clinical_history_text"] == ""
    assert tuple(sample["image"].shape) == (1, 224, 224)
    assert tuple(sample["clinical_features"].shape) == (5,)
    assert sample["modality_mask"].tolist() == [1.0, 0.0, 0.0, 0.0]   # structured absent for p3
    assert dataset[0]["modality_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]
    batch = collate_batch([dataset[i] for i in range(3)])
    assert tuple(batch["image"].shape) == (3, 1, 224, 224) and tuple(batch["modality_mask"].shape) == (3, 4)
    for ids, kind in ((["p0", "p0"], "duplicate"), (["p6"], "held-out overlap")):
        try:
            MultimodalDataset(ids, labels, table, encoder, FakeStore(), forbidden_ids=["p6", "p7"])
        except ValueError:
            continue
        raise AssertionError(f"{kind} must be rejected")


# ------------------------------------------------------------------ text (synthetic strings only)
def test_text_tokenizer_masks_and_truncation():
    tokenizer = SimpleTokenizer(max_len=4).fit(["synthetic cough fever", "synthetic fever"])
    ids, mask = tokenizer.encode("synthetic cough fever unknownword extra")
    assert len(ids) == 4 and mask == [1.0, 1.0, 1.0, 1.0]       # truncated to max_len
    ids, mask = tokenizer.encode("")
    assert sum(mask) == 0 and set(ids) == {0}                    # empty -> fully masked
    assert tokenizer.encode("zzz")[0][0] == 1                    # unknown token id


def test_text_encoder_is_zero_for_empty_text():
    tokenizer = SimpleTokenizer(max_len=6).fit(["synthetic cough", "synthetic fever"])
    encoder = ClinicalTextEncoder(tokenizer.vocab_size, out_dim=16)
    ids, mask = tokenizer.encode_batch(["synthetic cough", ""])
    out = encoder(ids, mask)
    assert out.shape == (2, 16) and out[1].abs().sum() == 0 and out[0].abs().sum() > 0


# ------------------------------------------------------------------ model
def test_forward_shapes_for_each_modality_combination():
    torch.manual_seed(0)
    for kwargs in (dict(), dict(use_structured=True, struct_dim=5),
                   dict(use_structured=True, struct_dim=5, use_text=True, vocab_size=20)):
        model = MultimodalPneumoniaNet(**kwargs).eval()
        batch = small_batch()
        if kwargs.get("use_text"):
            tokenizer = SimpleTokenizer(max_len=6).fit(["synthetic cough fever"])
            ids, mask = tokenizer.encode_batch(["synthetic cough", "", "synthetic fever"])
            batch.update(symptom_ids=ids, symptom_mask=mask, history_ids=ids, history_mask=mask * 0)
        with torch.no_grad():
            logits = model(batch)
        assert tuple(logits.shape) == (3, 1) and torch.isfinite(logits).all()


def test_missing_structured_modality_cannot_leak_clinical_values():
    torch.manual_seed(0)
    model = MultimodalPneumoniaNet(use_structured=True, struct_dim=5).eval()
    image = torch.rand(2, 1, 224, 224)
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
    with torch.no_grad():
        a = model({"image": image, "clinical_features": torch.zeros(2, 5), "modality_mask": mask})
        b = model({"image": image, "clinical_features": torch.randn(2, 5) * 50, "modality_mask": mask})
    assert torch.allclose(a, b)                       # masked-out features have no effect
    mixed = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        out = model({"image": image, "clinical_features": torch.randn(2, 5), "modality_mask": mixed})
    assert out.shape == (2, 1)                        # mixed missingness within a batch


def test_invalid_inputs_are_rejected():
    model = MultimodalPneumoniaNet(use_structured=True, struct_dim=5)
    good = small_batch(n=2)
    for key, value in (("image", torch.rand(2, 3, 224, 224)),
                       ("modality_mask", torch.ones(2, 3)),
                       ("clinical_features", torch.randn(2, 4))):
        bad = dict(good)
        bad[key] = value
        try:
            model(bad)
        except ValueError:
            continue
        raise AssertionError(f"bad {key} accepted")


def test_backward_pass_loss_and_gradient_flow():
    torch.manual_seed(0)
    model = MultimodalPneumoniaNet(use_structured=True, struct_dim=5)
    batch = small_batch(n=2)
    loss = torch.nn.BCEWithLogitsLoss()(model(batch).reshape(-1), torch.tensor([0.0, 1.0]))
    loss.backward()
    assert torch.isfinite(loss)
    for name in ("structured_encoder.0.weight", "fusion_head.1.weight", "image_projection.0.weight"):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def test_frozen_encoder_receives_no_gradient():
    model = MultimodalPneumoniaNet()
    model.set_image_encoder_trainable(False)
    loss = model(small_batch(n=2)).sum()
    loss.backward()
    assert all(p.grad is None for p in model.image_encoder.parameters())
    assert model.fusion_head[1].weight.grad is not None


def test_fixed_seed_gives_identical_initialisation_and_output():
    outputs = []
    for _ in range(2):
        torch.manual_seed(7)
        model = MultimodalPneumoniaNet(use_structured=True, struct_dim=5).eval()
        torch.manual_seed(8)
        with torch.no_grad():
            outputs.append(model(small_batch(n=2)))
    assert torch.equal(outputs[0], outputs[1])


def test_predict_proba_range_and_mode_restoration():
    model = MultimodalPneumoniaNet()
    model.train()
    probabilities = model.predict_proba(small_batch(n=2))
    assert probabilities.shape == (2,) and ((probabilities >= 0) & (probabilities <= 1)).all()
    assert model.training


def test_checkpoint_roundtrip_and_overwrite_protection():
    torch.manual_seed(0)
    model = MultimodalPneumoniaNet(use_structured=True, struct_dim=5).eval()
    batch = small_batch(n=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run" / "best_model.pth"
        save_multimodal_checkpoint(model, None, 3, 0.5, path, extra={"note": "synthetic"})
        loaded, checkpoint = load_multimodal_checkpoint(path)
        loaded.eval()
        with torch.no_grad():
            assert torch.equal(model(batch), loaded(batch))
        assert checkpoint["epoch"] == 3
        try:
            save_multimodal_checkpoint(model, None, 4, 0.6, path, extra={})
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing checkpoint was overwritten")
        # an image-only baseline checkpoint must not be loadable as a multimodal one
        torch.save({"model_state_dict": {}, "epoch": 1}, Path(tmp) / "baseline_like.pth")
        try:
            load_multimodal_checkpoint(Path(tmp) / "baseline_like.pth")
        except ValueError:
            return
        raise AssertionError("non-multimodal checkpoint accepted")


# ------------------------------------------------------------------ real artifacts (skip if absent)
def test_baseline_weights_load_without_touching_the_checkpoint():
    if not BASELINE.exists():
        SKIPPED.append("baseline checkpoint not found")
        return
    before = hashlib.sha256(BASELINE.read_bytes()).hexdigest().upper()
    model = MultimodalPneumoniaNet()
    report = model.load_image_encoder_from_baseline(BASELINE)
    assert report["tensors_loaded"] > 300 and report["tensors_not_loaded"] == ["fc.bias", "fc.weight"]
    saved = torch.load(BASELINE, map_location="cpu", weights_only=True)["model_state_dict"]
    assert torch.equal(model.image_encoder.conv1.weight, saved["conv1.weight"])
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest().upper() == before
    assert before == "77A82AAC41C7EE5D4D7B5A55964008E7DA7DFB999AAB602BC14DE045FA53ED26"


def test_real_clinical_table_joins_one_to_one_with_labels():
    if not CLINICAL_CSV.exists():
        SKIPPED.append("clinical_metadata.csv not found (run scripts/train_multimodal.py once)")
        return
    from src.data.labels import load_labels
    from src.utils.config import TRAIN_LABELS
    labels, _ = load_labels(TRAIN_LABELS)
    table = pd.read_csv(CLINICAL_CSV)
    validate_join(table, labels.keys())
    assert len(table) == 26684


def test_real_split_is_disjoint_and_encoder_uses_train_only():
    if not CLINICAL_CSV.exists():
        SKIPPED.append("clinical_metadata.csv not found")
        return
    from src.data.labels import load_labels
    from src.data.split_data import split_patients
    from src.utils.config import TRAIN_LABELS
    labels, _ = load_labels(TRAIN_LABELS)
    train, val, test = split_patients(labels)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    table = pd.read_csv(CLINICAL_CSV).set_index("patient_id")
    encoder = ClinicalFeatureEncoder().fit(table.loc[train].reset_index())
    assert abs(encoder.age_mean - table.loc[train, "age"].mean()) < 1e-9
    assert abs(encoder.age_mean - table["age"].mean()) > 1e-6     # not the all-patients mean


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS:", test.__name__)
    print(f"{len(tests)}/{len(tests)} executed" + (f"; skipped parts: {SKIPPED}" if SKIPPED else ""))
