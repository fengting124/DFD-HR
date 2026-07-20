# Explicit Validation Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require an explicit validation dataset and validation split so checkpoint selection cannot silently consume test data.

**Architecture:** Keep protocol enforcement at the two existing boundaries. `training/train.py` validates the configured validation dataset list before data loaders are built, while `training/dataset/abstract_dataset.py` rejects a missing split instead of substituting another split. Existing YAML configurations and README commands declare the validation dataset explicitly.

**Tech Stack:** Python 3.10, PyTorch, PyYAML, `unittest`, Markdown, Git.

---

## File map

- `training/train.py`: validate the top-level validation dataset list and accept a CLI override.
- `training/dataset/abstract_dataset.py`: resolve only the requested split and report missing split metadata.
- `training/config/detector/dfd_hr.yaml`: declare the default validation dataset and validation frame count.
- `training/config/detector/dfd_hr_paper_aligned.yaml`: declare the paper-aligned validation dataset.
- `tests/test_train_helpers.py`: cover accepted and rejected validation dataset configuration.
- `tests/test_dataset_sampling.py`: cover accepted and rejected validation split resolution.
- `README.md`: show the explicit validation argument in training commands.

### Task 1: Lock the validation contract with failing tests

**Files:**

- Modify: `tests/test_train_helpers.py`
- Modify: `tests/test_dataset_sampling.py`

- [x] **Step 1: Add missing-validation configuration coverage**

Add this test to `TrainHelpersTests`:

```python
def test_resolve_eval_loader_names_rejects_missing_validation_dataset(self):
    with self.assertRaisesRegex(ValueError, "validation_dataset"):
        train.resolve_eval_loader_names({"validation_dataset": []})
```

- [x] **Step 2: Replace the split-fallback test with strict split coverage**

Replace `test_validation_mode_falls_back_to_test_split` with:

```python
def test_validation_mode_rejects_missing_validation_split(self):
    split_dict = {
        "test": {
            "c23": {
                "vid0": {
                    "label": "FF-real",
                    "frames": ["a.png", "b.png"],
                }
            }
        }
    }

    with self.assertRaisesRegex(ValueError, "explicit 'val' split"):
        DeepfakeAbstractBaseDataset._resolve_mode_split(
            split_dict=split_dict,
            mode="val",
            compression="c23",
            dataset_name="FaceForensics++",
            cp=None,
        )
```

Add a positive test using an explicit `val` split:

```python
def test_validation_mode_uses_explicit_validation_split(self):
    split_dict = {
        "val": {
            "c23": {
                "vid0": {
                    "label": "FF-real",
                    "frames": ["a.png", "b.png"],
                }
            }
        }
    }

    resolved = DeepfakeAbstractBaseDataset._resolve_mode_split(
        split_dict=split_dict,
        mode="val",
        compression="c23",
        dataset_name="FaceForensics++",
        cp=None,
    )

    self.assertIn("vid0", resolved)
```

- [x] **Step 3: Run the focused tests and verify the new assertions fail**

Run:

```bash
"${DFDHR_PYTHON}" -m unittest \
  tests.test_train_helpers.TrainHelpersTests.test_resolve_eval_loader_names_rejects_missing_validation_dataset \
  tests.test_dataset_sampling.DatasetSamplingTests.test_validation_mode_rejects_missing_validation_split \
  tests.test_dataset_sampling.DatasetSamplingTests.test_validation_mode_uses_explicit_validation_split \
  -v
```

Expected: the two rejection tests fail because empty validation configuration and test-split fallback are still accepted; the explicit validation test passes.

- [x] **Step 4: Commit the regression tests**

```bash
git add tests/test_train_helpers.py tests/test_dataset_sampling.py
git commit -m "test(data): require explicit validation inputs"
```

### Task 2: Enforce the validation contract

**Files:**

- Modify: `training/train.py`
- Modify: `training/dataset/abstract_dataset.py`
- Test: `tests/test_train_helpers.py`
- Test: `tests/test_dataset_sampling.py`

- [x] **Step 1: Reject an empty validation dataset list**

Replace `resolve_eval_loader_names` with:

```python
def resolve_eval_loader_names(config):
    validation_dataset = config.get('validation_dataset')
    if not validation_dataset:
        raise ValueError(
            'validation_dataset must explicitly name at least one dataset; '
            'test_dataset cannot be used for checkpoint selection.'
        )
    return validation_dataset
```

- [x] **Step 2: Reject a missing requested split**

Replace the fallback at the start of `_resolve_mode_split` with:

```python
if mode not in split_dict:
    raise ValueError(
        f"Dataset {dataset_name!r} has no explicit {mode!r} split; "
        "a test split cannot substitute for validation."
    )
sub_dataset_info = split_dict[mode]
```

Keep the existing compression selection unchanged.

- [x] **Step 3: Run focused tests and verify they pass**

Run:

```bash
"${DFDHR_PYTHON}" -m unittest \
  tests.test_train_helpers \
  tests.test_dataset_sampling \
  -v
```

Expected: all tests in both modules pass.

- [x] **Step 4: Commit the minimal implementation**

```bash
git add training/train.py training/dataset/abstract_dataset.py
git commit -m "fix(data): reject implicit validation fallback"
```

### Task 3: Make existing training entry points explicit

**Files:**

- Modify: `training/train.py`
- Modify: `training/config/detector/dfd_hr.yaml`
- Modify: `training/config/detector/dfd_hr_paper_aligned.yaml`
- Modify: `README.md`
- Test: `tests/test_train_helpers.py`

- [x] **Step 1: Add a validation dataset CLI override**

Add beside the existing dataset arguments:

```python
parser.add_argument("--validation_dataset", nargs="+")
```

Apply the override before data loaders are created:

```python
if args.validation_dataset:
    config['validation_dataset'] = args.validation_dataset
```

- [x] **Step 2: Declare validation in both detector configurations**

Set:

```yaml
validation_dataset: [FaceForensics++]
```

In `training/config/detector/dfd_hr.yaml`, also change the frame count to:

```yaml
frame_num: {'train': 8, 'val': 32, 'test': 8}
```

Keep the paper-aligned configuration's existing validation frame count unchanged.

- [x] **Step 3: Update both README training commands**

Add this line between `--train_dataset` and `--test_dataset`:

```bash
    --validation_dataset FaceForensics++ \
```

- [x] **Step 4: Verify both YAML files parse and expose validation settings**

Run:

```bash
"${DFDHR_PYTHON}" - <<'PY'
from pathlib import Path
import yaml

for path in (
    Path("training/config/detector/dfd_hr.yaml"),
    Path("training/config/detector/dfd_hr_paper_aligned.yaml"),
):
    config = yaml.safe_load(path.read_text())
    assert config["validation_dataset"] == ["FaceForensics++"]
    assert "val" in config["frame_num"]
    print(path, config["validation_dataset"], config["frame_num"]["val"])
PY
```

Expected: both paths print with `['FaceForensics++']`; the default and paper-aligned validation frame counts print as `32`.

- [x] **Step 5: Commit configuration and documentation changes**

```bash
git add \
  training/train.py \
  training/config/detector/dfd_hr.yaml \
  training/config/detector/dfd_hr_paper_aligned.yaml \
  README.md
git commit -m "docs(train): declare validation dataset"
```

### Task 4: Verify the complete change

**Files:**

- Verify: `training/train.py`
- Verify: `training/dataset/abstract_dataset.py`
- Verify: `training/config/detector/dfd_hr.yaml`
- Verify: `training/config/detector/dfd_hr_paper_aligned.yaml`
- Verify: `tests/`
- Verify: `README.md`

- [x] **Step 1: Run the complete CPU test suite**

Run:

```bash
"${DFDHR_PYTHON}" -m unittest discover -s tests -v
```

Expected: all discovered tests pass.

- [x] **Step 2: Check syntax and whitespace**

Run:

```bash
"${DFDHR_PYTHON}" -m py_compile \
  training/train.py \
  training/dataset/abstract_dataset.py
git diff --check origin/main...HEAD
```

Expected: both commands exit with status zero.

- [x] **Step 3: Confirm test data is never used as an implicit validation source**

Run:

```bash
rg -n "mode_key = 'test'|falls_back_to_test|validation_dataset: \[\]" \
  training tests README.md
```

Expected: no matches.

- [x] **Step 4: Review the final branch diff**

Run:

```bash
git status --short --branch
git diff --stat origin/main...HEAD
git log --oneline origin/main..HEAD
```

Expected: the branch contains only the plan, validation tests, minimal validation enforcement, configuration declarations, and README updates.
