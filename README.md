# Fall Detection Project

This repository contains the Python-side code for a smartphone-based fall detection project.

The code is mainly used for three things:
- organizing raw phone sensor recordings into a clean dataset
- training and evaluating a **binary fall / non-fall model**
- training and evaluating a **multiclass activity recognition model**

It does **not** contain the full Android app interface or alert workflow.

---

## 1. Project structure

A practical working structure is:

```text
project_root/
├─ falling_data/                  # raw sensor recordings
├─ organized_data/                # organized dataset used for modelling
├─ organized_data_split/          # optional split sessions for running
├─ model_output/                  # binary fall detection outputs
├─ model_output_multiclass/       # multiclass recognition outputs
├─ organize_data.py
├─ split_running.py
├─ features_model.py
├─ multiclass_analysis.py
└─ README.md
```

---

## 2. Organized dataset structure

After running the organizer, the dataset should look like this:

```text
organized_data/
├─ fall/
│  ├─ trip_fall/
│  │  └─ complete/
│  │     └─ session_name/
│  │        ├─ accel.txt
│  │        ├─ gyro.txt
│  │        └─ pressure.txt
│  ├─ push_fall/
│  └─ ...
└─ non_fall/
   ├─ walking/
   ├─ running/
   ├─ sitting_down/
   ├─ phone_drop/
   └─ ...
```

Each session folder is expected to contain sensor files such as:
- `accel.txt`
- `gyro.txt`
- `pressure.txt` (optional in some workflows)

The code uses:
- a **binary label**: `fall` or `non_fall`
- a **raw activity label**: such as `walking`, `running`, `trip_fall`, `phone_drop`

---

## 3. Recommended execution order

### Step 1 - Organize raw data

Use `organize_data.py` first.

Example:

```bash
python organize_data.py --input_dir ./falling_data --output_dir ./organized_data --mode copy
```

Purpose:
- scan raw sensor files
- group files into sessions
- infer labels from folder names / file names
- create a clean dataset structure for later modelling

---

### Step 2 - Optionally split a long running session

Use `split_running.py` only when one category, especially `running`, has too few sessions but one recording is very long.

This script is edited directly at the top of the file before running.

Then run:

```bash
python split_running.py
```

Purpose:
- cut one very long recording into several shorter sessions
- increase the number of usable sessions for that class
- help multiclass training include that label more reliably

---

### Step 3 - Run binary fall detection

Use `features_model.py`.

Example:

```bash
python features_model.py --data_dir ./organized_data --output_dir ./model_output
```

Purpose:
- build feature windows from organized sessions
- train and compare binary classifiers
- evaluate fall vs non-fall performance
- apply a fusion rule based on model output and event thresholds

---

### Step 4 - Run multiclass activity analysis

Use `multiclass_analysis.py`.

Example:

```bash
python multiclass_analysis.py --data_dir ./organized_data --output_dir ./model_output_multiclass --min_sessions_per_label 1
```

Purpose:
- classify detailed activity labels such as walking, running, phone drop, and different fall types
- compare multiple classical machine learning models
- measure how well the dataset supports fine-grained activity recognition

---

## 4. Core points other people should understand

### Session-based design
The code does not train directly on whole recordings. It first organizes data into **sessions**, then converts each session into multiple sliding windows for feature extraction and model training.

### Two modelling tasks
This repository supports two separate but related tasks:
- **binary detection**: fall vs non-fall
- **multiclass recognition**: detailed activity category classification

### Fusion decision for fall detection
The binary fall pipeline is not just a plain classifier. It combines:
- model probability
- impact-related thresholds
- post-event stillness thresholds

This is used to reduce false positives from strong but normal activities.

### Grouped evaluation
The code uses session-aware validation where possible, so windows from the same original session are not freely mixed between training and testing. This makes the reported results more realistic.

---

## 5. Output files

### Output from `organize_data.py`
Main result:
- a clean `organized_data/` directory ready for modelling

Optional extra output:
- manifest CSV files if enabled

### Output from `split_running.py`
Main result:
- several new running session folders in `organized_data_split/`

Also produced:
- `running_split_manifest.csv`

### Output from `features_model.py`
Main result:
- figures saved into `model_output/`

Typical files:
- `session_counts.png`
- `binary_model_comparison.png`
- `*_confusion_matrix.png`
- `fusion_confusion_matrix.png`

### Output from `multiclass_analysis.py`
Main result:
- figures saved into `model_output_multiclass/`

Typical files:
- `multiclass_session_counts.png`
- `multiclass_model_comparison.png`
- `*_multiclass_confusion_matrix.png`

---

## 6. Minimal workflow summary

For someone using this project for the first time, the shortest correct workflow is:

1. put raw recordings into `falling_data/`
2. run `organize_data.py`
3. check the resulting `organized_data/` structure
4. optionally run `split_running.py` if one class has too few sessions
5. run `features_model.py` for fall detection
6. run `multiclass_analysis.py` for detailed activity classification
7. review the generated PNG result files
