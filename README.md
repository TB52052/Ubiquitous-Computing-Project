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

## 2. workflow

For someone using this project for the first time, the shortest correct workflow is:

1. put raw recordings into `falling_data/`
2. run `organize_data.py`
3. check the resulting `organized_data/` structure
4. optionally run `split_running.py` if one class has too few sessions
5. run `features_model.py` for fall detection
6. run `multiclass_analysis.py` for detailed activity classification
7. review the generated PNG result files
