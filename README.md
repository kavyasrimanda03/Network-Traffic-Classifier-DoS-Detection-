# 🛡️ Network Traffic Classifier — DoS Attack Detection

A machine learning pipeline that classifies network traffic as **benign or malicious** (DoS attacks), achieving **~99.99% accuracy** on the CIC-IDS2017 dataset. Built with Random Forest and SMOTE-based class balancing, with an interactive Tkinter GUI for live packet classification.

---

## 📌 Problem Statement

Denial-of-Service (DoS) attacks are among the most disruptive threats to network infrastructure. Detecting them in real time requires distinguishing subtle patterns in packet-level flow features from normal traffic. This project addresses that challenge using supervised machine learning on real-world network flow data.

---

## 🔍 Attack Types Detected

| Class | Description |
|---|---|
| **Benign** | Normal network traffic |
| **DoS attacks-GoldenEye** | HTTP-based DoS using persistent connections |
| **DoS attacks-Slowloris** | Slow HTTP attack exhausting server connections |

---

## ⚙️ Methodology

### Data Pipeline
- **Dataset:** CIC-IDS2017 (CICIDS network intrusion dataset)
- Loaded up to 1M+ records from `dataset2.csv`
- Dropped null values and duplicates
- Removed `Timestamp` column; encoded `Label` with `LabelEncoder`
- Replaced `inf` values with `NaN` and filtered invalid rows
- Applied `StandardScaler` for feature normalization

### Handling Class Imbalance
- Used **SMOTE (Synthetic Minority Oversampling Technique)** on the training split to synthetically balance minority attack classes before training

### Model
- **Algorithm:** Random Forest Classifier (`n_estimators=100`)
- **Train/Test Split:** 80/20 with stratification
- **Advanced version (`gui1.py`):** Includes `GridSearchCV` hyperparameter tuning and `class_weight='balanced'`

### Feature Importance
- Top 6 features identified via `feature_importances_` are used in the GUI:
  - Flow Duration, Total Fwd Packets, Total Backward Packets
  - Flow Bytes/s, Flow Packets/s, Fwd Packet Length Mean

---

## 📊 Results

### Classification Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Benign | 1.00 | 1.00 | 1.00 |
| DoS attacks-GoldenEye | 1.00 | 1.00 | 1.00 |
| DoS attacks-Slowloris | 1.00 | 0.9985 | 0.9993 |
| **Overall Accuracy** | | **~99.99%** | |

### Confusion Matrix Highlights
- **Benign:** 9429 / 9431 correctly classified (2 misclassified)
- **DoS-GoldenEye:** 8290 / 8291 correctly classified (1 misclassified)
- **DoS-Slowloris:** 2054 / 2057 correctly classified (3 misclassified)

> Results from the full-dataset run (`gui1.py`) on 1M+ records show 0 misclassifications for Benign and GoldenEye traffic.

---

## 🖥️ GUI — Live Packet Classifier

An interactive Tkinter GUI allows manual entry of network flow features for real-time classification.

**Input fields:**
- Flow Duration
- Total Fwd / Backward Packets
- Flow Bytes/s and Packets/s
- Fwd Packet Length Mean

The model predicts whether the packet is **Benign**, **DoS-GoldenEye**, or **DoS-Slowloris**.

---

## 📁 File Structure

```
├── class.py                  # Core training script with feature importance analysis
├── gui.py                    # GUI classifier (100k row dataset version)
├── gui1.py                   # GUI classifier with GridSearchCV tuning (full dataset)
├── network_classifier.py     # Standalone classifier with hardcoded GUI inputs
├── Confusion_matrix.png      # Confusion matrix heatmap (100k run)
├── Figure_1.png              # Confusion matrix (full dataset run)
├── Figure_2.png              # Confusion matrix (brute-force dataset run)
├── classification_report.png # Precision/Recall/F1 bar chart per class
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas / NumPy | Data loading and preprocessing |
| Scikit-learn | Model training, evaluation, scaling |
| imbalanced-learn | SMOTE oversampling |
| Matplotlib / Seaborn | Visualizations |
| Tkinter | Interactive GUI |

---

## 🚀 How to Run

**1. Install dependencies:**
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

**2. Place your dataset:**
Ensure `dataset2.csv` (CIC-IDS2017 format) is in the same directory.

**3. Run the core classifier:**
```bash
python class.py
```

**4. Launch the GUI:**
```bash
python gui.py
```

---

## 🔗 Dataset

This project uses the **CICIDS2017 dataset** from the Canadian Institute for Cybersecurity.
- [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

---

## 👩‍💻 Author

**Manda Kavya Sri Reddy**
M.S. Cybersecurity — University of Delaware
[LinkedIn](https://www.linkedin.com/in/kavyasrireddymanda/) | [GitHub](https://github.com/kavyasrimanda03)
