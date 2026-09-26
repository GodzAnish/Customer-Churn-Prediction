# Complete Setup Guide - Customer Churn Prediction System

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Running the Project](#running-the-project)
4. [Troubleshooting](#troubleshooting)
5. [Project Workflow](#project-workflow)

---

## System Requirements

### Software Requirements
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **VS Code**: Latest version (recommended) or any Python IDE
- **Git**: Optional, for version control
- **Operating System**: Windows, macOS, or Linux

### Hardware Requirements
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: 500MB free space
- **Processor**: Any modern CPU

---

## Installation Steps

### Step 1: Verify Python Installation

Open Command Prompt or Terminal and check:

```bash
python --version
```

Should show: Python 3.8 or higher

If not installed, download from: https://www.python.org/downloads/

**Important**: During installation, check "Add Python to PATH"

### Step 2: Extract Project Files

Extract the `Customer-Churn-Prediction` folder to:
- **Windows**: `C:\Users\YourName\Desktop\`
- **macOS/Linux**: `~/Desktop/`

### Step 3: Open in VS Code

1. Open Visual Studio Code
2. Click **File** → **Open Folder**
3. Navigate to and select `Customer-Churn-Prediction`
4. Click **Select Folder**

### Step 4: Install Python Extension (First Time)

1. In VS Code, press `Ctrl+Shift+X` (Extensions)
2. Search for "Python"
3. Install **Python** by Microsoft
4. Install **Pylance** by Microsoft
5. Restart VS Code

### Step 5: Create Virtual Environment

Open VS Code terminal (`` Ctrl+` ``) and run:

```bash
python -m venv venv
```

Wait 30-60 seconds for completion.

### Step 6: Activate Virtual Environment

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

If PowerShell gives an error:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activating again.

**macOS/Linux:**
```bash
source venv/bin/activate
```

**✅ Success Check**: You should see `(venv)` at the start of your terminal prompt

### Step 7: Install Dependencies

With virtual environment activated:

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

This installs:
- numpy (numerical computing)
- pandas (data manipulation)
- matplotlib (plotting)
- seaborn (statistical visualization)
- scikit-learn (machine learning)
- jupyter (notebook support)

**Wait**: Installation takes 2-5 minutes depending on internet speed.

### Step 8: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('✅ All packages installed!')"
```

Should print: `✅ All packages installed!`

---

## Running the Project

### Method 1: Complete Pipeline (Recommended)

Run both scripts in sequence:

```bash
# Step 1: Generate the dataset
python generate_dataset.py

# Step 2: Train and evaluate models
python churn_prediction.py
```

### Method 2: Using VS Code Debugger

1. Open `churn_prediction.py`
2. Press `F5`
3. Select "Python: Churn Prediction"
4. Watch the output in the terminal

### Method 3: Step-by-Step Execution

#### Generate Dataset Only
```bash
python generate_dataset.py
```

Output:
- Creates `data/customer_churn_data.csv`
- Prints dataset statistics

#### Run Prediction Only
```bash
python churn_prediction.py
```

Output:
- EDA statistics
- Model training progress
- Evaluation metrics
- Saved models in `models/` folder

### Method 4: Jupyter Notebook (Interactive)

```bash
jupyter notebook
```

Then open: `notebooks/EDA_Customer_Churn.ipynb`

---

## Project Workflow

### What Happens When You Run the Project

```
1. Load Data
   ↓
2. Exploratory Data Analysis (EDA)
   - Dataset statistics
   - Churn distribution
   - Missing values check
   ↓
3. Feature Engineering
   - Encode categorical variables
   - Create new features
   - Scale numerical features
   ↓
4. Train-Test Split (80/20)
   ↓
5. Train Models
   - Logistic Regression
   - Random Forest
   ↓
6. Evaluate Models
   - Confusion Matrix
   - ROC Curve
   - Precision-Recall Curve
   - Calculate metrics
   ↓
7. Compare Models
   - Show metrics table
   - Identify best model
   ↓
8. Save Everything
   - Trained models (.pkl)
   - Scaler & encoders
   - Evaluation plots (.png)
```

### Expected Output Files

After running, you'll have:

```
Customer-Churn-Prediction/
├── data/
│   └── customer_churn_data.csv     (5000 customer records)
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── logistic_regression_evaluation.png
│   └── random_forest_evaluation.png
```

---

## Troubleshooting

### Issue 1: "python: command not found"

**Cause**: Python not installed or not in PATH

**Fix**:
1. Download Python from https://www.python.org/
2. During installation, check "Add Python to PATH"
3. Restart your terminal

### Issue 2: "No module named 'pandas'" (or any other module)

**Cause**: Virtual environment not activated or packages not installed

**Fix**:
```bash
# Activate venv (you should see (venv) in terminal)
venv\Scripts\activate

# Reinstall packages
pip install -r requirements.txt
```

### Issue 3: "Permission Denied" when activating venv (PowerShell)

**Cause**: PowerShell execution policy

**Fix**:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activating again.

### Issue 4: "FileNotFoundError: data/customer_churn_data.csv"

**Cause**: Dataset not generated yet

**Fix**:
```bash
python generate_dataset.py
```

### Issue 5: Plots not displaying

**Cause**: Interactive backend not available

**Fix**: Plots are automatically saved to `models/` folder. Check there for PNG files.

### Issue 6: "ImportError: DLL load failed" (Windows)

**Cause**: Missing Visual C++ redistributables

**Fix**: Download and install:
- Microsoft Visual C++ Redistributable
- Link: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Issue 7: Slow execution

**Cause**: Large dataset or slow CPU

**Fix**:
- Reduce dataset size in `generate_dataset.py`: Change `n_samples = 5000` to `n_samples = 1000`
- Use fewer trees in Random Forest: Change `n_estimators=100` to `n_estimators=50`

---

## Understanding the Output

### Console Output Sections

1. **Loading Data** - Confirms dataset loaded
2. **EDA** - Shows dataset statistics and churn distribution
3. **Feature Engineering** - Lists created features
4. **Train-Test Split** - Shows data split sizes
5. **Model Training** - Confirms each model trained
6. **Evaluation** - Shows metrics for each model
7. **Comparison** - Compares both models side-by-side
8. **Saving** - Confirms models saved

### Metrics Explained

- **Precision**: Of predicted churns, how many were correct?
- **Recall**: Of actual churns, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall model performance (0.5 = random, 1.0 = perfect)

### Good Performance Indicators

- ✅ ROC-AUC > 0.80 (Good)
- ✅ ROC-AUC > 0.85 (Excellent)
- ✅ F1-Score > 0.70 (Good)
- ✅ Balanced Precision & Recall

---

## Advanced Usage

### Modify Dataset Size

Edit `generate_dataset.py`:
```python
n_samples = 10000  # Change from 5000 to 10000
```

### Change Test Size

Edit `churn_prediction.py` in the `run_full_pipeline()` method:
```python
self.split_and_scale(X, y, test_size=0.3)  # Change from 0.2 to 0.3
```

### Add More Models

Add to `churn_prediction.py`:
```python
from sklearn.ensemble import GradientBoostingClassifier

def train_gradient_boosting(self):
    self.gb_model = GradientBoostingClassifier(random_state=42)
    self.gb_model.fit(self.X_train, self.y_train)
```

### Export Results to CSV

Add to the end of `churn_prediction.py`:
```python
# Save predictions
results_df = pd.DataFrame({
    'Actual': self.y_test,
    'Predicted': y_pred
})
results_df.to_csv('models/predictions.csv', index=False)
```

---

## Next Steps

Once you've successfully run the project:

1. ✅ **Understand the Code** - Read through `churn_prediction.py` line by line
2. ✅ **Experiment** - Try different parameters and models
3. ✅ **Visualize** - Open the generated PNG files in `models/`
4. ✅ **Jupyter** - Explore the notebook for interactive analysis
5. ✅ **Customize** - Add your own features or models
6. ✅ **Deploy** - Create a web interface with Flask or Streamlit

---

## Getting Help

If you encounter issues:

1. **Check this guide** - Solutions for common problems above
2. **Read error messages** - They usually tell you what's wrong
3. **Verify installation**:
   ```bash
   python --version
   pip list
   ```
4. **Try fresh start**:
   ```bash
   # Delete venv folder
   # Create new venv
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Quick Command Reference

```bash
# Create venv
python -m venv venv

# Activate venv (Windows CMD)
venv\Scripts\activate

# Activate venv (Windows PowerShell)
venv\Scripts\Activate.ps1

# Activate venv (macOS/Linux)
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Generate dataset
python generate_dataset.py

# Run prediction
python churn_prediction.py

# Start Jupyter
jupyter notebook

# Deactivate venv
deactivate
```

---

**✅ You're all set! Enjoy predicting customer churn! 🎯**
