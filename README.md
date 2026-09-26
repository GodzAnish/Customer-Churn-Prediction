# Customer Churn Prediction System 🎯

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)
![Status](https://img.shields.io/badge/Status-Ready-green)

A comprehensive machine learning system built with **Python, NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn** to predict customer churn. This project performs EDA, feature engineering, and trains **Logistic Regression** and **Random Forest** models to identify customers at risk of churning. Models are evaluated using **Precision, Recall, F1-Score, and ROC-AUC** to handle class imbalance and deliver reliable, data-driven insights for customer retention.

---

## 📋 Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Performance](#model-performance)
- [License](#license)

---

## ✨ Features

- 📊 **Exploratory Data Analysis (EDA)** - Comprehensive data analysis with visualizations
- 🔧 **Feature Engineering** - Create meaningful features from raw data
- 🤖 **Multiple Models** - Logistic Regression & Random Forest implementations
- 📈 **Model Evaluation** - Precision, Recall, F1-Score, ROC-AUC metrics
- ⚖️ **Class Imbalance Handling** - Balanced class weights for better predictions
- 💾 **Model Persistence** - Save and load trained models
- 📊 **Visualizations** - Confusion matrices, ROC curves, precision-recall curves
- 🎯 **Production Ready** - Clean, modular, well-documented code

---

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning algorithms and metrics

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── data/                          # Data directory
│   └── customer_churn_data.csv   # Dataset (generated)
│
├── models/                        # Trained models directory
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── *.png                     # Evaluation plots
│
├── notebooks/                     # Jupyter notebooks
│   └── EDA_Customer_Churn.ipynb  # Exploratory analysis
│
├── .vscode/                       # VS Code configuration
│   ├── settings.json
│   └── launch.json
│
├── churn_prediction.py           # Main prediction script
├── generate_dataset.py           # Dataset generation script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
├── README.md                     # This file
└── SETUP_GUIDE.md               # Detailed setup instructions
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional)

### Step 1: Clone or Download
```bash
# Option A: Clone from GitHub
git clone https://github.com/YOUR-USERNAME/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Option B: Extract the downloaded ZIP file and navigate to it
cd Customer-Churn-Prediction
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Usage

### Method 1: Quick Start (All-in-One)

```bash
# Generate dataset and run prediction pipeline
python generate_dataset.py
python churn_prediction.py
```

### Method 2: Step-by-Step

#### 1. Generate Sample Dataset
```bash
python generate_dataset.py
```
This creates `data/customer_churn_data.csv` with 5000 customer records.

#### 2. Run Churn Prediction
```bash
python churn_prediction.py
```

This will:
- Load and explore the data
- Perform feature engineering
- Train Logistic Regression and Random Forest models
- Evaluate models with multiple metrics
- Save models and visualizations to `models/` folder

#### 3. Explore with Jupyter Notebook (Optional)
```bash
jupyter notebook notebooks/EDA_Customer_Churn.ipynb
```

### Method 3: Using VS Code

1. Open the folder in VS Code
2. Press `F5` and select "Python: Churn Prediction"
3. View results in the integrated terminal

---

## 📊 Results

### What You'll Get

1. **Console Output**
   - Detailed EDA statistics
   - Training progress
   - Model evaluation metrics
   - Comparison between models

2. **Saved Models** (in `models/` folder)
   - `logistic_regression_model.pkl`
   - `random_forest_model.pkl`
   - `scaler.pkl`
   - `label_encoders.pkl`

3. **Visualizations** (in `models/` folder)
   - Confusion matrices
   - ROC curves
   - Precision-recall curves

### Sample Output

```
📊 EVALUATING LOGISTIC REGRESSION
================================================
Classification Report:
              precision    recall  f1-score   support

   No Churn       0.82      0.89      0.85       650
       Churn       0.78      0.67      0.72       350

🎯 Key Metrics:
   ROC-AUC Score: 0.8543
   F1-Score: 0.7234

🏆 MODEL COMPARISON
================================================
             Model  ROC-AUC  F1-Score
 Logistic Regression   0.8543    0.7234
       Random Forest   0.8721    0.7456

🥇 Best Model (by ROC-AUC): Random Forest
```

---

## 📈 Model Performance

### Evaluation Metrics

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.78 | 0.67 | 0.72 | 0.85 |
| Random Forest | 0.81 | 0.72 | 0.76 | 0.87 |

### Key Insights

- **Random Forest** typically outperforms Logistic Regression
- Both models handle class imbalance effectively
- ROC-AUC scores indicate strong predictive power
- Models provide reliable insights for customer retention strategies

---

## 🎓 Learning Outcomes

This project demonstrates:

1. ✅ **Data Analysis** - EDA and visualization techniques
2. ✅ **Feature Engineering** - Creating predictive features
3. ✅ **Machine Learning** - Training and evaluating multiple models
4. ✅ **Model Evaluation** - Comprehensive metrics (Precision, Recall, F1, ROC-AUC)
5. ✅ **Class Imbalance** - Handling imbalanced datasets
6. ✅ **Best Practices** - Clean code, documentation, version control

---

## 🔧 Troubleshooting

### Issue: ModuleNotFoundError
**Fix:**
```bash
# Make sure venv is activated
venv\Scripts\activate

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: Dataset not found
**Fix:**
```bash
# Generate the dataset first
python generate_dataset.py
```

### Issue: Matplotlib plots not showing
**Fix:** Add this to the top of the script:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the maintainer.

---

## 🙏 Acknowledgments

- Dataset generated synthetically for educational purposes
- Inspired by real-world customer churn analysis problems
- Built with industry-standard tools and best practices

---

**Made with ❤️ using Python, Pandas, Scikit-learn, Matplotlib, and Seaborn**

---

## 🚀 Next Steps

- Try different algorithms (XGBoost, Neural Networks)
- Add hyperparameter tuning
- Implement cross-validation
- Create a web interface with Flask/Streamlit
- Deploy model to production
