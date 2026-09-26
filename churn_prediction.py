"""
Customer Churn Prediction System
Built with Python, Scikit-learn, performing EDA, feature engineering,
and training Logistic Regression and Random Forest models.
Evaluates models using Precision, Recall, F1-Score, and ROC-AUC.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class CustomerChurnPredictor:
    """
    Customer Churn Prediction Model
    Handles data preprocessing, feature engineering, model training, and evaluation
    """

    def __init__(self, data_path='data/customer_churn_data.csv'):
        """Initialize the predictor with data"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.lr_model = None
        self.rf_model = None

    def load_data(self):
        """Load the customer churn dataset"""
        print("📂 Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def explore_data(self):
        """Perform basic exploratory data analysis"""
        print("\n" + "="*60)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*60)

        print("\n🔍 Dataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"\nData Types:\n{self.df.dtypes.value_counts()}")

        print(f"\n📈 Churn Distribution:")
        churn_counts = self.df['Churn'].value_counts()
        print(f"No Churn (0): {churn_counts[0]} ({churn_counts[0]/len(self.df)*100:.2f}%)")
        print(f"Churn (1): {churn_counts[1]} ({churn_counts[1]/len(self.df)*100:.2f}%)")

        print(f"\n❓ Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found! ✓")
        else:
            print(missing[missing > 0])

        print(f"\n📊 Numerical Features Summary:")
        print(self.df.describe())

    def feature_engineering(self):
        """Perform feature engineering and preprocessing"""
        print("\n" + "="*60)
        print("⚙️  FEATURE ENGINEERING")
        print("="*60)

        # Create a copy for processing
        df_processed = self.df.copy()

        # Drop CustomerID as it's not useful for prediction
        df_processed = df_processed.drop('CustomerID', axis=1)

        # Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Churn']

        print(f"\n🔤 Encoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le

        # Create new features
        print("🔧 Creating new features...")
        df_processed['ChargesPerMonth'] = df_processed['TotalCharges'] / (df_processed['Tenure'] + 1)
        df_processed['TenureGroup'] = pd.cut(df_processed['Tenure'], bins=[0, 12, 24, 48, 72], labels=[0, 1, 2, 3])
        df_processed['AgeGroup'] = pd.cut(df_processed['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])

        # Separate features and target
        X = df_processed.drop('Churn', axis=1)
        y = df_processed['Churn']

        print(f"✅ Final feature set: {X.shape[1]} features")
        print(f"📋 Features: {list(X.columns)[:10]}... (showing first 10)")

        return X, y

    def split_and_scale(self, X, y, test_size=0.2):
        """Split data and apply scaling"""
        print("\n" + "="*60)
        print("✂️  TRAIN-TEST SPLIT & SCALING")
        print("="*60)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"📊 Training set: {self.X_train.shape[0]} samples")
        print(f"📊 Test set: {self.X_test.shape[0]} samples")

        # Scale features
        print("\n⚖️  Scaling features...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print("✅ Scaling complete")

    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "="*60)
        print("🤖 TRAINING LOGISTIC REGRESSION")
        print("="*60)

        self.lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )

        print("🔄 Training...")
        self.lr_model.fit(self.X_train, self.y_train)
        print("✅ Logistic Regression trained successfully")

    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("🌲 TRAINING RANDOM FOREST")
        print("="*60)

        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced'  # Handle class imbalance
        )

        print("🔄 Training...")
        self.rf_model.fit(self.X_train, self.y_train)
        print("✅ Random Forest trained successfully")

    def evaluate_model(self, model, model_name):
        """Evaluate model with comprehensive metrics"""
        print("\n" + "="*60)
        print(f"📈 EVALUATING {model_name.upper()}")
        print("="*60)

        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Churn', 'Churn']))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\n🔢 Confusion Matrix:")
        print(cm)

        # Calculate metrics
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        f1 = f1_score(self.y_test, y_pred)

        print(f"\n🎯 Key Metrics:")
        print(f"   ROC-AUC Score: {roc_auc:.4f}")
        print(f"   F1-Score: {f1:.4f}")

        # Visualizations
        self._plot_evaluation(model_name, y_pred_proba, cm)

        return {
            'model_name': model_name,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    def _plot_evaluation(self, model_name, y_pred_proba, cm):
        """Create evaluation plots"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Confusion Matrix Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        axes[0].set_title(f'{model_name} - Confusion Matrix')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'{model_name} - ROC Curve')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        axes[2].plot(recall, precision, color='green', lw=2)
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title(f'{model_name} - Precision-Recall Curve')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        os.makedirs('models', exist_ok=True)
        plt.savefig(f'models/{model_name.lower().replace(" ", "_")}_evaluation.png', dpi=300, bbox_inches='tight')
        print(f"📊 Evaluation plot saved: models/{model_name.lower().replace(' ', '_')}_evaluation.png")
        plt.close()

    def compare_models(self, lr_results, rf_results):
        """Compare both models"""
        print("\n" + "="*60)
        print("🏆 MODEL COMPARISON")
        print("="*60)

        comparison = pd.DataFrame({
            'Model': [lr_results['model_name'], rf_results['model_name']],
            'ROC-AUC': [lr_results['roc_auc'], rf_results['roc_auc']],
            'F1-Score': [lr_results['f1_score'], rf_results['f1_score']]
        })

        print("\n", comparison.to_string(index=False))

        # Determine best model
        best_model = 'Logistic Regression' if lr_results['roc_auc'] > rf_results['roc_auc'] else 'Random Forest'
        print(f"\n🥇 Best Model (by ROC-AUC): {best_model}")

    def save_models(self):
        """Save trained models"""
        print("\n" + "="*60)
        print("💾 SAVING MODELS")
        print("="*60)

        os.makedirs('models', exist_ok=True)

        # Save models
        with open('models/logistic_regression_model.pkl', 'wb') as f:
            pickle.dump(self.lr_model, f)
        print("✅ Logistic Regression model saved")

        with open('models/random_forest_model.pkl', 'wb') as f:
            pickle.dump(self.rf_model, f)
        print("✅ Random Forest model saved")

        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✅ Scaler saved")

        # Save label encoders
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print("✅ Label encoders saved")

        print("\n📦 All models saved to 'models/' directory")

    def run_full_pipeline(self):
        """Execute the complete churn prediction pipeline"""
        print("\n" + "="*60)
        print("🚀 CUSTOMER CHURN PREDICTION SYSTEM")
        print("="*60)

        # Load and explore
        self.load_data()
        self.explore_data()

        # Feature engineering
        X, y = self.feature_engineering()

        # Split and scale
        self.split_and_scale(X, y)

        # Train models
        self.train_logistic_regression()
        self.train_random_forest()

        # Evaluate models
        lr_results = self.evaluate_model(self.lr_model, "Logistic Regression")
        rf_results = self.evaluate_model(self.rf_model, "Random Forest")

        # Compare models
        self.compare_models(lr_results, rf_results)

        # Save models
        self.save_models()

        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE!")
        print("="*60)
        print("\n📁 Check the 'models/' folder for saved models and visualizations")


if __name__ == "__main__":
    # Create predictor instance
    predictor = CustomerChurnPredictor()

    # Run full pipeline
    predictor.run_full_pipeline()
