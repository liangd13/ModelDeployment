import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report
)
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'chinesebmtzxz.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'trained_model.pkl')
RANDOM_STATE = 2024


def load_and_preprocess_data():
    data = pd.read_csv(DATA_PATH)
    y = data['Metastasis']
    X = data.drop(columns=['Metastasis'])
    return X, y


def resample_data(X, y):
    smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE)
    X_smote, y_smote = smote.fit_resample(X, y)
    
    tomek = TomekLinks()
    X_resampled, y_resampled = tomek.fit_resample(X_smote, y_smote)
    
    print('Original dataset shape:', Counter(y))
    print('After SMOTE shape:', Counter(y_smote))
    print('After TomekLinks shape:', Counter(y_resampled))
    
    return X_resampled, y_resampled


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def build_stacking_model():
    base_learners = [
        ('lr', Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(
                penalty="l1",
                C=0.05,
                class_weight=None,
                solver="saga",
                max_iter=10000,
                random_state=RANDOM_STATE
            ))
        ])),
        
        ('svc', Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=10,
                gamma=0.1,
                class_weight=None,
                probability=True,
                kernel="rbf",
                random_state=RANDOM_STATE
            ))
        ])),
        
        ('rf', RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=1,
            max_features=0.3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        )),
        
        ('knn', Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=15,
                metric="manhattan",
                weights="distance"
            ))
        ])),
        
        ('xgb', XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.03,
            subsample=1.0,
            colsample_bytree=0.6,
            min_child_weight=1,
            reg_lambda=1,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=1
        )),
        
        ('mlp', Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128,),
                alpha=0.01,
                learning_rate_init=0.001,
                activation="relu",
                max_iter=800,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=RANDOM_STATE
            ))
        ]))
    ]
    
    meta_learner = RandomForestClassifier(
        n_estimators=1000,
        max_depth=5,
        min_samples_leaf=10,
        max_features=0.6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    
    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        stack_method="predict_proba",
        passthrough=False,
        cv=5,
        n_jobs=1
    )
    
    return stacking_clf


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print("Model Performance on Test Set")
    print("=" * 50)
    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_and_save_model():
    print("Loading data...")
    X, y = load_and_preprocess_data()
    
    print("\nResampling data...")
    X_resampled, y_resampled = resample_data(X, y)
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)
    
    print("\nBuilding Stacking model...")
    stacking_clf = build_stacking_model()
    
    print("\nTraining model...")
    stacking_clf.fit(X_train, y_train)
    
    metrics = evaluate_model(stacking_clf, X_test, y_test)
    
    print(f"\nSaving model to {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(stacking_clf, f)
    
    print("\nModel training completed!")
    print(f"Model saved: {MODEL_PATH}")
    
    return stacking_clf, metrics


if __name__ == '__main__':
    train_and_save_model()
