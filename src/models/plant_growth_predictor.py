import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class PlantGrowthPredictor:
    """
    Advanced IoT Plant Growth Prediction Model
    Predicts plant class based on growth parameters from IoT greenhouse data
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            ' Average  of chlorophyll in the plant (ACHP)',
            ' Plant height rate (PHR)',
            'Average wet weight of the growth vegetative (AWWGV)',
            'Average leaf area of the plant (ALAP)',
            'Average number of plant leaves (ANPL)',
            'Average root diameter (ARD)',
            ' Average dry weight of the root (ADWR)',
            ' Percentage of dry matter for vegetative growth (PDMVG)',
            'Average root length (ARL)',
            'Average wet weight of the root (AWWR)',
            ' Average dry weight of vegetative plants (ADWV)',
            'Percentage of dry matter for root growth (PDMRG)'
        ]

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the IoT dataset"""
        df = pd.read_csv(file_path)
        return df

    def feature_analysis(self, df):
        """Analyze features and their relationships"""
        correlation_matrix = df[self.feature_columns].corr()
        feature_importance = self._calculate_feature_importance(df)
        return feature_importance, correlation_matrix

    def _calculate_feature_importance(self, df):
        X = df[self.feature_columns]
        y = df['Class']
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

    def prepare_data(self, df):
        """Prepare data for machine learning"""
        X = df[self.feature_columns].copy()
        y = df['Class'].copy()
        X = X.fillna(X.mean())
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test

    def train_models(self, X_train, y_train):
        """Train multiple models and select the best one"""
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        cv_scores = {}
        for name, model in models.items():
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_scores[name] = scores
            
        best_model_name = max(cv_scores.keys(), key=lambda x: cv_scores[x].mean())
        self.model = models[best_model_name]
        self.model.fit(X_train, y_train)
        
        return best_model_name, cv_scores

    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for the best model"""
        param_grid = self._get_param_grid()
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def _get_param_grid(self):
        if isinstance(self.model, RandomForestClassifier):
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif isinstance(self.model, GradientBoostingClassifier):
            return {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:  # SVM
            return {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            }

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, y_pred, confusion_matrix(y_test, y_pred)

    def predict_single_sample(self, sample_data):
        """Predict class for a single sample"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if isinstance(sample_data, dict):
            sample_array = np.array([sample_data[col] for col in self.feature_columns]).reshape(1, -1)
        else:
            sample_array = np.array(sample_data).reshape(1, -1)

        sample_scaled = self.scaler.transform(sample_array)
        prediction = self.model.predict(sample_scaled)[0]
        probability = self.model.predict_proba(sample_scaled)[0]
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_class, probability

    def save_model(self, filepath):
        """Save the trained model and preprocessing objects"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        """Load a trained model and preprocessing objects"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
