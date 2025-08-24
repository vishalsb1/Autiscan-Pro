"""
Autism Classifier - Machine Learning Model for ASD Prediction
Privacy-preserving model that processes data in memory only
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)

class AutismClassifier:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = "../ml_models/autism_classifier.pkl"
        self.encoders_path = "../ml_models/label_encoders.pkl"
        self.scaler_path = "../ml_models/scaler.pkl"
        self.training_info = {}
        
    def load_training_data(self):
        """Load and prepare training data from expanded.csv"""
        try:
            # Load the expanded dataset
            data_path = "../data/expanded.csv"
            if not os.path.exists(data_path):
                logger.error(f"Training data not found at {data_path}")
                return None, None
            
            df = pd.read_csv(data_path)
            logger.info(f"Loaded training data: {df.shape[0]} samples, {df.shape[1]} features")
            
            # Define AQ-20 questions (A1_Score to A20_Score)
            aq_columns = [f'A{i}_Score' for i in range(1, 21)]
            
            # Define demographic columns
            demographic_columns = ['age', 'gender', 'ethnicity', 'jaundice', 'austim', 
                                  'contry_of_res', 'used_app_before', 'relation']
            
            # Select features and target
            feature_columns = aq_columns + demographic_columns
            target_column = 'Class/ASD'
            
            # Check if all required columns exist
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                feature_columns = [col for col in feature_columns if col in df.columns]
            
            if target_column not in df.columns:
                logger.error(f"Target column '{target_column}' not found")
                return None, None
            
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Store feature names
            self.feature_names = feature_columns
            
            logger.info(f"Features selected: {len(feature_columns)}")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            return None, None
    
    def preprocess_data(self, X, fit_encoders=False):
        """Preprocess features for training or prediction"""
        try:
            X_processed = X.copy()
            
            # Handle missing values
            X_processed = X_processed.fillna({
                'age': X_processed['age'].median() if 'age' in X_processed.columns else 25,
                'gender': 'unknown',
                'ethnicity': 'unknown',
                'jaundice': 'no',
                'austim': 'no',
                'contry_of_res': 'unknown',
                'used_app_before': 'no',
                'relation': 'Self'
            })
            
            # Encode categorical variables
            categorical_columns = ['gender', 'ethnicity', 'jaundice', 'austim', 
                                 'contry_of_res', 'used_app_before', 'relation']
            
            for col in categorical_columns:
                if col in X_processed.columns:
                    if fit_encoders:
                        # Fit new encoder during training
                        if col not in self.label_encoders:
                            self.label_encoders[col] = LabelEncoder()
                        X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col].astype(str))
                    else:
                        # Use existing encoder during prediction
                        if col in self.label_encoders:
                            # Handle unseen categories
                            unique_values = set(X_processed[col].astype(str))
                            known_values = set(self.label_encoders[col].classes_)
                            unseen_values = unique_values - known_values
                            
                            if unseen_values:
                                # Replace unseen values with the most common category
                                most_common = self.label_encoders[col].classes_[0]
                                X_processed[col] = X_processed[col].astype(str).replace(
                                    list(unseen_values), most_common
                                )
                            
                            X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
                        else:
                            # If encoder doesn't exist, use dummy encoding
                            X_processed[col] = 0
            
            # Scale numerical features
            numerical_columns = ['age'] + [f'A{i}_Score' for i in range(1, 21)]
            numerical_columns = [col for col in numerical_columns if col in X_processed.columns]
            
            if numerical_columns:
                if fit_encoders:
                    X_processed[numerical_columns] = self.scaler.fit_transform(X_processed[numerical_columns])
                else:
                    X_processed[numerical_columns] = self.scaler.transform(X_processed[numerical_columns])
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return X
    
    def train_model(self):
        """Train the autism classification model"""
        try:
            logger.info("🤖 Starting model training...")
            
            # Load training data
            X, y = self.load_training_data()
            if X is None or y is None:
                logger.error("Failed to load training data")
                return False
            
            # Preprocess data
            X_processed = self.preprocess_data(X, fit_encoders=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'  # Handle class imbalance
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_processed, y, cv=5)
            
            # Predictions for detailed metrics
            y_pred = self.model.predict(X_test)
            
            # Store training information
            self.training_info = {
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': len(self.feature_names),
                'train_accuracy': round(train_score, 4),
                'test_accuracy': round(test_score, 4),
                'cv_mean_accuracy': round(cv_scores.mean(), 4),
                'cv_std_accuracy': round(cv_scores.std(), 4),
                'feature_importance': dict(zip(
                    self.feature_names, 
                    [round(imp, 4) for imp in self.model.feature_importances_]
                ))
            }
            
            logger.info(f"✅ Model training completed!")
            logger.info(f"   Train Accuracy: {train_score:.4f}")
            logger.info(f"   Test Accuracy: {test_score:.4f}")
            logger.info(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Save model and encoders
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def save_model(self):
        """Save trained model and encoders"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save encoders
            with open(self.encoders_path, 'wb') as f:
                pickle.dump(self.label_encoders, f)
            
            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save training info
            training_info_path = "../ml_models/training_info.pkl"
            with open(training_info_path, 'wb') as f:
                pickle.dump({
                    'training_info': self.training_info,
                    'feature_names': self.feature_names
                }, f)
            
            logger.info("💾 Model and encoders saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load trained model and encoders"""
        try:
            # Load model
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                logger.warning("No saved model found")
                return False
            
            # Load encoders
            if os.path.exists(self.encoders_path):
                with open(self.encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
            
            # Load scaler
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load training info
            training_info_path = "../ml_models/training_info.pkl"
            if os.path.exists(training_info_path):
                with open(training_info_path, 'rb') as f:
                    data = pickle.load(f)
                    self.training_info = data.get('training_info', {})
                    self.feature_names = data.get('feature_names', [])
            
            logger.info("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, features):
        """Make prediction on processed features"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Convert to DataFrame if it's a dict
            if isinstance(features, dict):
                # Create DataFrame with expected features
                feature_df = pd.DataFrame([features])
                
                # Ensure all expected features are present
                for feature in self.feature_names:
                    if feature not in feature_df.columns:
                        if feature.startswith('A') and feature.endswith('_Score'):
                            feature_df[feature] = 0  # Default AQ score
                        else:
                            feature_df[feature] = 'unknown'  # Default demographic
                
                # Reorder columns to match training order
                feature_df = feature_df[self.feature_names]
            else:
                feature_df = features
            
            # Preprocess features
            X_processed = self.preprocess_data(feature_df, fit_encoders=False)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(X_processed)[0]
            prediction = self.model.predict(X_processed)[0]
            
            # Convert to human-readable format
            prediction_label = "ASD" if prediction == 1 else "No ASD"
            confidence = max(prediction_proba) * 100
            
            return {
                'prediction': prediction_label,
                'confidence': confidence,
                'probability_no_asd': prediction_proba[0] * 100,
                'probability_asd': prediction_proba[1] * 100 if len(prediction_proba) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            # Fallback to simple rule-based prediction
            aq_score = features.get('aq_total_score', 0) if isinstance(features, dict) else 0
            prediction = "ASD" if aq_score >= 6 else "No ASD"
            confidence = 70  # Lower confidence for fallback
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probability_no_asd': confidence if prediction == "No ASD" else 100 - confidence,
                'probability_asd': confidence if prediction == "ASD" else 100 - confidence
            }
    
    def initialize(self):
        """Initialize the model (load or train if needed)"""
        try:
            # Try to load existing model
            if self.load_model():
                logger.info("✅ Model loaded from saved files")
                return True
            else:
                logger.info("🤖 No saved model found, training new model...")
                if self.train_model():
                    logger.info("✅ New model trained successfully")
                    return True
                else:
                    logger.error("❌ Model training failed")
                    return False
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            return False
    
    def is_model_loaded(self):
        """Check if model is loaded and ready"""
        return self.model is not None
    
    def get_model_type(self):
        """Get model type information"""
        return "Random Forest Classifier" if self.model else "Not loaded"
    
    def get_training_info(self):
        """Get training information"""
        return self.training_info.get('training_samples', 0)
    
    def get_detailed_model_info(self):
        """Get detailed model information"""
        if not self.model:
            return {"status": "Model not loaded"}
        
        return {
            "model_type": "Random Forest Classifier",
            "is_loaded": True,
            "training_info": self.training_info,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "privacy_mode": True,
            "data_storage": False
        }
