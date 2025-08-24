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
        
        # Fix paths to use the newly trained model files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ml_models_dir = os.path.join(current_dir, '..', 'ml_models')
        
        self.model_path = os.path.join(ml_models_dir, "autism_model.pkl")  # Updated name
        self.encoders_path = os.path.join(ml_models_dir, "label_encoders.pkl")
        self.scaler_path = os.path.join(ml_models_dir, "scaler.pkl")
        self.feature_columns_path = os.path.join(ml_models_dir, "feature_columns.pkl")  # New file
        self.training_info = {}
        
        # Try to load the model immediately
        self.load_model()
        
    def load_training_data(self):
        """Load and prepare training data from expanded.csv"""
        try:
            # Load the expanded dataset - correct path from backend directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, '..', '..', 'data', 'expanded.csv')
            
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
                logger.info("✅ Loaded autism_model.pkl")
            else:
                logger.warning(f"No saved model found at {self.model_path}")
                return False
            
            # Load encoders
            if os.path.exists(self.encoders_path):
                with open(self.encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
                logger.info("✅ Loaded label_encoders.pkl")
            
            # Load scaler
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ Loaded scaler.pkl")
            
            # Load feature columns
            if os.path.exists(self.feature_columns_path):
                with open(self.feature_columns_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                logger.info(f"✅ Loaded {len(self.feature_names)} feature columns")
            else:
                logger.warning("Feature columns file not found, using default order")
                # Default feature order matching training data
                aq_columns = [f'A{i}_Score' for i in range(1, 21)]
                demographic_columns = ['age', 'gender', 'ethnicity', 'jaundice', 'austim', 
                                     'contry_of_res', 'used_app_before', 'relation']
                self.feature_names = aq_columns + demographic_columns
            
            logger.info("🎯 Model loaded successfully - ready for predictions")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, features):
        """Make prediction on processed features"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            logger.info(f"🔍 DEBUG: Received features: {features}")
            
            # Preprocess features to match training data format
            feature_array = self.preprocess_features_fixed(features)
            if feature_array is None:
                raise ValueError("Feature preprocessing failed")
            
            # Scale the features
            X_scaled = self.scaler.transform(feature_array)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Format results
            prediction_label = "ASD" if prediction == 1 else "No ASD"
            confidence = max(probabilities) * 100
            
            logger.info(f"🎯 ML Prediction: {prediction_label} ({confidence:.1f}%)")
            
            return {
                'prediction': prediction_label,
                'confidence': confidence,
                'aq_score': features.get('aq_total_score', 0),
                'probability_no_asd': probabilities[0] * 100,
                'probability_asd': probabilities[1] * 100 if len(probabilities) > 1 else 0
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
                'aq_score': aq_score,
                'probability_no_asd': confidence if prediction == "No ASD" else 100 - confidence,
                'probability_asd': confidence if prediction == "ASD" else 100 - confidence
            }
    
    def preprocess_features_fixed(self, data):
        """Preprocess features to match training data format"""
        try:
            # Create feature vector in correct order to match training data
            # Training data has: 20 AQ + age only = 21 features total
            features = []
            
            # Add AQ scores (already converted to 0-1 by data_processor)
            for i in range(1, 21):
                features.append(data.get(f'A{i}_Score', 0))
            
            # Add only age to match the 21-feature training model
            features.append(data.get('age', 25))
            
            # Convert to numpy array and reshape
            feature_array = np.array(features).reshape(1, -1)
            
            logger.info(f"🔄 Preprocessed features: {feature_array.shape} (20 AQ + 1 age = 21 total)")
            
            if feature_array.shape[1] != 21:
                logger.error(f"Feature count mismatch: expected 21, got {feature_array.shape[1]}")
                return None
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error in preprocess_features_fixed: {str(e)}")
            return None
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            return None
    
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
