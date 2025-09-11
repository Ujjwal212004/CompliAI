import numpy as np
import pandas as pd
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging
import pickle
import joblib
from pathlib import Path

# ML/AI imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb

# Text processing
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Try to download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

from dataset_manager import ComplianceDatasetManager

class ComplianceMLTrainer:
    """
    Advanced ML training pipeline for Legal Metrology compliance prediction
    Implements multiple models for field detection, text classification, and compliance scoring
    """
    
    def __init__(self, dataset_manager: ComplianceDatasetManager, model_dir: str = "models"):
        self.dataset_manager = dataset_manager
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.setup_logging()
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.encoders = {}
        
        # Field categories for Legal Metrology
        self.field_categories = [
            'manufacturer', 'net_quantity', 'mrp', 'consumer_care',
            'mfg_date', 'country_origin', 'product_name'
        ]
        
        # Initialize text preprocessing
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def setup_logging(self):
        """Setup logging for ML training"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ml_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing for Legal Metrology context"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important symbols for Legal Metrology
        # Keep: ₹, Rs, %, /, -, :, @, .
        text = re.sub(r'[^\w\s₹@%/:.-]', ' ', text)
        
        # Normalize common Legal Metrology terms
        text = re.sub(r'\b(rs|inr|rupees)\b', 'rupees', text)
        text = re.sub(r'\b(gm|gms|grams)\b', 'gram', text)
        text = re.sub(r'\b(kg|kgs|kilograms)\b', 'kilogram', text)
        text = re.sub(r'\b(ml|mls|millilitres)\b', 'millilitre', text)
        text = re.sub(r'\b(ltd|limited|pvt|private)\b', 'company', text)
        text = re.sub(r'\b(mfd|manufactured|mfg)\b', 'manufactured', text)
        text = re.sub(r'\b(incl|inclusive|including)\b', 'inclusive', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features_for_field(self, text: str, field_name: str) -> Dict[str, float]:
        """Extract specialized features for each Legal Metrology field"""
        features = {}
        
        if not text:
            return {f"{field_name}_missing": 1.0}
        
        text_lower = text.lower()
        
        # Common features for all fields
        features.update({
            f"{field_name}_length": len(text),
            f"{field_name}_word_count": len(text.split()),
            f"{field_name}_has_numbers": float(bool(re.search(r'\d', text))),
            f"{field_name}_has_special_chars": float(bool(re.search(r'[^\w\s]', text))),
        })
        
        # Field-specific features based on Legal Metrology Rules 2011
        if field_name == 'manufacturer':
            features.update({
                'manufacturer_has_company_indicator': float(any(
                    indicator in text_lower for indicator in ['ltd', 'pvt', 'inc', 'company', 'co']
                )),
                'manufacturer_has_address_parts': float(any(
                    part in text_lower for part in ['plot', 'building', 'street', 'road', 'area']
                )),
                'manufacturer_has_pin_code': float(bool(re.search(r'\b\d{6}\b', text))),
                'manufacturer_has_city': float(any(
                    city in text_lower for city in ['mumbai', 'delhi', 'bangalore', 'chennai', 'pune', 'kolkata']
                )),
                'manufacturer_completeness': self._calculate_manufacturer_completeness(text)
            })
        
        elif field_name == 'net_quantity':
            features.update({
                'net_quantity_has_weight_units': float(any(
                    unit in text_lower for unit in ['g', 'gm', 'gram', 'kg', 'kilogram']
                )),
                'net_quantity_has_volume_units': float(any(
                    unit in text_lower for unit in ['ml', 'millilitre', 'l', 'litre', 'liter']
                )),
                'net_quantity_has_count_units': float(any(
                    unit in text_lower for unit in ['pieces', 'pcs', 'nos', 'units']
                )),
                'net_quantity_has_net_indicator': float(any(
                    indicator in text_lower for indicator in ['net', 'wt', 'weight', 'qty', 'quantity']
                )),
                'net_quantity_proper_format': float(bool(re.search(
                    r'\d+\.?\d*\s*(g|gm|gram|kg|ml|l|litre|liter|pieces|pcs|nos|units)\b', text_lower
                )))
            })
        
        elif field_name == 'mrp':
            features.update({
                'mrp_has_currency': float(any(
                    currency in text_lower for currency in ['₹', 'rs', 'inr', 'rupees']
                )),
                'mrp_has_tax_clause': float(any(
                    clause in text_lower for clause in ['inclusive', 'incl', 'including', 'taxes', 'tax']
                )),
                'mrp_has_proper_label': float(any(
                    label in text_lower for label in ['mrp', 'maximum', 'retail', 'price']
                )),
                'mrp_numeric_value': self._extract_price_value(text),
                'mrp_format_score': self._calculate_mrp_format_score(text)
            })
        
        elif field_name == 'consumer_care':
            features.update({
                'consumer_care_has_phone': float(bool(re.search(r'\b\d{10}\b', text))),
                'consumer_care_has_email': float(bool(re.search(
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text
                ))),
                'consumer_care_has_toll_free': float(bool(re.search(r'1800\s*\d{6,7}', text))),
                'consumer_care_has_labels': float(any(
                    label in text_lower for label in ['customer', 'consumer', 'care', 'complaint', 'helpline']
                )),
                'consumer_care_contact_score': self._calculate_contact_score(text)
            })
        
        elif field_name == 'mfg_date':
            features.update({
                'mfg_date_has_month_year': float(bool(re.search(r'\d{1,2}/\d{4}', text))),
                'mfg_date_has_full_date': float(bool(re.search(r'\d{1,2}/\d{1,2}/\d{4}', text))),
                'mfg_date_has_month_name': float(bool(re.search(
                    r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', text_lower
                ))),
                'mfg_date_has_labels': float(any(
                    label in text_lower for label in ['mfd', 'manufactured', 'mfg', 'best', 'before', 'exp']
                )),
                'mfg_date_format_score': self._calculate_date_format_score(text)
            })
        
        elif field_name == 'country_origin':
            features.update({
                'country_origin_has_made_in': float('made in' in text_lower),
                'country_origin_has_origin_label': float(any(
                    label in text_lower for label in ['origin', 'country', 'manufactured in', 'imported from']
                )),
                'country_origin_has_country_name': float(any(
                    country in text_lower for country in [
                        'india', 'china', 'usa', 'thailand', 'indonesia', 'vietnam', 'malaysia'
                    ]
                )),
                'country_origin_completeness': self._calculate_origin_completeness(text)
            })
        
        elif field_name == 'product_name':
            features.update({
                'product_name_has_brand_indicators': float(any(
                    indicator in text_lower for indicator in ['brand', 'premium', 'organic', 'natural']
                )),
                'product_name_is_descriptive': float(len(text.split()) >= 2),
                'product_name_not_generic': float(not any(
                    generic in text_lower for generic in ['product', 'item', 'goods', 'package']
                ))
            })
        
        return features
    
    def _calculate_manufacturer_completeness(self, text: str) -> float:
        """Calculate manufacturer address completeness score"""
        components = {
            'company': any(indicator in text.lower() for indicator in ['ltd', 'pvt', 'inc', 'company']),
            'address': any(part in text.lower() for part in ['plot', 'building', 'street', 'road']),
            'pin': bool(re.search(r'\b\d{6}\b', text)),
            'location': any(city in text.lower() for city in ['mumbai', 'delhi', 'bangalore', 'chennai'])
        }
        return sum(components.values()) / len(components)
    
    def _extract_price_value(self, text: str) -> float:
        """Extract numeric price value"""
        match = re.search(r'(\d+(?:\.\d{2})?)', text)
        return float(match.group(1)) if match else 0.0
    
    def _calculate_mrp_format_score(self, text: str) -> float:
        """Calculate MRP format compliance score"""
        text_lower = text.lower()
        score = 0.0
        
        # Has currency symbol
        if any(currency in text_lower for currency in ['₹', 'rs', 'rupees']):
            score += 0.25
        
        # Has numeric value
        if re.search(r'\d+', text):
            score += 0.25
        
        # Has tax inclusion
        if any(tax in text_lower for tax in ['inclusive', 'incl', 'taxes']):
            score += 0.25
        
        # Has MRP label
        if 'mrp' in text_lower:
            score += 0.25
        
        return score
    
    def _calculate_contact_score(self, text: str) -> float:
        """Calculate consumer care contact score"""
        score = 0.0
        
        # Has phone number
        if re.search(r'\b\d{10}\b', text):
            score += 0.5
        
        # Has email
        if re.search(r'@.*\.\w+', text):
            score += 0.5
        
        # Has toll-free
        if re.search(r'1800', text):
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_date_format_score(self, text: str) -> float:
        """Calculate manufacturing date format score"""
        score = 0.0
        
        # Has proper date format
        if re.search(r'\d{1,2}/\d{4}', text):
            score += 0.4
        elif re.search(r'\d{1,2}/\d{1,2}/\d{4}', text):
            score += 0.4
        
        # Has date labels
        if any(label in text.lower() for label in ['mfd', 'manufactured', 'best before']):
            score += 0.6
        
        return score
    
    def _calculate_origin_completeness(self, text: str) -> float:
        """Calculate country of origin completeness"""
        text_lower = text.lower()
        score = 0.0
        
        # Has origin indicators
        if any(indicator in text_lower for indicator in ['made in', 'origin', 'country']):
            score += 0.5
        
        # Has specific country
        if any(country in text_lower for country in ['india', 'china', 'usa', 'thailand']):
            score += 0.5
        
        return score
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Prepare comprehensive training data with features and labels"""
        
        X_data = {}  # Features for each field
        y_data = {}  # Labels for each field
        
        for field_name in self.field_categories:
            field_features = []
            field_labels = []
            
            for _, row in df.iterrows():
                # Parse extracted fields
                extracted_fields = json.loads(row['extracted_fields']) if row['extracted_fields'] else {}
                field_data = extracted_fields.get(field_name, {})
                
                # Extract text value
                text_value = field_data.get('value', '')
                
                # Extract features
                features = self.extract_features_for_field(text_value, field_name)
                
                # Create compliance label based on ground truth or user corrections
                compliance_label = self._get_field_compliance_label(row, field_name)
                
                field_features.append(features)
                field_labels.append(compliance_label)
            
            # Convert to DataFrame for easier handling
            X_data[field_name] = pd.DataFrame(field_features).fillna(0)
            y_data[field_name] = np.array(field_labels)
            
            self.logger.info(f"Prepared {len(field_features)} samples for {field_name}")
        
        return X_data, y_data
    
    def _get_field_compliance_label(self, row: pd.Series, field_name: str) -> int:
        """Get compliance label for a specific field"""
        
        # Check user corrections first
        if row['user_corrections']:
            corrections = json.loads(row['user_corrections'])
            if field_name in corrections:
                correction_data = corrections[field_name]
                if 'compliance' in correction_data:
                    return 1 if correction_data['compliance'] == 'Pass' else 0
        
        # Check extracted field compliance
        extracted_fields = json.loads(row['extracted_fields']) if row['extracted_fields'] else {}
        field_data = extracted_fields.get(field_name, {})
        
        if field_data.get('compliance') == 'Pass':
            return 1
        elif field_data.get('found', False):
            # Found but may have issues - use confidence or other metrics
            confidence = field_data.get('confidence', 0.0)
            return 1 if confidence > 0.7 else 0
        else:
            return 0
    
    def train_field_classifiers(self, X_data: Dict, y_data: Dict) -> Dict:
        """Train individual classifiers for each Legal Metrology field"""
        
        trained_models = {}
        
        for field_name in self.field_categories:
            self.logger.info(f"Training classifier for {field_name}")
            
            X = X_data[field_name]
            y = y_data[field_name]
            
            if len(X) == 0 or len(set(y)) < 2:
                self.logger.warning(f"Insufficient data for {field_name}")
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[field_name] = scaler
            
            # Define multiple models to try
            models_to_try = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                ),
                'logistic_regression': LogisticRegression(
                    random_state=42, max_iter=1000
                ),
                'svm': SVC(random_state=42, probability=True)
            }
            
            best_model = None
            best_score = 0
            best_model_name = None
            
            # Cross-validation for model selection
            cv = StratifiedKFold(n_splits=min(5, len(set(y))), shuffle=True, random_state=42)
            
            for model_name, model in models_to_try.items():
                try:
                    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted')
                    avg_score = np.mean(scores)
                    
                    self.logger.info(f"{field_name} - {model_name}: F1={avg_score:.4f}")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_model_name = model_name
                
                except Exception as e:
                    self.logger.warning(f"Model {model_name} failed for {field_name}: {str(e)}")
                    continue
            
            if best_model:
                # Train the best model on full dataset
                best_model.fit(X_scaled, y)
                
                # Store model info
                trained_models[field_name] = {
                    'model': best_model,
                    'model_type': best_model_name,
                    'score': best_score,
                    'features': list(X.columns)
                }
                
                self.logger.info(f"Best model for {field_name}: {best_model_name} (F1={best_score:.4f})")
            
        return trained_models
    
    def train_compliance_score_predictor(self, df: pd.DataFrame) -> Dict:
        """Train model to predict overall compliance score"""
        
        self.logger.info("Training compliance score predictor")
        
        # Prepare features from all fields
        features = []
        scores = []
        
        for _, row in df.iterrows():
            try:
                # Extract field-level features
                extracted_fields = json.loads(row['extracted_fields']) if row['extracted_fields'] else {}
                
                sample_features = {}
                
                # Aggregate features from all fields
                for field_name in self.field_categories:
                    field_data = extracted_fields.get(field_name, {})
                    text_value = field_data.get('value', '')
                    field_features = self.extract_features_for_field(text_value, field_name)
                    
                    # Add field-level features
                    sample_features.update(field_features)
                    
                    # Add field-level compliance indicators
                    sample_features[f'{field_name}_found'] = float(field_data.get('found', False))
                    sample_features[f'{field_name}_confidence'] = field_data.get('confidence', 0.0)
                
                features.append(sample_features)
                scores.append(row['compliance_score'])
                
            except Exception as e:
                self.logger.warning(f"Error processing row: {str(e)}")
                continue
        
        if not features:
            self.logger.error("No valid features extracted for compliance score prediction")
            return {}
        
        # Convert to DataFrame
        X = pd.DataFrame(features).fillna(0)
        y = np.array(scores)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['compliance_score'] = scaler
        
        # Train regression model for continuous compliance scores
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor.fit(X_scaled, y)
        
        # Evaluate model
        y_pred = regressor.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        self.logger.info(f"Compliance score predictor - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return {
            'model': regressor,
            'model_type': 'random_forest_regressor',
            'mse': mse,
            'r2_score': r2,
            'features': list(X.columns)
        }
    
    def train_violation_predictor(self, df: pd.DataFrame) -> Dict:
        """Train model to predict common violations"""
        
        self.logger.info("Training violation predictor")
        
        # Extract common violations
        violation_counter = Counter()
        
        for _, row in df.iterrows():
            if row['violations']:
                try:
                    violations = json.loads(row['violations'])
                    for violation in violations:
                        violation_counter[violation] += 1
                except:
                    continue
        
        # Get top violations for multi-label classification
        top_violations = [v for v, count in violation_counter.most_common(10) if count >= 5]
        
        if not top_violations:
            self.logger.warning("Insufficient violation data for training")
            return {}
        
        # Prepare multi-label data
        features = []
        violation_labels = []
        
        for _, row in df.iterrows():
            try:
                # Extract features (same as compliance score predictor)
                extracted_fields = json.loads(row['extracted_fields']) if row['extracted_fields'] else {}
                
                sample_features = {}
                for field_name in self.field_categories:
                    field_data = extracted_fields.get(field_name, {})
                    text_value = field_data.get('value', '')
                    field_features = self.extract_features_for_field(text_value, field_name)
                    sample_features.update(field_features)
                
                # Create multi-label target
                sample_violations = json.loads(row['violations']) if row['violations'] else []
                violation_vector = [1 if violation in sample_violations else 0 for violation in top_violations]
                
                features.append(sample_features)
                violation_labels.append(violation_vector)
                
            except Exception as e:
                continue
        
        if not features:
            return {}
        
        X = pd.DataFrame(features).fillna(0)
        y = np.array(violation_labels)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['violations'] = scaler
        
        # Multi-label classifier
        multi_classifier = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=50, random_state=42)
        )
        multi_classifier.fit(X_scaled, y)
        
        return {
            'model': multi_classifier,
            'model_type': 'multi_output_random_forest',
            'violation_labels': top_violations,
            'features': list(X.columns)
        }
    
    def save_models(self, models: Dict, model_version: str = None) -> str:
        """Save trained models to disk"""
        
        if not model_version:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        version_dir = self.model_dir / f"v_{model_version}"
        version_dir.mkdir(exist_ok=True)
        
        # Save each model type
        for model_type, model_data in models.items():
            model_path = version_dir / f"{model_type}_model.pkl"
            joblib.dump(model_data, model_path)
        
        # Save scalers and encoders
        if self.scalers:
            scaler_path = version_dir / "scalers.pkl"
            joblib.dump(self.scalers, scaler_path)
        
        # Save metadata
        metadata = {
            'version': model_version,
            'created_at': datetime.now().isoformat(),
            'model_types': list(models.keys()),
            'field_categories': self.field_categories
        }
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Models saved to {version_dir}")
        return model_version
    
    def train_complete_pipeline(self, min_samples: int = 50) -> Dict[str, Any]:
        """Train complete ML pipeline for Legal Metrology compliance"""
        
        self.logger.info("Starting complete ML training pipeline")
        
        # Get training dataset
        dataset_splits = self.dataset_manager.get_training_dataset()
        
        if len(dataset_splits['train']) < min_samples:
            raise ValueError(f"Insufficient training data: {len(dataset_splits['train'])} samples, need {min_samples}")
        
        train_df = dataset_splits['train']
        val_df = dataset_splits['validation']
        
        self.logger.info(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")
        
        # Prepare training data
        X_data, y_data = self.prepare_training_data(train_df)
        
        # Train different model components
        results = {}
        
        # 1. Field classifiers
        self.logger.info("Training field classifiers...")
        field_models = self.train_field_classifiers(X_data, y_data)
        results['field_classifiers'] = field_models
        
        # 2. Compliance score predictor
        self.logger.info("Training compliance score predictor...")
        score_model = self.train_compliance_score_predictor(train_df)
        results['compliance_score_predictor'] = score_model
        
        # 3. Violation predictor
        self.logger.info("Training violation predictor...")
        violation_model = self.train_violation_predictor(train_df)
        results['violation_predictor'] = violation_model
        
        # Save models
        model_version = self.save_models(results)
        
        # Record training in database
        self._record_training_history(results, dataset_splits, model_version)
        
        self.logger.info(f"Training completed. Model version: {model_version}")
        
        return {
            'model_version': model_version,
            'training_results': results,
            'dataset_info': dataset_splits['metadata']
        }
    
    def _record_training_history(self, results: Dict, dataset_splits: Dict, model_version: str):
        """Record training history in database"""
        
        conn = sqlite3.connect(self.dataset_manager.db_path)
        cursor = conn.cursor()
        
        try:
            # Calculate overall metrics
            total_accuracy = 0
            total_models = 0
            
            field_metrics = {}
            
            for field_name, model_data in results.get('field_classifiers', {}).items():
                field_metrics[field_name] = {
                    'model_type': model_data['model_type'],
                    'f1_score': model_data['score']
                }
                total_accuracy += model_data['score']
                total_models += 1
            
            avg_accuracy = total_accuracy / total_models if total_models > 0 else 0
            
            cursor.execute("""
            INSERT INTO ml_training_history
            (model_version, model_type, training_samples, validation_samples, test_samples,
             accuracy, precision_score, recall_score, f1_score, field_wise_metrics,
             training_duration, hyperparameters, model_path, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_version,
                'ensemble_pipeline',
                len(dataset_splits['train']),
                len(dataset_splits['validation']),
                len(dataset_splits['test']),
                avg_accuracy,
                avg_accuracy,  # Simplified - would need actual precision calculation
                avg_accuracy,  # Simplified - would need actual recall calculation
                avg_accuracy,
                json.dumps(field_metrics),
                0,  # Would need to track actual duration
                json.dumps({'pipeline': 'field_classifiers_ensemble'}),
                f"models/v_{model_version}",
                datetime.now().isoformat(),
                True
            ))
            
            # Deactivate previous models
            cursor.execute("""
            UPDATE ml_training_history 
            SET is_active = FALSE 
            WHERE model_version != ? AND is_active = TRUE
            """, (model_version,))
            
            conn.commit()
            self.logger.info("Training history recorded in database")
            
        except Exception as e:
            self.logger.error(f"Error recording training history: {str(e)}")
        finally:
            conn.close()
    
    def predict_compliance(self, extracted_fields: Dict, model_version: str = None) -> Dict[str, Any]:
        """Predict compliance using trained models"""
        
        if not model_version:
            # Use latest active model
            model_version = self._get_latest_model_version()
        
        if not model_version:
            raise ValueError("No trained models available")
        
        # Load models
        version_dir = self.model_dir / f"v_{model_version}"
        
        try:
            field_classifiers = joblib.load(version_dir / "field_classifiers_model.pkl")
            compliance_predictor = joblib.load(version_dir / "compliance_score_predictor_model.pkl")
            scalers = joblib.load(version_dir / "scalers.pkl")
        except FileNotFoundError as e:
            raise ValueError(f"Model files not found: {str(e)}")
        
        predictions = {}
        field_predictions = {}
        
        # Predict for each field
        for field_name in self.field_categories:
            if field_name in field_classifiers:
                field_data = extracted_fields.get(field_name, {})
                text_value = field_data.get('value', '')
                
                # Extract features
                features = self.extract_features_for_field(text_value, field_name)
                feature_df = pd.DataFrame([features]).fillna(0)
                
                # Ensure feature order matches training
                model_features = field_classifiers[field_name]['features']
                for feature in model_features:
                    if feature not in feature_df.columns:
                        feature_df[feature] = 0
                feature_df = feature_df[model_features]
                
                # Scale and predict
                scaler = scalers[field_name]
                X_scaled = scaler.transform(feature_df)
                
                model = field_classifiers[field_name]['model']
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0]
                
                field_predictions[field_name] = {
                    'compliant': bool(prediction),
                    'confidence': float(max(probability)),
                    'probability_compliant': float(probability[1]) if len(probability) > 1 else float(probability[0])
                }
        
        # Predict overall compliance score
        overall_features = {}
        for field_name in self.field_categories:
            field_data = extracted_fields.get(field_name, {})
            text_value = field_data.get('value', '')
            field_features = self.extract_features_for_field(text_value, field_name)
            overall_features.update(field_features)
            
            # Add field-level indicators
            overall_features[f'{field_name}_found'] = float(field_data.get('found', False))
            overall_features[f'{field_name}_confidence'] = field_data.get('confidence', 0.0)
        
        # Predict compliance score
        if compliance_predictor:
            score_features = pd.DataFrame([overall_features]).fillna(0)
            score_model_features = compliance_predictor['features']
            
            for feature in score_model_features:
                if feature not in score_features.columns:
                    score_features[feature] = 0
            score_features = score_features[score_model_features]
            
            score_scaler = scalers['compliance_score']
            score_scaled = score_scaler.transform(score_features)
            
            predicted_score = compliance_predictor['model'].predict(score_scaled)[0]
        else:
            # Fallback: average of field predictions
            field_scores = [fp['probability_compliant'] for fp in field_predictions.values()]
            predicted_score = (sum(field_scores) / len(field_scores) * 100) if field_scores else 0
        
        return {
            'field_predictions': field_predictions,
            'overall_compliance_score': float(predicted_score),
            'model_version': model_version,
            'prediction_confidence': float(np.mean([fp['confidence'] for fp in field_predictions.values()]))
        }
    
    def _get_latest_model_version(self) -> Optional[str]:
        """Get latest active model version from database"""
        
        conn = sqlite3.connect(self.dataset_manager.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            SELECT model_version FROM ml_training_history 
            WHERE is_active = TRUE 
            ORDER BY created_at DESC 
            LIMIT 1
            """)
            
            result = cursor.fetchone()
            return result[0] if result else None
            
        except:
            return None
        finally:
            conn.close()

# Alias for easy import
MLTrainer = ComplianceMLTrainer
