import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd

# Import existing components
from compliance_engine import LegalMetrologyRuleEngine
from vision_processor import VisionProcessor
from ml_trainer import ComplianceMLTrainer
from dataset_manager import ComplianceDatasetManager

class CascadingComplianceAnalyzer:
    """
    Advanced cascading analysis system for Legal Metrology compliance
    
    Analysis Flow:
    1. Rule-based analysis (fast, deterministic)
    2. ML Model analysis (if rule-based fails or has low confidence)
    3. Gemini API analysis (if both above fail or need validation)
    4. Combine and return best result
    """
    
    def __init__(self, dataset_manager: Optional[ComplianceDatasetManager] = None):
        self.setup_logging()
        
        # Initialize components
        self.rule_engine = LegalMetrologyRuleEngine()
        self.vision_processor = VisionProcessor()
        self.dataset_manager = dataset_manager or ComplianceDatasetManager()
        self.ml_trainer = ComplianceMLTrainer(self.dataset_manager)
        
        # Analysis thresholds
        self.confidence_threshold = 0.7  # Minimum confidence to skip next layer
        self.rule_based_threshold = 0.8   # Rule-based confidence threshold
        self.ml_model_threshold = 0.75    # ML model confidence threshold
        
        # Track analysis steps
        self.analysis_steps = []
        
    def setup_logging(self):
        """Setup logging for cascading analyzer"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_compliance(self, image_input, use_advanced_flow: bool = True) -> Dict[str, Any]:
        """
        Main cascading analysis function
        
        Flow: Rule-based → ML Model (if rule-based fails) → Gemini API (if both fail) → Best result
        
        Args:
            image_input: Image file or path
            use_advanced_flow: Whether to use cascading analysis or just Gemini
            
        Returns:
            Combined analysis results with best result displayed
        """
        self.logger.info("Starting cascading compliance analysis")
        
        if not use_advanced_flow:
            return self._gemini_only_analysis(image_input)
        
        final_result = {
            'success': True,
            'analysis_method': 'cascading',
            'steps_performed': [],
            'confidence_scores': {},
            'compliance_data': {},
            'raw_responses': {},
            'best_result_source': '',
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Try Rule-based Analysis first
            self.logger.info("Step 1: Attempting rule-based analysis")
            rule_result = self._rule_based_analysis(image_input)
            final_result['steps_performed'].append('rule_based')
            final_result['raw_responses']['rule_based'] = rule_result
            final_result['confidence_scores']['rule_based'] = rule_result.get('confidence', 0.0)
            
            # Check if rule-based analysis succeeded
            if rule_result.get('success', False):
                self.logger.info("Rule-based analysis succeeded - using this result")
                final_result['compliance_data'] = rule_result['compliance_data']
                final_result['best_result_source'] = 'rule_based'
            else:
                self.logger.info("Rule-based analysis failed - trying ML model")
                
                # Step 2: Try ML Model Analysis
                ml_result = self._ml_model_analysis(image_input, rule_result)
                
                if ml_result and ml_result.get('success', False):
                    final_result['steps_performed'].append('ml_model')
                    final_result['raw_responses']['ml_model'] = ml_result
                    final_result['confidence_scores']['ml_model'] = ml_result.get('confidence', 0.0)
                    
                    self.logger.info("ML model analysis succeeded - using this result")
                    final_result['compliance_data'] = ml_result['compliance_data']
                    final_result['best_result_source'] = 'ml_model'
                else:
                    self.logger.info("ML model analysis failed - trying Gemini API")
                    
                    # Step 3: Try Gemini API Analysis as last resort
                    gemini_result = self._gemini_api_analysis(image_input)
                    
                    if gemini_result and gemini_result.get('success', False):
                        final_result['steps_performed'].append('gemini_api')
                        final_result['raw_responses']['gemini_api'] = gemini_result
                        final_result['confidence_scores']['gemini_api'] = gemini_result.get('confidence', 0.9)
                        
                        self.logger.info("Gemini API analysis succeeded - using this result")
                        final_result['compliance_data'] = gemini_result['compliance_data']
                        final_result['best_result_source'] = 'gemini_api'
                    else:
                        # All methods failed - return error
                        self.logger.error("All analysis methods failed")
                        return {
                            'success': False,
                            'error': 'All analysis methods (rule-based, ML model, Gemini API) failed',
                            'analysis_method': 'cascading',
                            'steps_performed': final_result['steps_performed']
                        }
            
            # Add validation using rule engine on the best result
            validation_results = self.rule_engine.validate_compliance(final_result['compliance_data'])
            final_result['validation_results'] = validation_results
            final_result['compliance_report'] = self.rule_engine.generate_compliance_report(validation_results)
            
            self.logger.info(f"Cascading analysis completed. Using result from: {final_result['best_result_source']}")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Cascading analysis failed: {str(e)}")
            return {
                'success': False,
                'error': f"Cascading analysis failed: {str(e)}",
                'analysis_method': 'cascading',
                'steps_performed': final_result.get('steps_performed', [])
            }
    
    def _rule_based_analysis(self, image_input) -> Dict[str, Any]:
        """
        Rule-based analysis using OCR + pattern matching
        Fast but may have lower accuracy for complex images
        """
        self.logger.info("Performing rule-based analysis")
        
        try:
            # Use basic OCR (you might want to integrate pytesseract or similar)
            extracted_text = self._extract_text_basic(image_input)
            
            if not extracted_text:
                return {
                    'success': False,
                    'confidence': 0.0,
                    'error': 'No text extracted from image',
                    'method': 'rule_based_ocr'
                }
            
            # Apply rule-based field detection
            compliance_data = self._rule_based_field_detection(extracted_text)
            
            # Calculate confidence based on field completeness and pattern matches
            confidence = self._calculate_rule_based_confidence(compliance_data, extracted_text)
            
            return {
                'success': True,
                'confidence': confidence,
                'compliance_data': compliance_data,
                'extracted_text': extracted_text,
                'method': 'rule_based_pattern_matching'
            }
            
        except Exception as e:
            self.logger.error(f"Rule-based analysis failed: {str(e)}")
            return {
                'success': False,
                'confidence': 0.0,
                'error': f"Rule-based analysis failed: {str(e)}",
                'method': 'rule_based_ocr'
            }
    
    def _ml_model_analysis(self, image_input, rule_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        ML model analysis using trained compliance models
        """
        self.logger.info("Performing ML model analysis")
        
        try:
            # Check if trained models are available
            latest_model = self.ml_trainer._get_latest_model_version()
            if not latest_model:
                self.logger.info("No trained ML models available, skipping ML analysis")
                return None
            
            # Use extracted text from rule-based analysis or extract new
            extracted_text = rule_result.get('extracted_text', '')
            if not extracted_text:
                extracted_text = self._extract_text_basic(image_input)
            
            # Prepare features for ML model
            extracted_fields = rule_result.get('compliance_data', {})
            
            # Use ML model to predict compliance
            ml_predictions = self.ml_trainer.predict_compliance(extracted_fields, latest_model)
            
            # Convert ML predictions to compliance data format
            compliance_data = self._convert_ml_predictions_to_compliance_data(
                ml_predictions, extracted_fields, extracted_text
            )
            
            # Calculate overall confidence from ML predictions
            confidence = ml_predictions.get('prediction_confidence', 0.0)
            
            return {
                'success': True,
                'confidence': confidence,
                'compliance_data': compliance_data,
                'ml_predictions': ml_predictions,
                'model_version': latest_model,
                'method': 'ml_classification'
            }
            
        except Exception as e:
            self.logger.error(f"ML model analysis failed: {str(e)}")
            return {
                'success': False,
                'confidence': 0.0,
                'error': f"ML model analysis failed: {str(e)}",
                'method': 'ml_classification'
            }
    
    def _gemini_api_analysis(self, image_input) -> Dict[str, Any]:
        """
        Gemini API analysis - highest accuracy but most expensive
        """
        self.logger.info("Performing Gemini API analysis")
        
        try:
            # Use existing vision processor
            vision_results = self.vision_processor.analyze_product_compliance(image_input)
            
            if not vision_results.get('success', False):
                return {
                    'success': False,
                    'confidence': 0.0,
                    'error': vision_results.get('error', 'Gemini API analysis failed'),
                    'method': 'gemini_vision_api'
                }
            
            # Add confidence score (Gemini typically has high confidence)
            compliance_data = vision_results.get('compliance_data', {})
            confidence = self._calculate_gemini_confidence(compliance_data)
            
            return {
                'success': True,
                'confidence': confidence,
                'compliance_data': compliance_data,
                'raw_response': vision_results.get('raw_response', ''),
                'method': 'gemini_vision_api'
            }
            
        except Exception as e:
            self.logger.error(f"Gemini API analysis failed: {str(e)}")
            return {
                'success': False,
                'confidence': 0.0,
                'error': f"Gemini API analysis failed: {str(e)}",
                'method': 'gemini_vision_api'
            }
    
    def _extract_text_basic(self, image_input) -> str:
        """
        Basic text extraction using simple OCR or pattern matching
        This is a placeholder - you might want to integrate pytesseract or similar
        """
        # For now, return empty string - this would be replaced with actual OCR
        # You could integrate pytesseract, EasyOCR, or PaddleOCR here
        self.logger.info("Basic text extraction (placeholder)")
        return ""
    
    def _rule_based_field_detection(self, extracted_text: str) -> Dict[str, Any]:
        """
        Rule-based field detection using regex patterns and keywords
        """
        text_lower = extracted_text.lower()
        
        fields = {
            "manufacturer": {"found": False, "value": "", "compliance": "Fail"},
            "net_quantity": {"found": False, "value": "", "compliance": "Fail"},
            "mrp": {"found": False, "value": "", "compliance": "Fail"},
            "consumer_care": {"found": False, "value": "", "compliance": "Fail"},
            "mfg_date": {"found": False, "value": "", "compliance": "Fail"},
            "country_origin": {"found": False, "value": "", "compliance": "Fail"},
            "product_name": {"found": False, "value": "", "compliance": "Pass"},
        }
        
        # Apply existing rule-based logic from vision_processor
        violations = []
        
        # Manufacturer detection
        if any(word in text_lower for word in ["manufacturer", "mfg", "made by", "packed by"]):
            fields["manufacturer"]["found"] = True
            fields["manufacturer"]["compliance"] = "Pass"
        else:
            violations.append("Manufacturer details not found")
        
        # MRP detection
        if any(word in text_lower for word in ["mrp", "price", "₹", "rs", "inr"]):
            fields["mrp"]["found"] = True
            fields["mrp"]["compliance"] = "Pass"
        else:
            violations.append("MRP not clearly visible")
        
        # Net quantity detection
        if any(unit in text_lower for unit in ["ml", "g", "kg", "l", "gm", "gram", "litre", "liter"]):
            fields["net_quantity"]["found"] = True
            fields["net_quantity"]["compliance"] = "Pass"
        else:
            violations.append("Net quantity not specified")
        
        # Consumer care detection
        if any(word in text_lower for word in ["phone", "email", "contact", "customer care", "toll free"]):
            fields["consumer_care"]["found"] = True
            fields["consumer_care"]["compliance"] = "Pass"
        else:
            violations.append("Consumer care details missing")
        
        # Manufacturing date detection
        if any(word in text_lower for word in ["mfd", "manufactured", "exp", "expiry", "best before"]):
            fields["mfg_date"]["found"] = True
            fields["mfg_date"]["compliance"] = "Pass"
        else:
            violations.append("Manufacturing/Expiry date not found")
        
        # Country of origin detection
        if any(word in text_lower for word in ["india", "made in", "origin", "country"]):
            fields["country_origin"]["found"] = True
            fields["country_origin"]["compliance"] = "Pass"
        else:
            violations.append("Country of origin not specified")
        
        # Calculate compliance score
        found_count = sum(1 for field in fields.values() if field["found"])
        compliance_score = int((found_count / len(fields)) * 100)
        
        fields.update({
            "compliance_score": compliance_score,
            "violations": violations,
            "overall_status": "Compliant" if compliance_score >= 85 else "Non-Compliant"
        })
        
        return fields
    
    def _calculate_rule_based_confidence(self, compliance_data: Dict[str, Any], extracted_text: str) -> float:
        """
        Calculate confidence score for rule-based analysis
        """
        if not extracted_text:
            return 0.0
        
        # Base confidence on text length and field detection
        text_quality_score = min(len(extracted_text) / 500, 1.0)  # Normalize text length
        
        # Field detection score
        fields_found = sum(1 for field in compliance_data.values() 
                          if isinstance(field, dict) and field.get("found", False))
        total_fields = 7  # Number of mandatory fields
        field_score = fields_found / total_fields
        
        # Pattern match strength (placeholder - would be more sophisticated)
        pattern_score = 0.5  # Default medium confidence for patterns
        
        # Combined confidence
        confidence = (text_quality_score * 0.3 + field_score * 0.5 + pattern_score * 0.2)
        
        return confidence
    
    def _calculate_gemini_confidence(self, compliance_data: Dict[str, Any]) -> float:
        """
        Calculate confidence score for Gemini API results
        """
        if not compliance_data:
            return 0.0
        
        # Gemini is generally high confidence, base on completeness
        fields_found = sum(1 for field in compliance_data.values() 
                          if isinstance(field, dict) and field.get("found", False))
        total_fields = 7
        
        # Gemini has high base confidence
        base_confidence = 0.8
        field_bonus = (fields_found / total_fields) * 0.2
        
        return base_confidence + field_bonus
    
    def _convert_ml_predictions_to_compliance_data(self, ml_predictions: Dict[str, Any], 
                                                   extracted_fields: Dict[str, Any], 
                                                   extracted_text: str) -> Dict[str, Any]:
        """
        Convert ML model predictions to compliance data format
        """
        field_predictions = ml_predictions.get('field_predictions', {})
        
        compliance_data = {}
        violations = []
        
        for field_name in ['manufacturer', 'net_quantity', 'mrp', 'consumer_care', 
                          'mfg_date', 'country_origin', 'product_name']:
            
            field_pred = field_predictions.get(field_name, {})
            original_field = extracted_fields.get(field_name, {})
            
            compliance_data[field_name] = {
                'found': field_pred.get('compliant', False),
                'value': original_field.get('value', ''),
                'compliance': 'Pass' if field_pred.get('compliant', False) else 'Fail',
                'confidence': field_pred.get('confidence', 0.0)
            }
            
            if not field_pred.get('compliant', False):
                violations.append(f"{field_name.replace('_', ' ').title()} is mandatory but not found")
        
        # Calculate overall compliance score
        compliant_fields = sum(1 for field in compliance_data.values() if field.get('compliance') == 'Pass')
        compliance_score = int((compliant_fields / len(compliance_data)) * 100)
        
        compliance_data.update({
            'compliance_score': ml_predictions.get('overall_compliance_score', compliance_score),
            'violations': violations,
            'overall_status': 'Compliant' if compliance_score >= 85 else 'Non-Compliant'
        })
        
        return compliance_data
    
    
    def _get_default_compliance_data(self) -> Dict[str, Any]:
        """
        Return default compliance data structure when all analyses fail
        """
        return {
            "manufacturer": {"found": False, "value": "", "compliance": "Fail"},
            "net_quantity": {"found": False, "value": "", "compliance": "Fail"},
            "mrp": {"found": False, "value": "", "compliance": "Fail"},
            "consumer_care": {"found": False, "value": "", "compliance": "Fail"},
            "mfg_date": {"found": False, "value": "", "compliance": "Fail"},
            "country_origin": {"found": False, "value": "", "compliance": "Fail"},
            "product_name": {"found": False, "value": "", "compliance": "Fail"},
            "compliance_score": 0,
            "violations": ["Analysis failed - unable to extract compliance data"],
            "overall_status": "Non-Compliant"
        }
    
    def _gemini_only_analysis(self, image_input) -> Dict[str, Any]:
        """
        Fallback to original Gemini-only analysis
        """
        self.logger.info("Performing Gemini-only analysis (original method)")
        
        vision_results = self.vision_processor.analyze_product_compliance(image_input)
        
        if not vision_results.get('success', False):
            return vision_results
        
        compliance_data = vision_results.get('compliance_data', {})
        validation_results = self.rule_engine.validate_compliance(compliance_data)
        compliance_report = self.rule_engine.generate_compliance_report(validation_results)
        
        return {
            'success': True,
            'analysis_method': 'gemini_only',
            'vision_results': vision_results,
            'validation_results': validation_results,
            'compliance_report': compliance_report,
            'compliance_data': compliance_data,
            'raw_data': compliance_data
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get summary of the last analysis performed
        """
        return {
            'steps_performed': self.analysis_steps,
            'total_steps': len(self.analysis_steps),
            'analysis_timestamp': datetime.now().isoformat()
        }
