import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
import hashlib
import base64
from PIL import Image
import io
import numpy as np
import re
from collections import defaultdict
import logging

class ComplianceDatasetManager:
    """
    Advanced dataset manager for Legal Metrology compliance ML training
    Based on Legal Metrology (Packaged Commodities) Rules, 2011
    """
    
    def __init__(self, db_path: str = "compliance_dataset.db"):
        self.db_path = db_path
        self.setup_logging()
        self.init_database()
        self.load_legal_metrology_rules()
        
    def setup_logging(self):
        """Setup logging for ML operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('compliance_ml.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def init_database(self):
        """Initialize comprehensive SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced compliance samples table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS compliance_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_hash TEXT UNIQUE,
            image_data BLOB,
            image_metadata TEXT,
            extracted_fields TEXT,
            ground_truth_labels TEXT,
            user_corrections TEXT,
            compliance_score REAL,
            field_scores TEXT,
            violations TEXT,
            model_predictions TEXT,
            confidence_scores TEXT,
            feedback_score INTEGER,
            validation_errors TEXT,
            price_variations TEXT,
            date_variations TEXT,
            address_variations TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            verified BOOLEAN DEFAULT FALSE,
            data_source TEXT,
            annotation_quality REAL,
            training_weight REAL DEFAULT 1.0
        )
        """)
        
        # Legal Metrology rules knowledge base with detailed patterns
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS legal_metrology_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_section TEXT,
            rule_number TEXT,
            rule_category TEXT,
            rule_name TEXT,
            rule_description TEXT,
            mandatory_status TEXT,
            validation_patterns TEXT,
            valid_examples TEXT,
            invalid_examples TEXT,
            common_violations TEXT,
            severity_level TEXT,
            penalty_amount TEXT,
            regex_patterns TEXT,
            validation_logic TEXT,
            created_at TIMESTAMP
        )
        """)
        
        # ML model training history and performance
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT,
            model_type TEXT,
            training_samples INTEGER,
            validation_samples INTEGER,
            test_samples INTEGER,
            accuracy REAL,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            field_wise_metrics TEXT,
            confusion_matrix TEXT,
            training_duration REAL,
            hyperparameters TEXT,
            feature_importance TEXT,
            model_path TEXT,
            created_at TIMESTAMP,
            is_active BOOLEAN DEFAULT FALSE
        )
        """)
        
        # User feedback with detailed annotations
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER,
            field_name TEXT,
            original_value TEXT,
            corrected_value TEXT,
            correction_type TEXT,
            feedback_category TEXT,
            user_id TEXT,
            confidence_score REAL,
            annotation_notes TEXT,
            review_status TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (sample_id) REFERENCES compliance_samples (id)
        )
        """)
        
        # Price variations tracking
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_variations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER,
            detected_price TEXT,
            standardized_price TEXT,
            currency_symbol TEXT,
            tax_inclusion_status TEXT,
            format_variations TEXT,
            validation_status TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (sample_id) REFERENCES compliance_samples (id)
        )
        """)
        
        # Date format variations
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS date_variations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER,
            detected_date TEXT,
            standardized_date TEXT,
            date_format TEXT,
            date_type TEXT,
            validation_status TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (sample_id) REFERENCES compliance_samples (id)
        )
        """)
        
        # Address/Contact variations
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS address_variations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER,
            field_type TEXT,
            detected_text TEXT,
            standardized_format TEXT,
            contact_type TEXT,
            completeness_score REAL,
            validation_issues TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (sample_id) REFERENCES compliance_samples (id)
        )
        """)
        
        conn.commit()
        conn.close()
        
    def load_legal_metrology_rules(self):
        """Load comprehensive Legal Metrology Rules 2011 into database"""
        
        rules_data = [
            # Rule 6 - Mandatory Declarations
            {
                "rule_section": "Chapter II",
                "rule_number": "Rule 6",
                "rule_category": "manufacturer",
                "rule_name": "Name and Address of Manufacturer/Packer/Importer",
                "rule_description": "Complete address including postal address, street, number, city, state, PIN code",
                "mandatory_status": "Mandatory",
                "validation_patterns": json.dumps([
                    "Must include complete address",
                    "PIN code mandatory (6 digits)",
                    "Name of manufacturer/packer/importer",
                    "If imported, importer address required"
                ]),
                "valid_examples": json.dumps([
                    "ABC Foods Pvt Ltd, Plot 123, Industrial Area, Mumbai - 400001",
                    "XYZ Manufacturers, Building 45, Sector 18, Gurgaon - 122015",
                    "Imported by: DEF Traders, 789 Commerce Street, Delhi - 110001"
                ]),
                "invalid_examples": json.dumps([
                    "ABC Foods (incomplete)",
                    "Mumbai Company",
                    "No address provided",
                    "Address without PIN code"
                ]),
                "common_violations": json.dumps([
                    "Missing PIN code",
                    "Incomplete address",
                    "Only company name",
                    "Unclear manufacturer identity"
                ]),
                "severity_level": "Critical",
                "penalty_amount": "Rs. 2000",
                "regex_patterns": json.dumps([
                    r"\b\d{6}\b",  # PIN code pattern
                    r"(pvt|private|ltd|limited|inc|company|co)",  # Company indicators
                    r"(plot|building|street|road|area|sector)",  # Address components
                ]),
                "validation_logic": json.dumps({
                    "min_length": 20,
                    "requires_pin": True,
                    "requires_address_components": True,
                    "company_name_required": True
                })
            },
            {
                "rule_section": "Chapter II",
                "rule_number": "Rule 6",
                "rule_category": "net_quantity",
                "rule_name": "Net Quantity Declaration",
                "rule_description": "Net quantity in standard units (weight/measure/number)",
                "mandatory_status": "Mandatory",
                "validation_patterns": json.dumps([
                    "Must use standard units (g, kg, ml, l, pieces)",
                    "Numeric value required",
                    "No misleading terms like 'approximately'",
                    "Clear quantity declaration"
                ]),
                "valid_examples": json.dumps([
                    "500g", "1kg", "250ml", "2 litres", "12 pieces",
                    "Net Wt: 750gm", "Contents: 1.5L", "Count: 24 units"
                ]),
                "invalid_examples": json.dumps([
                    "500 (no unit)", "Large pack", "Family size",
                    "Approximately 1kg", "About 250ml"
                ]),
                "common_violations": json.dumps([
                    "Missing unit specification",
                    "Non-standard units",
                    "Ambiguous quantity terms",
                    "No numeric value"
                ]),
                "severity_level": "Critical",
                "penalty_amount": "Rs. 2000",
                "regex_patterns": json.dumps([
                    r"\d+\.?\d*\s*(g|gm|gram|kg|ml|l|litre|liter|pieces|pcs|nos|units)\b",
                    r"net\s*(wt|weight|qty|quantity)",
                    r"contents?\s*:?\s*\d+"
                ]),
                "validation_logic": json.dumps({
                    "requires_numeric": True,
                    "requires_unit": True,
                    "valid_units": ["g", "gm", "gram", "kg", "ml", "l", "litre", "liter", "pieces", "pcs", "nos", "units"],
                    "prohibit_qualifiers": ["approximately", "about", "around", "roughly"]
                })
            },
            {
                "rule_section": "Chapter II", 
                "rule_number": "Rule 6",
                "rule_category": "mrp",
                "rule_name": "Maximum Retail Price",
                "rule_description": "MRP inclusive of all taxes with proper format",
                "mandatory_status": "Mandatory",
                "validation_patterns": json.dumps([
                    "Must include 'inclusive of all taxes' or 'incl. of all taxes'",
                    "Price with currency symbol (₹ or Rs.)",
                    "Format: MRP Rs.XXX inclusive of all taxes",
                    "Fraction handling as per rules"
                ]),
                "valid_examples": json.dumps([
                    "MRP ₹125 (Incl. of all taxes)",
                    "Maximum Retail Price Rs.299/- inclusive of all taxes",
                    "MRP: Rs.50 incl. of all taxes"
                ]),
                "invalid_examples": json.dumps([
                    "₹125", "Price: 299", "MRP 125",
                    "Rs.50 + taxes", "Price excluding taxes"
                ]),
                "common_violations": json.dumps([
                    "Missing tax inclusion clause",
                    "No currency symbol",
                    "Incorrect format",
                    "Additional charges mentioned"
                ]),
                "severity_level": "Critical",
                "penalty_amount": "Rs. 2000",
                "regex_patterns": json.dumps([
                    r"mrp\s*:?\s*[₹rs]\s*\d+.*incl",
                    r"maximum\s*retail\s*price.*incl",
                    r"[₹rs]\s*\d+.*inclusive.*tax"
                ]),
                "validation_logic": json.dumps({
                    "requires_currency": True,
                    "requires_tax_clause": True,
                    "currency_symbols": ["₹", "rs", "inr", "rupees"],
                    "tax_keywords": ["inclusive", "incl", "including", "taxes", "tax"]
                })
            },
            {
                "rule_section": "Chapter II",
                "rule_number": "Rule 6", 
                "rule_category": "consumer_care",
                "rule_name": "Consumer Care Details",
                "rule_description": "Contact details for consumer complaints (phone/email/address)",
                "mandatory_status": "Mandatory",
                "validation_patterns": json.dumps([
                    "Phone number OR email OR address for complaints",
                    "Clear labeling as customer care/consumer care",
                    "Functional contact information",
                    "Toll-free numbers acceptable"
                ]),
                "valid_examples": json.dumps([
                    "Customer Care: 1800-123-456",
                    "Consumer Complaints: complaints@company.com", 
                    "For complaints: Customer Care, Plot 123, Mumbai",
                    "Toll Free: 1800-COMPANY"
                ]),
                "invalid_examples": json.dumps([
                    "See website", "Contact dealer",
                    "No contact provided", "Invalid phone format"
                ]),
                "common_violations": json.dumps([
                    "No contact details provided",
                    "Invalid contact format",
                    "Unclear complaint mechanism",
                    "Only website reference"
                ]),
                "severity_level": "Critical", 
                "penalty_amount": "Rs. 2000",
                "regex_patterns": json.dumps([
                    r"\b\d{10}\b",  # 10 digit phone
                    r"\+91\s*\d{10}",  # International format
                    r"1800\s*\d{6,7}",  # Toll free
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email
                ]),
                "validation_logic": json.dumps({
                    "requires_contact": True,
                    "valid_contact_types": ["phone", "email", "address"],
                    "phone_patterns": [r"\b\d{10}\b", r"\+91\s*\d{10}", r"1800\s*\d{6,7}"],
                    "email_pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                })
            },
            {
                "rule_section": "Chapter II",
                "rule_number": "Rule 6",
                "rule_category": "mfg_date", 
                "rule_name": "Manufacturing/Packing Date",
                "rule_description": "Month and year of manufacture/packing/import",
                "mandatory_status": "Mandatory",
                "validation_patterns": json.dumps([
                    "Month and year mandatory",
                    "Clear labeling (MFD/Mfg Date/Best Before)",
                    "Proper date format",
                    "Not applicable to certain exempted items"
                ]),
                "valid_examples": json.dumps([
                    "MFD: 10/2024", "Manufactured: Oct 2024",
                    "Mfg Date: October 2024", "Best Before: 12/2025",
                    "Packed: Nov 2024"
                ]),
                "invalid_examples": json.dumps([
                    "Fresh", "Recent", "New stock",
                    "Made yesterday", "Latest batch"
                ]),
                "common_violations": json.dumps([
                    "No date provided",
                    "Unclear date format", 
                    "Missing month/year",
                    "Vague date references"
                ]),
                "severity_level": "High",
                "penalty_amount": "Rs. 2000", 
                "regex_patterns": json.dumps([
                    r"(mfd|manufactured|mfg\s*date|best\s*before|exp|expiry)",
                    r"\d{1,2}/\d{4}",  # MM/YYYY
                    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{4}"  # Month Year
                ]),
                "validation_logic": json.dumps({
                    "requires_date": True,
                    "date_formats": ["MM/YYYY", "Month YYYY", "DD/MM/YYYY"],
                    "date_labels": ["mfd", "manufactured", "mfg date", "best before", "expiry", "exp"],
                    "exemptions": ["bidis", "incense sticks", "LPG cylinders"]
                })
            },
            {
                "rule_section": "Chapter II",
                "rule_number": "Rule 6",
                "rule_category": "country_origin",
                "rule_name": "Country of Origin", 
                "rule_description": "Country where product is manufactured/made/imported from",
                "mandatory_status": "Mandatory",
                "validation_patterns": json.dumps([
                    "Clear country identification",
                    "Format: 'Made in [Country]' or 'Country of Origin: [Country]'",
                    "For imports: 'Imported from [Country]'",
                    "Must be specific country name"
                ]),
                "valid_examples": json.dumps([
                    "Made in India", "Country of Origin: India",
                    "Manufactured in Thailand", "Imported from China",
                    "Origin: Indonesia"
                ]),
                "invalid_examples": json.dumps([
                    "Local", "Domestic", "Foreign",
                    "Asian product", "Made locally"
                ]),
                "common_violations": json.dumps([
                    "No origin specified",
                    "Vague origin terms",
                    "Missing country name",
                    "Unclear origin statement"
                ]),
                "severity_level": "Medium",
                "penalty_amount": "Rs. 2000",
                "regex_patterns": json.dumps([
                    r"made\s*in\s*\w+",
                    r"country\s*of\s*origin\s*:?\s*\w+", 
                    r"origin\s*:?\s*\w+",
                    r"manufactured\s*in\s*\w+",
                    r"imported\s*from\s*\w+"
                ]),
                "validation_logic": json.dumps({
                    "requires_country": True,
                    "valid_patterns": ["made in", "country of origin", "origin", "manufactured in", "imported from"],
                    "requires_specific_country": True
                })
            },
            {
                "rule_section": "Chapter II",
                "rule_number": "Rule 6",
                "rule_category": "product_name",
                "rule_name": "Product Name and Brand",
                "rule_description": "Common or generic name of commodity, brand name",
                "mandatory_status": "Mandatory",
                "validation_patterns": json.dumps([
                    "Clear product identification",
                    "Generic/common name required",
                    "Brand name if applicable",
                    "For multi-product packages: list all products"
                ]),
                "valid_examples": json.dumps([
                    "Premium Basmati Rice", "XYZ Brand Biscuits",
                    "ABC Shampoo", "DEF Cooking Oil"
                ]),
                "invalid_examples": json.dumps([
                    "Product", "Item", "Goods",
                    "Package contents", ""
                ]),
                "common_violations": json.dumps([
                    "Generic terms only",
                    "Missing product name",
                    "Unclear identification",
                    "No brand/product distinction"
                ]),
                "severity_level": "Low",
                "penalty_amount": "Rs. 2000",
                "regex_patterns": json.dumps([
                    r".{3,}",  # Minimum 3 characters
                    r"\b(brand|premium|organic|natural)\b"
                ]),
                "validation_logic": json.dumps({
                    "min_length": 3,
                    "requires_specific_name": True,
                    "avoid_generic_terms": ["product", "item", "goods", "package"]
                })
            }
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if rules already loaded
        cursor.execute("SELECT COUNT(*) FROM legal_metrology_rules")
        count = cursor.fetchone()[0]
        
        if count == 0:
            for rule in rules_data:
                cursor.execute("""
                INSERT INTO legal_metrology_rules 
                (rule_section, rule_number, rule_category, rule_name, rule_description,
                 mandatory_status, validation_patterns, valid_examples, invalid_examples,
                 common_violations, severity_level, penalty_amount, regex_patterns,
                 validation_logic, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rule["rule_section"], rule["rule_number"], rule["rule_category"],
                    rule["rule_name"], rule["rule_description"], rule["mandatory_status"],
                    rule["validation_patterns"], rule["valid_examples"], rule["invalid_examples"],
                    rule["common_violations"], rule["severity_level"], rule["penalty_amount"],
                    rule["regex_patterns"], rule["validation_logic"], datetime.now().isoformat()
                ))
            
            self.logger.info(f"Loaded {len(rules_data)} Legal Metrology rules into database")
        
        conn.commit()
        conn.close()
    
    def add_compliance_sample(self, image_data: bytes, extracted_fields: Dict,
                            predictions: Dict, compliance_result: Dict, 
                            data_source: str = "user_upload") -> str:
        """Add comprehensive compliance sample with ML features"""
        
        image_hash = hashlib.sha256(image_data).hexdigest()
        
        # Extract image metadata
        try:
            image = Image.open(io.BytesIO(image_data))
            metadata = {
                "width": image.width,
                "height": image.height, 
                "format": image.format,
                "mode": image.mode,
                "size_bytes": len(image_data),
                "aspect_ratio": image.width / image.height
            }
        except:
            metadata = {"size_bytes": len(image_data), "error": "Could not process image"}
        
        # Calculate field-wise scores
        field_scores = {}
        for field, data in extracted_fields.items():
            if isinstance(data, dict):
                confidence = data.get('confidence', 0.0)
                found = data.get('found', False)
                field_scores[field] = {
                    'confidence': confidence,
                    'found': found,
                    'score': confidence if found else 0.0
                }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO compliance_samples 
            (image_hash, image_data, image_metadata, extracted_fields, 
             model_predictions, field_scores, compliance_score, violations,
             confidence_scores, created_at, updated_at, data_source, training_weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_hash, image_data, json.dumps(metadata), 
                json.dumps(extracted_fields), json.dumps(predictions),
                json.dumps(field_scores), compliance_result.get('compliance_score', 0),
                json.dumps(compliance_result.get('violations', [])),
                json.dumps(predictions.get('confidence_scores', {})),
                datetime.now().isoformat(), datetime.now().isoformat(),
                data_source, 1.0
            ))
            
            sample_id = cursor.lastrowid
            
            # Store price, date, and address variations
            self._store_variations(cursor, sample_id, extracted_fields)
            
            conn.commit()
            self.logger.info(f"Added compliance sample: {image_hash}")
            return image_hash
            
        except Exception as e:
            self.logger.error(f"Error adding sample: {str(e)}")
            return None
        finally:
            conn.close()
    
    def _store_variations(self, cursor, sample_id: int, extracted_fields: Dict):
        """Store price, date, and address variations for ML training"""
        
        for field_name, field_data in extracted_fields.items():
            if not isinstance(field_data, dict):
                continue
                
            value = field_data.get('value', '')
            
            # Store price variations
            if field_name == 'mrp' and value:
                price_info = self._analyze_price_format(value)
                cursor.execute("""
                INSERT INTO price_variations 
                (sample_id, detected_price, standardized_price, currency_symbol,
                 tax_inclusion_status, format_variations, validation_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sample_id, value, price_info['standardized'],
                    price_info['currency'], price_info['tax_status'],
                    json.dumps(price_info['variations']), price_info['valid'],
                    datetime.now().isoformat()
                ))
            
            # Store date variations  
            if field_name == 'mfg_date' and value:
                date_info = self._analyze_date_format(value)
                cursor.execute("""
                INSERT INTO date_variations
                (sample_id, detected_date, standardized_date, date_format,
                 date_type, validation_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    sample_id, value, date_info['standardized'],
                    date_info['format'], date_info['type'],
                    date_info['valid'], datetime.now().isoformat()
                ))
            
            # Store address/contact variations
            if field_name in ['manufacturer', 'consumer_care'] and value:
                addr_info = self._analyze_address_format(value, field_name)
                cursor.execute("""
                INSERT INTO address_variations
                (sample_id, field_type, detected_text, standardized_format,
                 contact_type, completeness_score, validation_issues, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sample_id, field_name, value, addr_info['standardized'],
                    addr_info['type'], addr_info['completeness'],
                    json.dumps(addr_info['issues']), datetime.now().isoformat()
                ))
    
    def _analyze_price_format(self, price_text: str) -> Dict:
        """Analyze price format variations for ML training"""
        analysis = {
            'standardized': '',
            'currency': None,
            'tax_status': 'unknown',
            'variations': [],
            'valid': False
        }
        
        text_lower = price_text.lower()
        
        # Detect currency
        if '₹' in price_text:
            analysis['currency'] = '₹'
        elif any(symbol in text_lower for symbol in ['rs', 'inr', 'rupees']):
            analysis['currency'] = 'Rs'
        
        # Detect tax inclusion
        tax_keywords = ['inclusive', 'incl', 'including', 'taxes', 'tax']
        if any(keyword in text_lower for keyword in tax_keywords):
            analysis['tax_status'] = 'included'
        else:
            analysis['tax_status'] = 'unclear'
        
        # Extract numeric value
        price_match = re.search(r'(\d+(?:\.\d{2})?)', price_text)
        if price_match:
            numeric_value = price_match.group(1)
            analysis['standardized'] = f"₹{numeric_value} (Incl. of all taxes)"
        
        # Check validity
        analysis['valid'] = (
            analysis['currency'] is not None and 
            analysis['tax_status'] == 'included' and
            price_match is not None
        )
        
        # Record variations
        analysis['variations'] = [
            'currency_format',
            'tax_clause_format',
            'numeric_format'
        ]
        
        return analysis
    
    def _analyze_date_format(self, date_text: str) -> Dict:
        """Analyze date format variations"""
        analysis = {
            'standardized': '',
            'format': 'unknown',
            'type': 'unknown', 
            'valid': False
        }
        
        # Common date patterns
        patterns = [
            (r'(\d{1,2})/(\d{4})', 'MM/YYYY'),
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', 'DD/MM/YYYY'),
            (r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*(\d{4})', 'Month YYYY'),
            (r'(\d{4})', 'YYYY')
        ]
        
        text_lower = date_text.lower()
        
        for pattern, format_type in patterns:
            match = re.search(pattern, text_lower)
            if match:
                analysis['format'] = format_type
                analysis['valid'] = True
                break
        
        # Detect date type
        if any(keyword in text_lower for keyword in ['mfd', 'manufactured', 'mfg']):
            analysis['type'] = 'manufacturing'
        elif any(keyword in text_lower for keyword in ['exp', 'expiry', 'best before']):
            analysis['type'] = 'expiry'
        elif 'import' in text_lower:
            analysis['type'] = 'import'
        
        return analysis
    
    def _analyze_address_format(self, address_text: str, field_type: str) -> Dict:
        """Analyze address/contact format variations"""
        analysis = {
            'standardized': '',
            'type': 'unknown',
            'completeness': 0.0,
            'issues': []
        }
        
        text_lower = address_text.lower()
        completeness_score = 0.0
        
        if field_type == 'manufacturer':
            # Check manufacturer address completeness
            components = {
                'company_name': any(indicator in text_lower for indicator in ['ltd', 'pvt', 'inc', 'company', 'co']),
                'address_parts': any(part in text_lower for part in ['plot', 'building', 'street', 'road', 'area']),
                'city': any(city in text_lower for city in ['mumbai', 'delhi', 'bangalore', 'chennai', 'pune']),
                'pin_code': bool(re.search(r'\b\d{6}\b', address_text))
            }
            
            completeness_score = sum(components.values()) / len(components)
            analysis['type'] = 'manufacturer_address'
            
            if not components['pin_code']:
                analysis['issues'].append('Missing PIN code')
            if not components['company_name']:
                analysis['issues'].append('Company name unclear')
            if not components['address_parts']:
                analysis['issues'].append('Address components missing')
                
        elif field_type == 'consumer_care':
            # Check consumer care completeness
            contact_types = {
                'phone': bool(re.search(r'\b\d{10}\b', address_text)),
                'email': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', address_text)),
                'address': any(keyword in text_lower for keyword in ['address', 'office', 'care'])
            }
            
            completeness_score = max(contact_types.values()) * 1.0  # At least one contact method
            analysis['type'] = 'consumer_care'
            
            if not any(contact_types.values()):
                analysis['issues'].append('No valid contact method found')
        
        analysis['completeness'] = completeness_score
        return analysis
    
    def add_user_feedback(self, sample_hash: str, corrections: Dict, 
                         overall_rating: int, user_id: str = "anonymous") -> bool:
        """Add detailed user feedback for ML improvement"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get sample ID
        cursor.execute("SELECT id FROM compliance_samples WHERE image_hash = ?", (sample_hash,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return False
        
        sample_id = result[0]
        
        # Store detailed field corrections
        for field_name, correction_data in corrections.items():
            cursor.execute("""
            INSERT INTO user_feedback
            (sample_id, field_name, original_value, corrected_value,
             correction_type, feedback_category, user_id, confidence_score,
             annotation_notes, review_status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sample_id, field_name,
                correction_data.get('original_value', ''),
                correction_data.get('corrected_value', ''),
                correction_data.get('correction_type', 'value_correction'),
                correction_data.get('category', 'user_correction'),
                user_id,
                correction_data.get('confidence', 1.0),
                correction_data.get('notes', ''),
                'pending_review',
                datetime.now().isoformat()
            ))
        
        # Update sample with feedback
        cursor.execute("""
        UPDATE compliance_samples 
        SET user_corrections = ?, feedback_score = ?, updated_at = ?
        WHERE image_hash = ?
        """, (
            json.dumps(corrections), overall_rating,
            datetime.now().isoformat(), sample_hash
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Added user feedback for sample: {sample_hash}")
        return True
    
    def get_training_dataset(self, split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                           min_quality_score: float = 0.7) -> Dict[str, pd.DataFrame]:
        """Get stratified training dataset split"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            cs.*,
            COUNT(uf.id) as feedback_count,
            AVG(uf.confidence_score) as avg_feedback_confidence
        FROM compliance_samples cs
        LEFT JOIN user_feedback uf ON cs.id = uf.sample_id
        WHERE cs.annotation_quality >= ?
        GROUP BY cs.id
        ORDER BY cs.created_at
        """
        
        df = pd.read_sql_query(query, conn, params=[min_quality_score])
        conn.close()
        
        # Stratified split by compliance_score ranges
        df['compliance_category'] = pd.cut(
            df['compliance_score'], 
            bins=[0, 50, 75, 85, 100], 
            labels=['poor', 'fair', 'good', 'excellent']
        )
        
        # Calculate split sizes
        total_samples = len(df)
        train_size = int(total_samples * split_ratio[0])
        val_size = int(total_samples * split_ratio[1])
        test_size = total_samples - train_size - val_size
        
        # Stratified sampling
        train_df = df.groupby('compliance_category', group_keys=False).apply(
            lambda x: x.sample(int(len(x) * split_ratio[0]), random_state=42)
        ).reset_index(drop=True)
        
        remaining_df = df[~df.index.isin(train_df.index)]
        val_df = remaining_df.groupby('compliance_category', group_keys=False).apply(
            lambda x: x.sample(int(len(x) * 0.5), random_state=42)
        ).reset_index(drop=True)
        
        test_df = remaining_df[~remaining_df.index.isin(val_df.index)]
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'metadata': {
                'total_samples': total_samples,
                'train_size': len(train_df),
                'val_size': len(val_df), 
                'test_size': len(test_df),
                'quality_threshold': min_quality_score
            }
        }
    
    def get_ml_insights(self) -> Dict[str, Any]:
        """Get comprehensive ML training insights"""
        
        conn = sqlite3.connect(self.db_path)
        
        insights = {}
        
        # Dataset statistics
        cursor = conn.cursor()
        cursor.execute("""
        SELECT 
            COUNT(*) as total_samples,
            AVG(compliance_score) as avg_compliance,
            AVG(annotation_quality) as avg_quality,
            COUNT(CASE WHEN verified = TRUE THEN 1 END) as verified_samples
        FROM compliance_samples
        """)
        
        stats = cursor.fetchone()
        insights['dataset_stats'] = {
            'total_samples': stats[0],
            'avg_compliance_score': round(stats[1], 2) if stats[1] else 0,
            'avg_annotation_quality': round(stats[2], 2) if stats[2] else 0,
            'verified_samples': stats[3]
        }
        
        # Field-wise performance
        cursor.execute("""
        SELECT 
            rule_category,
            COUNT(cs.id) as samples_count,
            AVG(cs.compliance_score) as avg_score
        FROM legal_metrology_rules lmr
        LEFT JOIN compliance_samples cs ON 1=1
        GROUP BY rule_category
        """)
        
        field_performance = {}
        for row in cursor.fetchall():
            field_performance[row[0]] = {
                'samples': row[1],
                'avg_score': round(row[2], 2) if row[2] else 0
            }
        
        insights['field_performance'] = field_performance
        
        # Violation patterns
        cursor.execute("""
        SELECT 
            violations,
            COUNT(*) as frequency
        FROM compliance_samples 
        WHERE violations != '[]' AND violations IS NOT NULL
        GROUP BY violations
        ORDER BY frequency DESC
        LIMIT 10
        """)
        
        violation_patterns = []
        for row in cursor.fetchall():
            try:
                violations = json.loads(row[0])
                violation_patterns.append({
                    'violations': violations,
                    'frequency': row[1]
                })
            except:
                pass
        
        insights['violation_patterns'] = violation_patterns
        
        # Feedback trends
        cursor.execute("""
        SELECT 
            DATE(created_at) as feedback_date,
            COUNT(*) as feedback_count,
            AVG(confidence_score) as avg_confidence
        FROM user_feedback
        WHERE created_at >= date('now', '-30 days')
        GROUP BY DATE(created_at)
        ORDER BY feedback_date
        """)
        
        feedback_trends = []
        for row in cursor.fetchall():
            feedback_trends.append({
                'date': row[0],
                'count': row[1],
                'avg_confidence': round(row[2], 2) if row[2] else 0
            })
        
        insights['feedback_trends'] = feedback_trends
        
        conn.close()
        return insights
    
    def store_analysis(self, image_data: bytes, extracted_text: str, 
                      compliance_results: Dict, filename: str = "unknown.jpg") -> str:
        """Store analysis results for ML training"""
        
        # Generate hash for image
        image_hash = hashlib.md5(image_data).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Prepare data
            timestamp = datetime.now().isoformat()
            compliance_score = compliance_results.get('compliance_score', 0)
            extracted_fields = json.dumps(compliance_results.get('field_validations', {}))
            violations = json.dumps(compliance_results.get('violations', []))
            
            # Store sample
            cursor.execute("""
            INSERT OR REPLACE INTO compliance_samples
            (image_hash, image_data, image_metadata, extracted_fields, 
             compliance_score, violations, created_at, updated_at, 
             annotation_quality, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_hash, image_data, 
                json.dumps({'filename': filename, 'size': len(image_data)}),
                extracted_fields, compliance_score, violations,
                timestamp, timestamp, 0.8, 'vision_analysis'
            ))
            
            conn.commit()
            self.logger.info(f"Stored analysis for sample: {image_hash}")
            return image_hash
            
        except Exception as e:
            self.logger.error(f"Error storing analysis: {str(e)}")
            return ""
        finally:
            conn.close()

# Alias for easy import
DatasetManager = ComplianceDatasetManager
