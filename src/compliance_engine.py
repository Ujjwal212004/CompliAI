import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd

class LegalMetrologyRuleEngine:
    """
    Rule engine for Legal Metrology compliance based on Packaged Commodities Rules, 2011
    Implements mandatory declaration requirements for pre-packaged goods
    """
    
    def __init__(self):
        self.mandatory_fields = [
            'manufacturer',
            'net_quantity', 
            'mrp',
            'consumer_care',
            'mfg_date',
            'country_origin',
            'product_name'
        ]
        
        # Weight/Volume unit patterns for net quantity validation
        self.valid_units = {
            'weight': ['g', 'gm', 'gram', 'kg', 'kilogram', 'mg', 'milligram'],
            'volume': ['ml', 'millilitre', 'l', 'litre', 'liter'],
            'count': ['pieces', 'pcs', 'units', 'nos', 'numbers']
        }
    
    def validate_compliance(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main validation function that checks all Legal Metrology requirements
        """
        validation_results = {}
        violations = []
        compliance_score = 0
        
        # Validate each mandatory field
        for field in self.mandatory_fields:
            field_data = extracted_data.get(field, {})
            validation_result = self._validate_field(field, field_data)
            validation_results[field] = validation_result
            
            if validation_result['compliance'] == 'Pass':
                compliance_score += 1
        
        # Calculate percentage score
        total_score = int((compliance_score / len(self.mandatory_fields)) * 100)
        
        # Collect all violations
        for field, result in validation_results.items():
            if result['compliance'] == 'Fail':
                violations.extend(result.get('violations', []))
        
        # Determine overall compliance status
        overall_status = self._determine_compliance_status(total_score, violations)
        
        return {
            'field_validations': validation_results,
            'compliance_score': total_score,
            'violations': violations,
            'overall_status': overall_status,
            'mandatory_fields_found': compliance_score,
            'total_mandatory_fields': len(self.mandatory_fields)
        }
    
    def _validate_field(self, field_name: str, field_data: Dict) -> Dict[str, Any]:
        """
        Validate individual field based on Legal Metrology rules
        """
        if not field_data.get('found', False):
            return {
                'compliance': 'Fail',
                'violations': [f'{field_name.replace("_", " ").title()} is mandatory but not found'],
                'severity': 'Critical'
            }
        
        value = field_data.get('value', '').strip()
        if not value:
            return {
                'compliance': 'Fail',
                'violations': [f'{field_name.replace("_", " ").title()} field is empty'],
                'severity': 'Critical'
            }
        
        # Field-specific validation
        validation_method = getattr(self, f'_validate_{field_name}', None)
        if validation_method:
            return validation_method(value)
        
        # Default validation for fields without specific rules
        return {
            'compliance': 'Pass',
            'violations': [],
            'severity': 'None'
        }
    
    def _validate_manufacturer(self, value: str) -> Dict[str, Any]:
        """
        Validate manufacturer/packer/importer details
        Must include name and complete address
        """
        violations = []
        severity = 'None'
        
        # Check minimum length
        if len(value) < 10:
            violations.append("Manufacturer details too brief - must include complete name and address")
            severity = 'High'
        
        # Check for address components
        address_indicators = ['ltd', 'pvt', 'private', 'limited', 'inc', 'co', 'company', 
                            'road', 'street', 'plot', 'building', 'floor', 'area',
                            'city', 'state', 'pin', 'pincode', '-', 'mumbai', 'delhi', 
                            'bangalore', 'chennai', 'hyderabad', 'pune', 'kolkata']
        
        if not any(indicator in value.lower() for indicator in address_indicators):
            violations.append("Manufacturer address appears incomplete - must include complete address with city/state")
            severity = 'High'
        
        # Check for PIN code pattern
        pin_pattern = r'\b\d{6}\b'
        if not re.search(pin_pattern, value):
            violations.append("PIN code not found in manufacturer address")
            severity = 'Medium'
        
        compliance = 'Pass' if not violations or severity == 'Medium' else 'Fail'
        
        return {
            'compliance': compliance,
            'violations': violations,
            'severity': severity
        }
    
    def _validate_net_quantity(self, value: str) -> Dict[str, Any]:
        """
        Validate net quantity declaration
        Must include quantity and proper unit
        """
        violations = []
        severity = 'None'
        
        # Check for numeric value
        numeric_pattern = r'\d+\.?\d*'
        if not re.search(numeric_pattern, value):
            violations.append("Net quantity must include numeric value")
            severity = 'Critical'
            return {'compliance': 'Fail', 'violations': violations, 'severity': severity}
        
        # Check for valid units
        value_lower = value.lower()
        has_valid_unit = False
        
        for unit_type, units in self.valid_units.items():
            if any(unit in value_lower for unit in units):
                has_valid_unit = True
                break
        
        if not has_valid_unit:
            violations.append("Net quantity must include proper unit (g, kg, ml, l, pieces, etc.)")
            severity = 'Critical'
        
        # Check for proper format (number + unit)
        proper_format_pattern = r'\d+\.?\d*\s*(g|gm|gram|kg|ml|l|litre|liter|pieces|pcs|nos)\b'
        if not re.search(proper_format_pattern, value_lower):
            violations.append("Net quantity format should be: number + unit (e.g., 500g, 1kg, 250ml)")
            severity = 'Medium'
        
        compliance = 'Pass' if severity != 'Critical' else 'Fail'
        
        return {
            'compliance': compliance,
            'violations': violations,
            'severity': severity
        }
    
    def _validate_mrp(self, value: str) -> Dict[str, Any]:
        """
        Validate Maximum Retail Price declaration
        Must include price and "inclusive of all taxes"
        """
        violations = []
        severity = 'None'
        
        # Check for currency symbols or indicators
        currency_indicators = ['₹', 'rs', 'inr', 'rupees']
        if not any(indicator in value.lower() for indicator in currency_indicators):
            violations.append("MRP must include currency symbol (₹) or indicator")
            severity = 'High'
        
        # Check for numeric value
        if not re.search(r'\d+\.?\d*', value):
            violations.append("MRP must include numeric price value")
            severity = 'Critical'
            return {'compliance': 'Fail', 'violations': violations, 'severity': severity}
        
        # Check for "inclusive of taxes" declaration
        tax_indicators = ['incl', 'inclusive', 'including', 'taxes', 'tax', 'all taxes']
        has_tax_declaration = any(indicator in value.lower() for indicator in tax_indicators)
        
        if not has_tax_declaration:
            violations.append("MRP must state 'Inclusive of all taxes' as per Legal Metrology rules")
            severity = 'High'
        
        compliance = 'Pass' if severity != 'Critical' else 'Fail'
        
        return {
            'compliance': compliance,
            'violations': violations,
            'severity': severity
        }
    
    def _validate_consumer_care(self, value: str) -> Dict[str, Any]:
        """
        Validate consumer care details
        Must include phone/email/address for complaints
        """
        violations = []
        severity = 'None'
        
        # Check for phone number
        phone_patterns = [
            r'\b\d{10}\b',  # 10 digit number
            r'\b\d{4}-\d{6}\b',  # Format: 1234-567890
            r'\b\d{5}-\d{5}\b',  # Format: 12345-67890
            r'\+91\s*\d{10}',    # International format
            r'1800\s*\d{6,7}'    # Toll-free
        ]
        
        has_phone = any(re.search(pattern, value) for pattern in phone_patterns)
        
        # Check for email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        has_email = re.search(email_pattern, value)
        
        # Check for address
        address_keywords = ['address', 'office', 'head office', 'customer care', 'complaints']
        has_address_info = any(keyword in value.lower() for keyword in address_keywords)
        
        if not (has_phone or has_email or has_address_info):
            violations.append("Consumer care details must include phone number, email, or address for complaints")
            severity = 'Critical'
        
        # Additional checks for proper format
        if has_phone and not re.search(r'customer|care|complaint|help|support|toll', value.lower()):
            violations.append("Consumer care should be clearly labeled as customer care/helpline")
            severity = 'Low'
        
        compliance = 'Pass' if severity != 'Critical' else 'Fail'
        
        return {
            'compliance': compliance,
            'violations': violations,
            'severity': severity
        }
    
    def _validate_mfg_date(self, value: str) -> Dict[str, Any]:
        """
        Validate manufacturing date/import date
        Must be clearly mentioned and in proper format
        """
        violations = []
        severity = 'None'
        
        # Common date patterns
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',      # DD/MM/YYYY or MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',      # DD-MM-YYYY
            r'\d{2}/\d{4}',                # MM/YYYY
            r'\d{4}',                      # YYYY
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{4}',  # Month Year
        ]
        
        has_date = any(re.search(pattern, value.lower()) for pattern in date_patterns)
        
        if not has_date:
            violations.append("Manufacturing/Import date must be in proper format (MM/YYYY, DD/MM/YYYY, etc.)")
            severity = 'Critical'
            return {'compliance': 'Fail', 'violations': violations, 'severity': severity}
        
        # Check for proper labeling
        date_labels = ['mfd', 'manufactured', 'mfg date', 'import', 'packed', 'best before', 'exp', 'expiry']
        if not any(label in value.lower() for label in date_labels):
            violations.append("Date should be clearly labeled as MFD/Manufacturing/Import date")
            severity = 'Medium'
        
        compliance = 'Pass' if severity != 'Critical' else 'Fail'
        
        return {
            'compliance': compliance,
            'violations': violations,
            'severity': severity
        }
    
    def _validate_country_origin(self, value: str) -> Dict[str, Any]:
        """
        Validate country of origin declaration
        Must clearly state where product is made/imported from
        """
        violations = []
        severity = 'None'
        
        # Common country indicators
        origin_patterns = ['made in', 'origin', 'country', 'manufactured in', 'imported from']
        has_origin_label = any(pattern in value.lower() for pattern in origin_patterns)
        
        if not has_origin_label:
            violations.append("Country of origin should be clearly stated as 'Made in [Country]' or 'Origin: [Country]'")
            severity = 'Medium'
        
        # Check for actual country name (basic validation)
        if len(value.split()) < 2:
            violations.append("Country of origin appears incomplete")
            severity = 'High'
        
        compliance = 'Pass' if severity != 'Critical' else 'Fail'
        
        return {
            'compliance': compliance,
            'violations': violations,
            'severity': severity
        }
    
    def _validate_product_name(self, value: str) -> Dict[str, Any]:
        """
        Validate product name and brand
        Must be clearly visible and identifiable
        """
        violations = []
        severity = 'None'
        
        if len(value) < 3:
            violations.append("Product name appears too brief or unclear")
            severity = 'Medium'
        
        # Generally, product name is less strictly regulated
        # Main requirement is that it should be clearly visible
        
        return {
            'compliance': 'Pass',
            'violations': violations,
            'severity': severity
        }
    
    def _determine_compliance_status(self, score: int, violations: List[str]) -> str:
        """
        Determine overall compliance status based on score and violation severity
        """
        # Check for critical violations
        critical_violations = [v for v in violations if 'mandatory' in v.lower() or 'must' in v.lower()]
        
        if critical_violations or score < 70:
            return 'Non-Compliant'
        elif score >= 85:
            return 'Compliant'
        else:
            return 'Partially Compliant'
    
    def generate_compliance_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed compliance report for regulatory review
        """
        report = {
            'summary': {
                'overall_status': validation_results['overall_status'],
                'compliance_score': validation_results['compliance_score'],
                'fields_compliant': validation_results['mandatory_fields_found'],
                'total_fields': validation_results['total_mandatory_fields']
            },
            'field_details': {},
            'violations_by_severity': {
                'Critical': [],
                'High': [],
                'Medium': [],
                'Low': []
            },
            'recommendations': []
        }
        
        # Process field validations
        for field, validation in validation_results['field_validations'].items():
            field_name = field.replace('_', ' ').title()
            report['field_details'][field_name] = {
                'status': validation['compliance'],
                'issues': validation.get('violations', []),
                'severity': validation.get('severity', 'None')
            }
            
            # Categorize violations by severity
            severity = validation.get('severity', 'None')
            if severity != 'None':
                report['violations_by_severity'][severity].extend(validation.get('violations', []))
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(validation_results)
        
        return report
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on compliance gaps
        """
        recommendations = []
        
        for field, validation in validation_results['field_validations'].items():
            if validation['compliance'] == 'Fail':
                field_name = field.replace('_', ' ')
                
                if field == 'manufacturer':
                    recommendations.append("Ensure manufacturer details include complete name and address with PIN code")
                elif field == 'net_quantity':
                    recommendations.append("Display net quantity with proper units (g, kg, ml, l, pieces)")
                elif field == 'mrp':
                    recommendations.append("Include MRP with currency symbol and 'Inclusive of all taxes' statement")
                elif field == 'consumer_care':
                    recommendations.append("Provide consumer care details (phone/email/address for complaints)")
                elif field == 'mfg_date':
                    recommendations.append("Clearly mention manufacturing/import date in proper format")
                elif field == 'country_origin':
                    recommendations.append("State country of origin as 'Made in [Country]'")
        
        if validation_results['compliance_score'] < 100:
            recommendations.append("Review packaging to ensure all Legal Metrology mandatory declarations are present")
        
        return recommendations
