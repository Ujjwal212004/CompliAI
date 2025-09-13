import base64
import requests
import json
from typing import Dict, Any
from PIL import Image
import io
import os
import google.generativeai as genai

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without it
    pass

class VisionProcessor:
    """
    Vision processor for Legal Metrology compliance analysis using Google Gemini 2.5 Flash
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("\n⚠️  WARNING: Gemini API key not configured!")
            print("📝 To enable real image analysis:")
            print("   1. Get API key from: https://makersuite.google.com/app/apikey")
            print("   2. Set environment variable: GEMINI_API_KEY=your_key_here")
            print("   3. Or create .env file with: GEMINI_API_KEY=your_key_here")
            print("🔄 Currently using randomized mock data for demonstration\n")
            self.model = None
        else:
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def prepare_image(self, image_input):
        """Prepare image for Gemini API"""
        try:
            if isinstance(image_input, str):
                # File path
                return Image.open(image_input)
            elif hasattr(image_input, 'read'):
                # File-like object (Streamlit uploaded file)
                return Image.open(image_input)
            else:
                # Already PIL Image
                return image_input
        except Exception as e:
            raise Exception(f"Failed to prepare image: {str(e)}")
    
    def analyze_product_compliance(self, image_input) -> Dict[str, Any]:
        """
        Enhanced multi-stage analysis for better field extraction accuracy
        """
        if not self.model:
            return self._get_mock_response()
        
        try:
            image = self.prepare_image(image_input)
            
            # Stage 1: Get raw text extraction
            raw_text = self._extract_raw_text(image)
            
            # Stage 2: Use enhanced structured extraction prompt
            structured_result = self._extract_structured_fields(image, raw_text)
            
            if structured_result.get('success'):
                return structured_result
            
            # Stage 3: Fallback to intelligent parsing if structured extraction fails
            return self._intelligent_fallback_parsing(image, raw_text)
            
        except Exception as e:
            return {"success": False, "error": f"Analysis failed: {str(e)}"}
    
    def _extract_raw_text(self, image) -> str:
        """Extract all visible text from the image"""
        try:
            prompt = """Extract ALL visible text from this product packaging image. 
            List every word, number, symbol, and text element you can see, line by line.
            Be very thorough and include everything readable."""
            
            response = self.model.generate_content([prompt, image])
            return response.text if response.text else ""
        except:
            return ""
    
    def _extract_structured_fields(self, image, raw_text: str) -> Dict[str, Any]:
        """Enhanced structured field extraction with better prompting"""
        try:
            prompt = f"""PRODUCT PACKAGING ANALYSIS

I can see this text on the packaging:
{raw_text[:1000]}...

Now analyze this product packaging image and extract the exact information for Legal Metrology compliance.

For each field below, find the EXACT text as it appears on the package:

1. PRODUCT NAME: Look for the main product/brand name (like "Gem", "Oreo", etc.) - NOT company names
2. MANUFACTURER: Find company name with address (like "XYZ Pvt Ltd, Address")
3. NET QUANTITY: Find weight/volume with units (like "6.61g", "500ml")
4. MRP: Find price information (like "Rs 5", "MRP ₹10")
5. CONSUMER CARE: Find contact info (email, phone, address)
6. MFG DATE: Find manufacturing/expiry dates (MFD, EXP, Best Before)
7. COUNTRY: Find origin information ("Made in India")

CRITICAL: Extract ONLY what you can actually see on this specific package. Do not guess or use examples.

Return ONLY this JSON format:
{{
  "manufacturer": {{"found": true/false, "value": "exact text from package", "compliance": "Pass/Fail"}},
  "net_quantity": {{"found": true/false, "value": "exact text from package", "compliance": "Pass/Fail"}},
  "mrp": {{"found": true/false, "value": "exact text from package", "compliance": "Pass/Fail"}},
  "consumer_care": {{"found": true/false, "value": "exact text from package", "compliance": "Pass/Fail"}},
  "mfg_date": {{"found": true/false, "value": "exact text from package", "compliance": "Pass/Fail"}},
  "country_origin": {{"found": true/false, "value": "exact text from package", "compliance": "Pass/Fail"}},
  "product_name": {{"found": true/false, "value": "exact text from package", "compliance": "Pass/Fail"}},
  "compliance_score": 0-100,
  "violations": ["list of issues"],
  "overall_status": "Compliant/Non-Compliant"
}}"""

            response = self.model.generate_content([prompt, image])
            
            if not response.text:
                return {"success": False, "error": "No structured response"}
            
            # Parse the JSON response
            return self._parse_gemini_response(response.text)
            
        except Exception as e:
            return {"success": False, "error": f"Structured extraction failed: {str(e)}"}
    
    def _intelligent_fallback_parsing(self, image, raw_text: str) -> Dict[str, Any]:
        """Intelligent fallback parsing using regex and text analysis"""
        try:
            import re
            
            # Initialize fields
            fields = {
                "manufacturer": {"found": False, "value": "", "compliance": "Fail"},
                "net_quantity": {"found": False, "value": "", "compliance": "Fail"},
                "mrp": {"found": False, "value": "", "compliance": "Fail"},
                "consumer_care": {"found": False, "value": "", "compliance": "Fail"},
                "mfg_date": {"found": False, "value": "", "compliance": "Fail"},
                "country_origin": {"found": False, "value": "", "compliance": "Fail"},
                "product_name": {"found": False, "value": "", "compliance": "Fail"},
            }
            
            lines = raw_text.split('\n')
            text_lower = raw_text.lower()
            
            # 1. Extract Product Name (first significant non-company line)
            for line in lines:
                line = line.strip()
                if (len(line) >= 2 and len(line) <= 20 and 
                    not any(skip in line.lower() for skip in ['pvt', 'ltd', 'company', 'industries', 'plot', 'sector', 'mfg', 'mrp', 'net', '@', 'phone', 'email'])):
                    if line[0].isupper() or line.isupper():
                        fields["product_name"] = {"found": True, "value": line, "compliance": "Pass"}
                        break
            
            # 2. Extract Net Quantity
            quantity_pattern = r'(\d+(?:\.\d+)?\s*(?:g|gm|gram|kg|ml|l|litre|liter|oz|pieces?))\b'
            quantity_match = re.search(quantity_pattern, text_lower)
            if quantity_match:
                fields["net_quantity"] = {"found": True, "value": quantity_match.group(1), "compliance": "Pass"}
            
            # 3. Extract MRP
            mrp_patterns = [
                r'mrp[:\s]*[rs₹]*\s*(\d+(?:\.\d{2})?)',
                r'[rs₹]\s*(\d+(?:\.\d{2})?)',
                r'price[:\s]*[rs₹]*\s*(\d+(?:\.\d{2})?)',
                r'(\d+(?:\.\d{2})?)\s*rupees?'
            ]
            for pattern in mrp_patterns:
                mrp_match = re.search(pattern, text_lower)
                if mrp_match:
                    price_val = mrp_match.group(1)
                    fields["mrp"] = {"found": True, "value": f"Rs {price_val}", "compliance": "Pass"}
                    break
            
            # 4. Extract Consumer Care
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', raw_text)
            phone_match = re.search(r'(\+91[\s-]?)?[6789]\d{9}|1800[\s-]?\d{3}[\s-]?\d{4}', raw_text)
            
            if email_match:
                fields["consumer_care"] = {"found": True, "value": email_match.group(), "compliance": "Pass"}
            elif phone_match:
                fields["consumer_care"] = {"found": True, "value": phone_match.group(), "compliance": "Pass"}
            
            # 5. Extract Manufacturing Date
            date_patterns = [
                r'mfd[:\s]*([a-z]{3}\s+\d{4}|\d{1,2}[\/\-]\d{4}|\d{4})',
                r'manufactured[:\s]*(\d{1,2}[\/\-]\d{4}|\d{4})',
                r'best\s+before[:\s]*([a-z]{3}\s+\d{4}|\d{1,2}[\/\-]\d{4}|\d{4})',
                r'exp[:\s]*(\d{1,2}[\/\-]\d{4}|\d{4})'
            ]
            for pattern in date_patterns:
                date_match = re.search(pattern, text_lower)
                if date_match:
                    fields["mfg_date"] = {"found": True, "value": date_match.group(0).title(), "compliance": "Pass"}
                    break
            
            # 6. Extract Country of Origin
            if "made in" in text_lower:
                country_match = re.search(r'made in\s+(\w+)', text_lower)
                if country_match:
                    fields["country_origin"] = {"found": True, "value": f"Made in {country_match.group(1).title()}", "compliance": "Pass"}
            elif "india" in text_lower:
                fields["country_origin"] = {"found": True, "value": "India", "compliance": "Pass"}
            
            # 7. Extract Manufacturer
            for line in lines:
                line = line.strip()
                if any(word in line.lower() for word in ['pvt ltd', 'company', 'industries', 'ltd', 'inc']):
                    # Look for address in next line
                    idx = lines.index(line.strip()) if line.strip() in lines else -1
                    address = ""
                    if idx >= 0 and idx + 1 < len(lines):
                        next_line = lines[idx + 1].strip()
                        if any(addr_word in next_line.lower() for addr_word in ['plot', 'sector', 'area', 'road', '-']):
                            address = f", {next_line}"
                    
                    fields["manufacturer"] = {"found": True, "value": f"{line}{address}", "compliance": "Pass"}
                    break
            
            # Calculate compliance
            violations = []
            found_count = sum(1 for field in fields.values() if field["found"])
            compliance_score = int((found_count / len(fields)) * 100)
            
            for field_name, field_data in fields.items():
                if not field_data["found"]:
                    violations.append(f"{field_name.replace('_', ' ').title()} is mandatory but not found")
            
            return {
                "success": True,
                "compliance_data": {
                    **fields,
                    "compliance_score": compliance_score,
                    "violations": violations,
                    "overall_status": "Compliant" if compliance_score >= 85 else "Non-Compliant"
                },
                "raw_response": f"Fallback parsing completed. Found {found_count}/{len(fields)} fields."
            }
            
        except Exception as e:
            return {"success": False, "error": f"Fallback parsing failed: {str(e)}"}
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response with robust JSON extraction"""
        try:
            response_text = response_text.strip()
            
            # Find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return {"success": False, "error": "No JSON found in response"}
            
            json_str = response_text[start_idx:end_idx]
            compliance_data = json.loads(json_str)
            
            return {
                "success": True,
                "compliance_data": compliance_data,
                "raw_response": response_text
            }
            
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON parsing failed: {str(e)}"}
    
    def quick_text_extract(self, image_input) -> Dict[str, Any]:
        """
        Quick text extraction without detailed compliance analysis
        """
        if not self.model:
            return {"success": False, "error": "Gemini API key not configured"}
        
        try:
            image = self.prepare_image(image_input)
            
            prompt = "Extract all visible text from this product packaging image. List all text elements clearly."
            
            response = self.model.generate_content([prompt, image])
            
            if response.text:
                return {"success": True, "extracted_text": response.text}
            else:
                return {"success": False, "error": "No text extracted"}
                
        except Exception as e:
            return {"success": False, "error": f"Text extraction failed: {str(e)}"}
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """
        Fallback parser for non-JSON responses
        """
        fields = {
            "manufacturer": {"found": False, "value": "", "compliance": "Fail"},
            "net_quantity": {"found": False, "value": "", "compliance": "Fail"},
            "mrp": {"found": False, "value": "", "compliance": "Fail"},
            "consumer_care": {"found": False, "value": "", "compliance": "Fail"},
            "mfg_date": {"found": False, "value": "", "compliance": "Fail"},
            "country_origin": {"found": False, "value": "", "compliance": "Fail"},
            "product_name": {"found": False, "value": "", "compliance": "Pass"},
        }
        
        violations = []
        text_lower = text.lower()
        
        # Simple keyword-based detection
        if any(word in text_lower for word in ["manufacturer", "mfg", "made by", "packed by"]):
            fields["manufacturer"]["found"] = True
            fields["manufacturer"]["compliance"] = "Pass"
        else:
            violations.append("Manufacturer details not found")
        
        if any(word in text_lower for word in ["mrp", "price", "₹", "rs", "inr"]):
            fields["mrp"]["found"] = True
            fields["mrp"]["compliance"] = "Pass"
        else:
            violations.append("MRP not clearly visible")
        
        if any(unit in text_lower for unit in ["ml", "g", "kg", "l", "gm", "gram", "litre", "liter"]):
            fields["net_quantity"]["found"] = True
            fields["net_quantity"]["compliance"] = "Pass"
        else:
            violations.append("Net quantity not specified")
        
        # Enhanced consumer care detection
        consumer_care_keywords = [
            "phone", "email", "contact", "customer care", "toll free", "helpline",
            "support", "service", "complaint", "@", "www", "http", "customer",
            "care@", "info@", "support@", "+91", "1800", "080", "011", "022", "033", "044"
        ]
        
        if any(word in text_lower for word in consumer_care_keywords):
            fields["consumer_care"]["found"] = True
            fields["consumer_care"]["compliance"] = "Pass"
            # Try to extract the actual contact info
            import re
            # Look for email patterns
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
            # Look for phone patterns
            phone_match = re.search(r'(\+91[\s-]?)?[6789]\d{9}|1800[\s-]?\d{3}[\s-]?\d{4}', text)
            
            if email_match:
                fields["consumer_care"]["value"] = email_match.group()
            elif phone_match:
                fields["consumer_care"]["value"] = phone_match.group()
        else:
            violations.append("Consumer care details missing")
        
        if any(word in text_lower for word in ["mfd", "manufactured", "exp", "expiry", "best before"]):
            fields["mfg_date"]["found"] = True
            fields["mfg_date"]["compliance"] = "Pass"
        else:
            violations.append("Manufacturing/Expiry date not found")
        
        if any(word in text_lower for word in ["india", "made in", "origin", "country"]):
            fields["country_origin"]["found"] = True
            fields["country_origin"]["compliance"] = "Pass"
        else:
            violations.append("Country of origin not specified")
        
        # Calculate compliance score
        found_count = sum(1 for field in fields.values() if field["found"])
        compliance_score = int((found_count / len(fields)) * 100)
        
        return {
            **fields,
            "compliance_score": compliance_score,
            "violations": violations,
            "overall_status": "Compliant" if compliance_score >= 85 else "Non-Compliant"
        }
    
    def _get_mock_response(self) -> Dict[str, Any]:
        """
        Return mock response for demo purposes when API key is not available
        """
        import random
        
        # Randomize mock data to simulate real analysis
        mock_scenarios = [
            {
                "manufacturer": {"found": True, "value": "FreshFood Industries Pvt Ltd, Sector 12, Gurgaon - 122001", "compliance": "Pass"},
                "net_quantity": {"found": True, "value": "250g", "compliance": "Pass"},
                "mrp": {"found": True, "value": "₹89 (Incl. of all taxes)", "compliance": "Pass"},
                "consumer_care": {"found": True, "value": "Customer Care: 1800-123-4567", "compliance": "Pass"},
                "mfg_date": {"found": True, "value": "MFD: 12/2024", "compliance": "Pass"},
                "country_origin": {"found": True, "value": "Made in India", "compliance": "Pass"},
                "product_name": {"found": True, "value": "Organic Cookies", "compliance": "Pass"},
                "compliance_score": 100,
                "violations": [],
                "overall_status": "Compliant"
            },
            {
                "manufacturer": {"found": True, "value": "HealthyBites Co., Plot 45, MIDC, Pune - 411019", "compliance": "Pass"},
                "net_quantity": {"found": True, "value": "500ml", "compliance": "Pass"},
                "mrp": {"found": True, "value": "₹145", "compliance": "Pass"},
                "consumer_care": {"found": True, "value": "Email: care@healthybites.com, Ph: +91-98765-43210", "compliance": "Pass"},
                "mfg_date": {"found": False, "value": "", "compliance": "Fail"},
                "country_origin": {"found": True, "value": "India", "compliance": "Pass"},
                "product_name": {"found": True, "value": "Natural Fruit Juice", "compliance": "Pass"},
                "compliance_score": 85,
                "violations": ["Manufacturing date not clearly visible"],
                "overall_status": "Compliant"
            },
            {
                "manufacturer": {"found": True, "value": "TastyTreats Pvt Ltd, Industrial Estate, Chennai - 600058", "compliance": "Pass"},
                "net_quantity": {"found": True, "value": "200g", "compliance": "Pass"},
                "mrp": {"found": True, "value": "₹75 (Inclusive of all taxes)", "compliance": "Pass"},
                "consumer_care": {"found": False, "value": "", "compliance": "Fail"},
                "mfg_date": {"found": True, "value": "Best Before: 06/2025", "compliance": "Pass"},
                "country_origin": {"found": True, "value": "Made in India", "compliance": "Pass"},
                "product_name": {"found": True, "value": "Chocolate Wafers", "compliance": "Pass"},
                "compliance_score": 85,
                "violations": ["Consumer care details missing - no contact information provided"],
                "overall_status": "Compliant"
            }
        ]
        
        # Select a random scenario
        selected_scenario = random.choice(mock_scenarios)
        
        return {
            "success": True,
            "compliance_data": selected_scenario,
            "raw_response": "Mock response - Gemini API key not configured. Using randomized sample data for demonstration. Configure GEMINI_API_KEY for real analysis."
        }
