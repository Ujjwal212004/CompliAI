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
        Analyze product image for Legal Metrology compliance using Gemini 2.5 Flash
        """
        if not self.model:
            # Return mock data for demo when API key is not available
            return self._get_mock_response()
        
        try:
            # Prepare image for Gemini
            image = self.prepare_image(image_input)
            
            # Create the prompt for Legal Metrology compliance analysis
            prompt = """Analyze this product packaging image for Legal Metrology compliance in India. 
            
You need to extract and identify the following mandatory information:

1. MANUFACTURER/PACKER/IMPORTER: Name and complete address
2. NET QUANTITY: Weight, volume, or count with proper units (g, kg, ml, l, pieces, etc.)
3. MRP (Maximum Retail Price): Price inclusive of all taxes
4. CONSUMER CARE: Phone number, email, or address for complaints
5. MANUFACTURING DATE: Date of manufacture/import/packing  
6. COUNTRY OF ORIGIN: Where the product is made
7. PRODUCT NAME: Brand and product name

For each field, analyze and provide:
- FOUND: true/false (whether the information is visible)
- VALUE: The extracted text (if found, otherwise empty string)
- COMPLIANCE: "Pass" if found and properly formatted, "Fail" if missing or unclear

Also provide:
- COMPLIANCE SCORE: Number from 0-100 based on how many mandatory fields are present
- VIOLATIONS: Array of specific issues found
- OVERALL STATUS: "Compliant" if score >= 85, otherwise "Non-Compliant"

Respond ONLY with valid JSON in this exact format:
{
  "manufacturer": {"found": true/false, "value": "extracted text", "compliance": "Pass/Fail"},
  "net_quantity": {"found": true/false, "value": "extracted text", "compliance": "Pass/Fail"},
  "mrp": {"found": true/false, "value": "extracted text", "compliance": "Pass/Fail"},
  "consumer_care": {"found": true/false, "value": "extracted text", "compliance": "Pass/Fail"},
  "mfg_date": {"found": true/false, "value": "extracted text", "compliance": "Pass/Fail"},
  "country_origin": {"found": true/false, "value": "extracted text", "compliance": "Pass/Fail"},
  "product_name": {"found": true/false, "value": "extracted text", "compliance": "Pass/Fail"},
  "compliance_score": 75,
  "violations": ["Missing consumer care details", "MRP format unclear"],
  "overall_status": "Non-Compliant"
}"""

            # Generate response using Gemini
            response = self.model.generate_content([prompt, image])
            
            if not response.text:
                return {"success": False, "error": "No response from Gemini API"}
            
            # Try to parse JSON from the response
            try:
                # Clean the response text
                response_text = response.text.strip()
                
                # Find JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx == -1 or end_idx == 0:
                    # If no JSON found, use fallback parser
                    return {
                        "success": True,
                        "raw_response": response_text,
                        "compliance_data": self._parse_text_response(response_text)
                    }
                
                json_str = response_text[start_idx:end_idx]
                compliance_data = json.loads(json_str)
                
                return {
                    "success": True,
                    "compliance_data": compliance_data,
                    "raw_response": response_text
                }
                
            except json.JSONDecodeError as e:
                # If JSON parsing fails, use fallback parser
                return {
                    "success": True,
                    "raw_response": response.text,
                    "compliance_data": self._parse_text_response(response.text)
                }
                
        except Exception as e:
            return {"success": False, "error": f"Gemini API error: {str(e)}"}
    
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
