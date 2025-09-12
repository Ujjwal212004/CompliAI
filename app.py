import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import sys
import os
import numpy as np
from datetime import datetime, timedelta
import json



# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from vision_processor import VisionProcessor
from compliance_engine import LegalMetrologyRuleEngine
from dataset_manager import DatasetManager
from ml_trainer import MLTrainer
from feedback_loop import FeedbackLoop
from cascading_analyzer import CascadingComplianceAnalyzer

# Configure Streamlit page
st.set_page_config(
    page_title="CompliAI - Legal Metrology Compliance Checker",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling, including dark mode
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1f77b4, #2e8b57);
        color: white;
        padding: 2rem 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .site-title {
        font-size: 2.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }
    .site-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    .footer {
        background-color: #2c3e50;
        color: #ecf0f1;
        padding: 3rem 2rem 2rem;
        margin-top: 3rem;
        border-radius: 10px 10px 0 0;
    }
    .footer h3 {
        color: #3498db;
        margin-bottom: 1rem;
    }
    .footer-content {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin-bottom: 2rem;
    }
    .footer-section {
        line-height: 1.6;
    }
    .footer-bottom {
        text-align: center;
        padding-top: 2rem;
        border-top: 1px solid #34495e;
        color: #bdc3c7;
    }
    .compliance-pass {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #28a745;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .compliance-pass:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3);
    }
    .compliance-fail {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #dc3545;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .compliance-fail:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(220, 53, 69, 0.3);
    }
    .compliance-partial {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #ffc107;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .compliance-partial:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 193, 7, 0.3);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .hover-card {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .hover-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        background-color: #f9f9f9;
    }
    /* Dark Mode specific styles */
    body.dark-mode, .stApp.dark-mode {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card.dark-mode {
        background-color: #1a1a1a;
        color: #fafafa;
        border: 1px solid #333;
    }
    .footer.dark-mode {
        background-color: #1a1a1a;
        border-top: 1px solid #333;
    }
    .hover-card.dark-mode:hover {
        background-color: #2a2a2a;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    if 'dataset_manager' not in st.session_state:
        st.session_state.dataset_manager = DatasetManager()
    if 'ml_trainer' not in st.session_state:
        st.session_state.ml_trainer = MLTrainer(st.session_state.dataset_manager)
    if 'feedback_loop' not in st.session_state:
        st.session_state.feedback_loop = FeedbackLoop(st.session_state.dataset_manager, st.session_state.ml_trainer)
    if 'cascading_analyzer' not in st.session_state:
        st.session_state.cascading_analyzer = CascadingComplianceAnalyzer(st.session_state.dataset_manager)
    if 'analysis_method' not in st.session_state:
        st.session_state.analysis_method = 'cascading'  # Default to cascading analysis

def set_theme(theme):
    """Set Streamlit theme dynamically (requires rerun)"""
    st.session_state.theme = theme
    if theme == 'dark':
        st._config.set_option('theme.base', 'dark')
        st._config.set_option('theme.backgroundColor', '#0e1117')
        st._config.set_option('theme.textColor', '#fafafa')
        st._config.set_option('theme.primaryColor', '#FFD700')
        st._config.set_option('theme.secondaryBackgroundColor', '#1a1a1a')
    else:
        st._config.set_option('theme.base', 'light')
        st._config.set_option('theme.backgroundColor', '#fafafa')
        st._config.set_option('theme.textColor', '#333333')
        st._config.set_option('theme.primaryColor', '#1f77b4')
        st._config.set_option('theme.secondaryBackgroundColor', '#f8f9fa')
    st.rerun()

def render_header():
    # Main header with enhanced styling
    st.markdown("""
    <div class="main-header">
        <div class="site-title">CompliAI</div>
        <div class="site-subtitle">AI-Powered Legal Metrology Compliance Checker for E-Commerce Excellence</div>
    </div>
    """, unsafe_allow_html=True)
    
    # How It Works mini expandable section
    with st.expander("🚀 How It Works - Quick Guide", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            **Step 1: Upload**  
            📤 Select product image  
            (PNG, JPG, JPEG, WEBP)
            """)
        
        with col2:
            st.markdown("""
            **Step 2: Analyze**  
            🔍 Click 'Analyze Compliance'  
            AI scans with Gemini Vision
            """)
        
        with col3:
            st.markdown("""
            **Step 3: Review**  
            📊 View compliance score  
            Field-by-field analysis
            """)
        
        with col4:
            st.markdown("""
            **Step 4: Export**  
            📥 Download reports  
            CSV data for records
            """)
    
    # Legal Metrology Requirements with hover card styling
    with st.expander("Legal Metrology Requirements (Packaged Commodities Rules, 2011)"):
        st.markdown("**Mandatory Information Required on Pre-packaged Goods:**")
        st.markdown("""
        **1. Manufacturer/Packer/Importer:** Complete name and address details  
        **2. Net Quantity:** Weight, volume, or count with standard metric units  
        **3. MRP (Maximum Retail Price):** Price inclusive of all applicable taxes  
        **4. Consumer Care Details:** Contact information for consumer complaints  
        **5. Date of Manufacture/Import:** Clear manufacturing or import date  
        **6. Country of Origin:** Manufacturing or import origin country  
        **7. Product Name:** Brand name and product identification  
        
        *All fields are mandatory for legal compliance in Indian markets under the Packaged Commodities Rules, 2011.*
        """)

def render_help_section():
    """Render comprehensive help section"""
    st.markdown('<div id="help-section"></div>', unsafe_allow_html=True)
    st.markdown("### How CompliAI Works")
    
    # Use streamlit components instead of raw HTML to prevent rendering issues
    st.info("**CompliAI automatically analyzes product packaging images to ensure compliance with Indian Legal Metrology requirements for e-commerce platforms.**")
    
    st.markdown("#### Step-by-Step Guide:")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Step 1:**")
    with col2:
        st.markdown("**Upload Product Image** - Click 'Browse files' and select a clear image of your product packaging. Supported formats: PNG, JPG, JPEG, WEBP.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Step 2:**")
    with col2:
        st.markdown("**Click 'Analyze Compliance'** - Our AI-powered system will scan your image using Google's Gemini Pro Vision technology.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Step 3:**")
    with col2:
        st.markdown("**Review Compliance Results** - View your compliance score, field-by-field analysis, and detailed violation reports.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Step 4:**")
    with col2:
        st.markdown("**Download Reports (Optional)** - Generate and download detailed compliance reports for your records.")
    
    st.markdown("#### Best Practices for Accurate Results:")
    st.markdown("""
    - Use high-resolution images with good lighting
    - Ensure all text is clearly visible and not blurred
    - Capture the entire product packaging in the frame
    - Avoid shadows or glare that might obscure important text
    - Include both front and back labels if compliance information is split
    """)
    
    st.markdown("#### What CompliAI Checks:")
    st.markdown("""
    - Manufacturer, packer, or importer details with complete address
    - Net quantity in standard units (grams, liters, pieces)
    - Maximum Retail Price (MRP) including taxes
    - Consumer care contact information
    - Manufacturing or import date
    - Country of origin
    - Product name and brand identification
    """)

def render_file_upload():
    st.markdown("### 📤 Upload Product Image")
    
    # Analysis method selector
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a product packaging image",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload a clear image of the product packaging with visible text"
        )
    with col2:
        st.markdown("**Analysis Method:**")
        analysis_method = st.radio(
            "Choose analysis approach",
            options=['cascading', 'gemini_only'],
            format_func=lambda x: {
                'cascading': '🔄 Sequential Analysis (Rule-based → ML → Gemini)',
                'gemini_only': '🤖 Gemini API Only (Original)'
            }[x],
            index=0,
            help="Sequential analysis tries rule-based first (fast), then ML model if rule-based fails, finally Gemini API if both fail. Uses the first successful result for efficiency."
        )
        st.session_state.analysis_method = analysis_method
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Product Image", use_column_width=True)
        st.session_state.uploaded_image = uploaded_file
        if st.button("🔍 Analyze Compliance", type="primary", use_container_width=True):
            analyze_image(uploaded_file)
    return uploaded_file

def analyze_image(uploaded_file):
    # Get analysis method from session state
    analysis_method = st.session_state.get('analysis_method', 'cascading')
    
    if analysis_method == 'cascading':
        with st.spinner("🔄 Performing sequential analysis: Rule-based first, then ML if needed, finally Gemini if required..."):
            analyze_with_cascading_system(uploaded_file)
    else:
        with st.spinner("🤖 Analyzing image with Gemini Vision API..."):
            analyze_with_gemini_only(uploaded_file)

def analyze_with_cascading_system(uploaded_file):
    """Perform cascading analysis with rule-based → ML → Gemini flow"""
    try:
        uploaded_file.seek(0)
        
        # Use cascading analyzer
        cascading_results = st.session_state.cascading_analyzer.analyze_compliance(
            uploaded_file, use_advanced_flow=True
        )
        
        if not cascading_results.get('success', False):
            st.error(f"❌ Cascading analysis failed: {cascading_results.get('error', 'Unknown error')}")
            return
        
        # Store results in session state
        st.session_state.analysis_results = {
            'cascading_results': cascading_results,
            'analysis_method': 'cascading',
            'validation_results': cascading_results.get('validation_results', {}),
            'compliance_report': cascading_results.get('compliance_report', {}),
            'raw_data': cascading_results.get('compliance_data', {}),
            'steps_performed': cascading_results.get('steps_performed', []),
            'confidence_scores': cascading_results.get('confidence_scores', {}),
            'best_result_source': cascading_results.get('best_result_source', 'unknown')
        }
        
        # Store in dataset for ML training
        try:
            uploaded_file.seek(0)
            image_data = uploaded_file.read()
            
            # Use extracted text from the best result source
            raw_responses = cascading_results.get('raw_responses', {})
            extracted_text = ''
            if cascading_results.get('best_result_source') in raw_responses:
                best_response = raw_responses[cascading_results.get('best_result_source')]
                extracted_text = best_response.get('extracted_text', '') or best_response.get('raw_response', '')
            
            sample_hash = st.session_state.dataset_manager.store_analysis(
                image_data=image_data,
                extracted_text=extracted_text,
                compliance_results=cascading_results.get('validation_results', {}),
                filename=uploaded_file.name
            )
            
            # Store the hash for feedback use
            st.session_state.analysis_results['sample_hash'] = sample_hash
        except Exception as storage_error:
            st.warning(f"Results analyzed but not stored in dataset: {str(storage_error)}")
        
        # Show success message with analysis details
        best_source = cascading_results.get('best_result_source', 'unknown')
        steps_performed = cascading_results.get('steps_performed', [])
        
        success_msg = f"✅ Analysis completed successfully!\n"
        success_msg += f"📊 Result from: **{best_source.replace('_', ' ').title()}**\n"
        success_msg += f"🔄 Steps tried: {' → '.join([step.replace('_', ' ').title() for step in steps_performed])}"
        
        st.success(success_msg)
        
        # Show confidence scores in an info box
        confidence_scores = cascading_results.get('confidence_scores', {})
        if confidence_scores:
            conf_text = "**Method Confidence Scores:**\n"
            for method, score in confidence_scores.items():
                conf_text += f"• {method.replace('_', ' ').title()}: {score:.2%}\n"
            st.info(conf_text)
            
    except Exception as e:
        st.error(f"❌ Error during cascading analysis: {str(e)}")

def analyze_with_gemini_only(uploaded_file):
    """Fallback to original Gemini-only analysis"""
    try:
        vision_processor = VisionProcessor()
        rule_engine = LegalMetrologyRuleEngine()
        uploaded_file.seek(0)
        vision_results = vision_processor.analyze_product_compliance(uploaded_file)
        if not vision_results.get('success', False):
            st.error(f"❌ Analysis failed: {vision_results.get('error', 'Unknown error')}")
            return
        compliance_data = vision_results.get('compliance_data', {})
        validation_results = rule_engine.validate_compliance(compliance_data)
        compliance_report = rule_engine.generate_compliance_report(validation_results)
        
        # Store results in session state
        st.session_state.analysis_results = {
            'vision_results': vision_results,
            'analysis_method': 'gemini_only',
            'validation_results': validation_results,
            'compliance_report': compliance_report,
            'raw_data': compliance_data,
            'best_result_source': 'gemini_api'
        }
        
        # Store in dataset for ML training
        try:
            uploaded_file.seek(0)
            image_data = uploaded_file.read()
            sample_hash = st.session_state.dataset_manager.store_analysis(
                image_data=image_data,
                extracted_text=vision_results.get('extracted_text', ''),
                compliance_results=validation_results,
                filename=uploaded_file.name
            )
            # Store the hash for feedback use
            st.session_state.analysis_results['sample_hash'] = sample_hash
        except Exception as storage_error:
            st.warning(f"Results analyzed but not stored in dataset: {str(storage_error)}")
        
        st.success("✅ Analysis completed successfully using Gemini Vision API!")
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")

def render_compliance_overview():
    if not st.session_state.analysis_results:
        return
    results = st.session_state.analysis_results
    validation = results.get('validation_results', {})
    report = results.get('compliance_report', {})
    st.markdown("### 📊 Compliance Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score = validation.get('compliance_score', 0)
        st.metric(
            "Compliance Score",
            f"{score}%",
            delta=f"{score-70}%" if score >= 70 else f"{score-70}%"
        )
    with col2:
        status = validation.get('overall_status', 'Unknown')
        st.metric("Overall Status", status)
    with col3:
        found = validation.get('mandatory_fields_found', 0)
        total = validation.get('total_mandatory_fields', 7)
        st.metric("Fields Found", f"{found}/{total}")
    with col4:
        violations = len(validation.get('violations', []))
        st.metric("Violations", violations)
    status = validation.get('overall_status', 'Unknown')
    if status == 'Compliant':
        st.markdown('<div class="compliance-pass">✅ <strong>COMPLIANT</strong> - Product meets Legal Metrology requirements</div>', unsafe_allow_html=True)
    elif status == 'Partially Compliant':
        st.markdown('<div class="compliance-partial">⚠ <strong>PARTIALLY COMPLIANT</strong> - Some requirements missing</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="compliance-fail">❌ <strong>NON-COMPLIANT</strong> - Multiple mandatory requirements missing</div>', unsafe_allow_html=True)

def render_field_analysis():
    if not st.session_state.analysis_results:
        return
    results = st.session_state.analysis_results
    raw_data = results['raw_data']
    validation = results['validation_results']
    st.markdown("### 🔍 Detailed Field Analysis")
    field_labels = {
        'manufacturer': ' Manufacturer/Packer/Importer',
        'net_quantity': ' Net Quantity',
        'mrp': ' Maximum Retail Price (MRP)',
        'consumer_care': ' Consumer Care Details',
        'mfg_date': ' Manufacturing Date',
        'country_origin': ' Country of Origin',
        'product_name': ' Product Name'
    }
    for field, label in field_labels.items():
        with st.expander(f"{label}", expanded=False):
            field_data = raw_data.get(field, {})
            field_validation = validation.get('field_validations', {}).get(field, {})
            col1, col2 = st.columns([1, 1])
            with col1:
                # Handle both dict and non-dict field data
                if isinstance(field_data, dict):
                    found = field_data.get('found', False)
                    value = field_data.get('value', 'Not found')
                else:
                    found = bool(field_data)
                    value = str(field_data) if field_data else 'Not found'
                
                if found:
                    st.success("✅ Found")
                    st.write(f"Extracted Text: {value}")
                else:
                    st.error("❌ Not Found")
            with col2:
                compliance = field_validation.get('compliance', 'Unknown')
                violations = field_validation.get('violations', [])
                if compliance == 'Pass':
                    st.success("✅ Compliant")
                else:
                    st.error("❌ Non-Compliant")
                if violations:
                    st.write("Issues:")
                    for violation in violations:
                        st.write(f"• {violation}")

def render_violations_and_recommendations():
    if not st.session_state.analysis_results:
        return
    results = st.session_state.analysis_results
    validation = results.get('validation_results', {})
    report = results.get('compliance_report', {})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚠ Violations Found")
        violations = validation.get('violations', [])
        if violations:
            for i, violation in enumerate(violations, 1):
                st.write(f"{i}. {violation}")
        else:
            st.success("No violations found! ✅")
    with col2:
        st.markdown("### 💡 Recommendations")
        recommendations = report.get('recommendations', [])
        if recommendations:
            for i, recommendation in enumerate(recommendations, 1):
                st.write(f"{i}. {recommendation}")
        else:
            st.success("No recommendations needed! ✅")

def render_compliance_chart():
    if not st.session_state.analysis_results:
        return
    results = st.session_state.analysis_results
    raw_data = results['raw_data']
    st.markdown("### 📈 Compliance Visualization")
    fields = []
    statuses = []
    colors = []
    field_labels = {
        'manufacturer': 'Manufacturer',
        'net_quantity': 'Net Quantity',
        'mrp': 'MRP',
        'consumer_care': 'Consumer Care',
        'mfg_date': 'Mfg Date',
        'country_origin': 'Country Origin',
        'product_name': 'Product Name'
    }
    for field, label in field_labels.items():
        field_data = raw_data.get(field, {})
        # Safely get compliance status
        if isinstance(field_data, dict):
            compliance = field_data.get('compliance', 'Fail')
        else:
            compliance = 'Fail'
        
        fields.append(label)
        statuses.append(1 if compliance == 'Pass' else 0)
        colors.append('#28a745' if compliance == 'Pass' else '#dc3545')
    fig = go.Figure(data=[
        go.Bar(
            x=fields,
            y=statuses,
            marker_color=colors,
            text=[status for status in statuses],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="Field Compliance Status",
        yaxis_title="Compliance Status",
        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Non-Compliant', 'Compliant']),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def render_sidebar():
    with st.sidebar:
        # Theme toggle button (single-click, always in sync)
        if st.session_state.theme == "light":
            if st.button("🌙 Dark Mode", use_container_width=True):
                set_theme("dark")
        else:
            if st.button("☀ Light Mode", use_container_width=True):
                set_theme("light")

        st.markdown("---")
        # Navigation
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Go to",
            [
                "Compliance Analysis",
                "ML Management",
                "Dataset Insights"
            ],
            index=0,
            label_visibility="collapsed"
        )
        st.session_state["active_page"] = page

        st.markdown("---")
        st.markdown("### 🎯 Quick Actions")
        if st.button("📄 Generate Report", use_container_width=True):
            generate_compliance_report()
        if st.button("📊 Export Data", use_container_width=True):
            export_compliance_data()
        st.markdown("---")
        st.markdown("**Team:** Tech Optimistic")
        st.markdown("**PS:** SIH25057")

def generate_compliance_report():
    if not st.session_state.analysis_results:
        st.warning("Please analyze an image first!")
        return
    results = st.session_state.analysis_results
    report = results.get('compliance_report', {})
    st.markdown("### 📄 Compliance Report")
    summary = report.get('summary', {})
    st.markdown(f"""
    Overall Status: {summary.get('overall_status', 'Unknown')}  
    Compliance Score: {summary.get('compliance_score', 0)}%  
    Fields Compliant: {summary.get('fields_compliant', 0)}/{summary.get('total_fields', 7)}
    """)
    st.markdown("Detailed Findings:")
    field_details = report.get('field_details', {})
    for field_name, details in field_details.items():
        status_emoji = "✅" if details.get('status') == 'Pass' else "❌"
        st.write(f"{status_emoji} {field_name}: {details.get('status', 'Unknown')}")
        issues = details.get('issues', [])
        if issues:
            for issue in issues:
                st.write(f"   • {issue}")

def export_compliance_data():
    if not st.session_state.analysis_results:
        st.warning("Please analyze an image first!")
        return
    
    results = st.session_state.analysis_results
    raw_data = results.get('raw_data', {})
    validation = results.get('validation_results', {})
    
    export_data = []
    
    # Safely access field validations
    field_validations = validation.get('field_validations', {})
    
    for field, field_data in raw_data.items():
        # Defaults
        found = False
        value = ''
        compliance = 'Unknown'
        violations = []
        
        if isinstance(field_data, dict):
            found = field_data.get('found', False)
            value = field_data.get('value', '')
            compliance = field_data.get('compliance', 'Unknown')
            violations = field_validations.get(field, {}).get('violations', [])
        else:
            # Non-dict field data; best effort
            value = str(field_data) if field_data is not None else ''
            found = bool(field_data)
        
        export_data.append({
            'Field': field.replace('_', ' ').title(),
            'Found': 'Yes' if found else 'No',
            'Value': value,
            'Compliance': compliance,
            'Violations': '; '.join(violations) if violations else 'None'
        })
    
    # Append overall summary
    overall_status = validation.get('overall_status', 'Unknown')
    compliance_score = validation.get('compliance_score', 0)
    violations_total = len(validation.get('violations', []))
    export_data.append({
        'Field': 'OVERALL SUMMARY',
        'Found': '-',
        'Value': f"Status: {overall_status}",
        'Compliance': f"Score: {compliance_score}%",
        'Violations': f"Total Violations: {violations_total}"
    })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.success(f"Compliance report ready for download ({len(export_data)} rows)")

def render_compliance_analysis():
    """Render the main compliance analysis page"""
    uploaded_file = render_file_upload()
    
    if st.session_state.analysis_results:
        render_compliance_overview()
        render_field_analysis()
        render_violations_and_recommendations()
        render_compliance_chart()
        
        # Show feedback loop UI after analysis
        st.markdown("---")
        st.session_state.feedback_loop.render_feedback_interface(
            st.session_state.analysis_results
        )
    else:
        st.markdown("###  Get Started")
        st.info("Upload a product packaging image to start the Legal Metrology compliance analysis.")
        with st.expander("Sample Use Cases"):
            st.markdown("#### Perfect for analyzing:")
            st.markdown("""
            - Food product packaging and labels
            - Cosmetic and personal care product labels
            - Electronic device packaging and warranty cards
            - Pharmaceutical product boxes and strips
            - Consumer goods packaging across categories
            - Import/Export product labels and documentation
            """)
            
            st.markdown("#### Target Industries:")
            st.markdown("""
            - E-commerce platforms and marketplaces
            - Food and Beverage manufacturers
            - FMCG and consumer goods companies
            - Import/Export trading businesses
            - Regulatory compliance and quality assurance teams
            - Third-party logistics and fulfillment centers
            """)
            
            st.markdown("#### Business Benefits:")
            st.markdown("""
            - Automated compliance verification reduces manual errors
            - Faster product onboarding for e-commerce platforms
            - Proactive identification of compliance violations
            - Detailed audit trails for regulatory requirements
            - Cost reduction in compliance management processes
            """)

def render_ml_management():
    """Render ML model management and training page"""
    st.markdown("### 🧠 ML Model Management")
    
    # Training status
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            dataset = st.session_state.dataset_manager.get_training_dataset()
            total_data = len(dataset['train']) + len(dataset['validation']) + len(dataset['test'])
        except:
            total_data = 0
        st.metric("Dataset Size", total_data)
    with col2:
        try:
            # Count feedback entries
            import sqlite3
            conn = sqlite3.connect(st.session_state.dataset_manager.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM user_feedback")
            feedback_count = cursor.fetchone()[0]
            conn.close()
        except:
            feedback_count = 0
        st.metric("Feedback Entries", feedback_count)
    with col3:
        try:
            # Count trained models
            import sqlite3
            conn = sqlite3.connect(st.session_state.dataset_manager.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ml_training_history WHERE is_active = TRUE")
            model_count = cursor.fetchone()[0]
            conn.close()
        except:
            model_count = 0
        st.metric("Trained Models", model_count)
    
    # Training controls
    st.markdown("#### 🧠 Model Training")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(" Train Complete Pipeline", use_container_width=True):
            with st.spinner("Training complete ML pipeline..."):
                try:
                    results = st.session_state.ml_trainer.train_complete_pipeline(min_samples=10)
                    model_version = results.get('model_version', 'Unknown')
                    training_results = results.get('training_results', {})
                    field_count = len(training_results.get('field_classifiers', {}))
                    st.success(f"✅ Complete pipeline trained! Model: {model_version}, Fields: {field_count}")
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    
    with col2:
        if st.button(" View Training Status", use_container_width=True):
            try:
                import sqlite3
                conn = sqlite3.connect(st.session_state.dataset_manager.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                SELECT model_version, accuracy, f1_score, training_samples, created_at 
                FROM ml_training_history 
                WHERE is_active = TRUE 
                ORDER BY created_at DESC LIMIT 5
                """)
                results = cursor.fetchall()
                conn.close()
                
                if results:
                    st.write("Recent Training Results:")
                    for row in results:
                        st.write(f"Model: {row[0]}, Accuracy: {row[1]:.3f}, F1: {row[2]:.3f}, Samples: {row[3]}")
                else:
                    st.info("No training history available.")
            except Exception as e:
                st.error(f"❌ Error loading training status: {str(e)}")
    
    # Model performance
    st.markdown("#### 📈 Model Performance")
    try:
        import sqlite3
        conn = sqlite3.connect(st.session_state.dataset_manager.db_path)
        performance_df = pd.read_sql_query("""
        SELECT model_version, accuracy, f1_score, training_samples, created_at as date
        FROM ml_training_history 
        ORDER BY created_at DESC LIMIT 10
        """, conn)
        conn.close()
        
        if not performance_df.empty:
            fig = px.line(performance_df, x='date', y='accuracy', title='Model Accuracy Over Time')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available yet. Train models to see analytics.")
    except Exception as e:
        st.info(f"Performance analytics will be available after model training: {str(e)}")
    
    # Feedback analytics
    st.markdown("#### 🔁 Feedback Analytics")
    try:
        import sqlite3
        conn = sqlite3.connect(st.session_state.dataset_manager.db_path)
        
        # User ratings from feedback_score in compliance_samples
        ratings_df = pd.read_sql_query("""
        SELECT feedback_score as rating, COUNT(*) as count
        FROM compliance_samples 
        WHERE feedback_score IS NOT NULL
        GROUP BY feedback_score
        """, conn)
        
        # Field corrections frequency
        corrections_df = pd.read_sql_query("""
        SELECT field_name, COUNT(*) as correction_count
        FROM user_feedback 
        WHERE field_name NOT LIKE '_%'
        GROUP BY field_name
        """, conn)
        
        conn.close()
        
        col1, col2 = st.columns(2)
        with col1:
            if not ratings_df.empty:
                fig = px.bar(ratings_df, x='rating', y='count', title='User Ratings Distribution')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rating data available yet.")
        
        with col2:
            if not corrections_df.empty:
                fig = px.bar(corrections_df, x='field_name', y='correction_count', 
                            title='Field Corrections Frequency')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No correction data available yet.")
    except Exception as e:
        st.info(f"Feedback analytics will be available after collecting user feedback: {str(e)}")

def render_dataset_insights():
    """Render dataset insights and analytics page"""
    st.markdown("### 📈 Dataset Insights & Analytics")
    
    try:
        # Get dataset statistics from database
        import sqlite3
        conn = sqlite3.connect(st.session_state.dataset_manager.db_path)
        cursor = conn.cursor()
        
        # Check if we have any data
        cursor.execute("SELECT COUNT(*) FROM compliance_samples")
        total_analyses = cursor.fetchone()[0]
        
        if total_analyses == 0:
            conn.close()
            st.info("No analysis data available yet. Analyze some images to see insights.")
            return
        
        # Overview metrics
        st.markdown("#### 🏆 Compliance Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        # Count compliant products (score >= 80)
        cursor.execute("SELECT COUNT(*) FROM compliance_samples WHERE compliance_score >= 80")
        compliant_count = cursor.fetchone()[0]
        
        # Average compliance score
        cursor.execute("SELECT AVG(compliance_score) FROM compliance_samples")
        avg_score = cursor.fetchone()[0] or 0
        
        with col1:
            st.metric("Total Analyses", total_analyses)
        with col2:
            st.metric("Compliant Products", compliant_count)
        with col3:
            compliance_rate = (compliant_count/total_analyses*100) if total_analyses > 0 else 0
            st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
        with col4:
            st.metric("Average Score", f"{avg_score:.1f}%")
        
        # Compliance score distribution
        st.markdown("#### 📉 Compliance Score Distribution")
        scores_df = pd.read_sql_query("""
        SELECT id as analysis_id, compliance_score, created_at as timestamp
        FROM compliance_samples 
        ORDER BY created_at
        """, conn)
        
        if not scores_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(scores_df, x='compliance_score', nbins=20, 
                                 title='Compliance Score Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(scores_df, x='timestamp', y='compliance_score', 
                             title='Compliance Scores Over Time')
                st.plotly_chart(fig, use_container_width=True)
        
        # Field-wise analysis (simplified)
        st.markdown("#### 🔍 Analysis Summary")
        cursor.execute("""
        SELECT 
            COUNT(*) as total_samples,
            AVG(compliance_score) as avg_score,
            COUNT(CASE WHEN compliance_score >= 80 THEN 1 END) as compliant,
            COUNT(CASE WHEN user_corrections IS NOT NULL THEN 1 END) as has_feedback
        FROM compliance_samples
        """)
        
        stats = cursor.fetchone()
        summary_df = pd.DataFrame([{
            'Metric': 'Total Samples',
            'Value': stats[0]
        }, {
            'Metric': 'Average Score',
            'Value': f"{stats[1]:.1f}%" if stats[1] else "0%"
        }, {
            'Metric': 'Compliant Samples',
            'Value': stats[2]
        }, {
            'Metric': 'Samples with Feedback',
            'Value': stats[3]
        }])
        st.dataframe(summary_df, use_container_width=True)
        
        # Export capabilities
        st.markdown("#### 📥 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Analysis History", use_container_width=True):
                history_df = pd.read_sql_query("""
                SELECT id, image_hash, compliance_score, extracted_fields, violations, created_at
                FROM compliance_samples 
                ORDER BY created_at DESC
                """, conn)
                
                if not history_df.empty:
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="Download Analysis History CSV",
                        data=csv,
                        file_name="analysis_history.csv",
                        mime="text/csv"
                    )
                    st.success(f"Exported {len(history_df)} analysis records")
                else:
                    st.warning("No analysis data to export")
        
        with col2:
            if st.button("Export Feedback Data", use_container_width=True):
                feedback_df = pd.read_sql_query("""
                SELECT uf.*, cs.image_hash, cs.compliance_score
                FROM user_feedback uf
                LEFT JOIN compliance_samples cs ON uf.sample_id = cs.id
                ORDER BY uf.created_at DESC
                """, conn)
                
                if not feedback_df.empty:
                    csv = feedback_df.to_csv(index=False)
                    st.download_button(
                        label="Download Feedback CSV",
                        data=csv,
                        file_name="feedback_data.csv",
                        mime="text/csv"
                    )
                    st.success(f"Exported {len(feedback_df)} feedback records")
                else:
                    st.warning("No feedback data to export")
        
        conn.close()
    
    except Exception as e:
        st.error(f"Error loading dataset insights: {str(e)}")
        # Close connection if it exists
        try:
            conn.close()
        except:
            pass

def render_footer():
    """Render compact footer with expandable about section"""
    # Add visual separator
    st.markdown("---")
    
    # Compact footer with expandable about section
    with st.expander("💼 About CompliAI - Features & Roadmap", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Features:**")
            st.markdown("""
            • AI-Powered Analysis (Gemini Pro Vision)  
            • 7-Field Legal Metrology Compliance  
            • Intelligent Scoring & Violation Reports  
            • Machine Learning Integration  
            • Export & Audit Capabilities  
            """)
        
        with col2:
            st.markdown("**Future Enhancements:**")
            st.markdown("""
            • Multi-Language OCR Support  
            • Batch Processing & API Integration  
            • Real-time Monitoring Systems  
            • Advanced Predictive Analytics  
            • Regulatory Updates Sync  
            """)
    
    # Compact footer bottom
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 1rem;'>"
        "CompliAI - AI-Powered Legal Metrology Compliance • "
        "Smart India Hackathon 2025 (SIH25057)"
        "</div>",
        unsafe_allow_html=True
    )

def main():
    initialize_session_state()
    render_header()
    render_sidebar()
    
    # Multi-page navigation
    active_page = st.session_state.get("active_page", "Compliance Analysis")
    
    if active_page == "Compliance Analysis":
        render_compliance_analysis()
    elif active_page == "ML Management":
        render_ml_management()
    elif active_page == "Dataset Insights":
        render_dataset_insights()
    
    # Render footer on all pages
    render_footer()

if __name__ == "__main__":
    main()
