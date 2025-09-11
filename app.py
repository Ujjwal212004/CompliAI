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
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .compliance-pass {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #28a745;
    }
    .compliance-fail {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #dc3545;
    }
    .compliance-partial {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #ffc107;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
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
        st.session_state.ml_trainer = MLTrainer()
    if 'feedback_loop' not in st.session_state:
        st.session_state.feedback_loop = FeedbackLoop(st.session_state.dataset_manager)

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
    st.markdown('<h1 class="main-header">⚖ CompliAI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Automated Legal Metrology Compliance Checker for E-Commerce Platforms</p>', unsafe_allow_html=True)
    with st.expander("📋 Legal Metrology Requirements (Packaged Commodities Rules, 2011)"):
        st.markdown("""
        Mandatory Information Required on Pre-packaged Goods:
        1. Manufacturer/Packer/Importer: Name and complete address
        2. Net Quantity: Weight, volume, or count with standard units
        3. MRP (Maximum Retail Price): Price inclusive of all taxes  
        4. Consumer Care Details: Phone number, email, or address for complaints
        5. Date of Manufacture/Import: Clearly mentioned manufacturing or import date
        6. Country of Origin: Where the product is made or imported from
        7. Product Name: Brand and product name clearly visible
        All fields are mandatory for legal compliance in India.
        """)

def render_file_upload():
    st.markdown("### 📤 Upload Product Image")
    uploaded_file = st.file_uploader(
        "Choose a product packaging image",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload a clear image of the product packaging with visible text"
    )
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
    with st.spinner("🤖 Analyzing image with Gemini Pro Vision..."):
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
                'validation_results': validation_results,
                'compliance_report': compliance_report,
                'raw_data': compliance_data
            }
            
            # Store in dataset for ML training
            try:
                uploaded_file.seek(0)
                image_data = uploaded_file.read()
                st.session_state.dataset_manager.store_analysis(
                    image_data=image_data,
                    extracted_text=vision_results.get('extracted_text', ''),
                    compliance_results=validation_results,
                    filename=uploaded_file.name
                )
            except Exception as storage_error:
                st.warning(f"Results analyzed but not stored in dataset: {str(storage_error)}")
            
            st.success("✅ Analysis completed successfully!")
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

def render_compliance_overview():
    if not st.session_state.analysis_results:
        return
    results = st.session_state.analysis_results
    validation = results['validation_results']
    report = results['compliance_report']
    st.markdown("### 📊 Compliance Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score = validation['compliance_score']
        st.metric(
            "Compliance Score",
            f"{score}%",
            delta=f"{score-70}%" if score >= 70 else f"{score-70}%"
        )
    with col2:
        status = validation['overall_status']
        st.metric("Overall Status", status)
    with col3:
        found = validation['mandatory_fields_found']
        total = validation['total_mandatory_fields']
        st.metric("Fields Found", f"{found}/{total}")
    with col4:
        violations = len(validation['violations'])
        st.metric("Violations", violations)
    status = validation['overall_status']
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
        'manufacturer': '🏭 Manufacturer/Packer/Importer',
        'net_quantity': '⚖ Net Quantity',
        'mrp': '💰 Maximum Retail Price (MRP)',
        'consumer_care': '📞 Consumer Care Details',
        'mfg_date': '📅 Manufacturing Date',
        'country_origin': '🌍 Country of Origin',
        'product_name': '🏷 Product Name'
    }
    for field, label in field_labels.items():
        with st.expander(f"{label}", expanded=False):
            field_data = raw_data.get(field, {})
            field_validation = validation['field_validations'].get(field, {})
            col1, col2 = st.columns([1, 1])
            with col1:
                found = field_data.get('found', False)
                value = field_data.get('value', 'Not found')
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
    validation = results['validation_results']
    report = results['compliance_report']
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚠ Violations Found")
        violations = validation['violations']
        if violations:
            for i, violation in enumerate(violations, 1):
                st.write(f"{i}. {violation}")
        else:
            st.success("No violations found! ✅")
    with col2:
        st.markdown("### 💡 Recommendations")
        recommendations = report['recommendations']
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
        compliance = field_data.get('compliance', 'Fail')
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
        st.markdown("### ℹ About CompliAI")
        st.markdown("""
        CompliAI is an AI-powered compliance checker for Legal Metrology requirements in India.
        Features:
        - 🤖 Gemini Pro Vision AI analysis
        - ⚖ Legal Metrology Rules compliance
        - 📊 Real-time compliance scoring
        - 📈 Detailed violation reports
        - 💡 Actionable recommendations
        - 🔁 Feedback learning loop for continuous improvement
        - 🧠 Field-wise ML classifiers and compliance predictors
        Built for:
        - E-commerce platforms
        - Regulatory compliance
        - Product manufacturers
        - Quality assurance teams
        """)
        st.markdown("---")
        st.markdown("Team: Vadodara Hackathon 6.0")
        st.markdown("PS: SIH25057")

def generate_compliance_report():
    if not st.session_state.analysis_results:
        st.warning("Please analyze an image first!")
        return
    results = st.session_state.analysis_results
    report = results['compliance_report']
    st.markdown("### 📄 Compliance Report")
    summary = report['summary']
    st.markdown(f"""
    Overall Status: {summary['overall_status']}  
    Compliance Score: {summary['compliance_score']}%  
    Fields Compliant: {summary['fields_compliant']}/{summary['total_fields']}
    """)
    st.markdown("Detailed Findings:")
    for field_name, details in report['field_details'].items():
        status_emoji = "✅" if details['status'] == 'Pass' else "❌"
        st.write(f"{status_emoji} {field_name}: {details['status']}")
        if details['issues']:
            for issue in details['issues']:
                st.write(f"   • {issue}")

def export_compliance_data():
    if not st.session_state.analysis_results:
        st.warning("Please analyze an image first!")
        return
    results = st.session_state.analysis_results
    raw_data = results['raw_data']
    validation = results['validation_results']
    export_data = []
    for field, field_data in raw_data.items():
        field_validation = validation['field_validations'].get(field, {})
        export_data.append({
            'Field': field.replace('_', ' ').title(),
            'Found': field_data.get('found', False),
            'Value': field_data.get('value', ''),
            'Compliance': field_data.get('compliance', 'Unknown'),
            'Violations': '; '.join(field_validation.get('violations', []))
        })
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="compliance_report.csv",
        mime="text/csv"
    )

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
        st.markdown("### 🔁 Feedback & Learning")
        feedback_ui = st.session_state.feedback_loop.render_feedback_ui(
            st.session_state.analysis_results
        )
        
        if feedback_ui['submitted']:
            # Store feedback and potentially retrain
            try:
                st.session_state.feedback_loop.collect_feedback(
                    analysis_id=feedback_ui['analysis_id'],
                    user_corrections=feedback_ui['corrections'],
                    overall_rating=feedback_ui['rating']
                )
                st.success("✅ Feedback submitted successfully! Thank you for helping improve CompliAI.")
            except Exception as e:
                st.error(f"❌ Error submitting feedback: {str(e)}")
    else:
        st.markdown("### 🚀 Get Started")
        st.info("Upload a product packaging image to start the Legal Metrology compliance analysis.")
        with st.expander("📖 Sample Use Cases"):
            st.markdown("""
            Perfect for analyzing:
            - Food product packaging
            - Cosmetic product labels  
            - Electronic device packaging
            - Pharmaceutical product boxes
            - Consumer goods packaging
            - Import/Export product labels
            
            Industries:
            - E-commerce platforms (Amazon, Flipkart, etc.)
            - Food & Beverage companies
            - FMCG manufacturers
            - Import/Export businesses
            - Regulatory compliance teams
            """)

def render_ml_management():
    """Render ML model management and training page"""
    st.markdown("### 🧠 ML Model Management")
    
    # Training status
    col1, col2, col3 = st.columns(3)
    with col1:
        total_data = len(st.session_state.dataset_manager.get_analysis_history())
        st.metric("Dataset Size", total_data)
    with col2:
        feedback_data = len(st.session_state.dataset_manager.get_feedback_data())
        st.metric("Feedback Entries", feedback_data)
    with col3:
        # Check if models exist
        try:
            models_info = st.session_state.ml_trainer.get_models_info()
            model_count = len(models_info) if models_info else 0
        except:
            model_count = 0
        st.metric("Trained Models", model_count)
    
    # Training controls
    st.markdown("#### 🎧 Model Training")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Train Field Classifiers", use_container_width=True):
            with st.spinner("Training field classification models..."):
                try:
                    results = st.session_state.ml_trainer.train_field_classifiers()
                    st.success(f"✅ Field classifiers trained! Accuracy: {results.get('avg_accuracy', 'N/A')}")
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    
    with col2:
        if st.button("🎯 Train Compliance Predictor", use_container_width=True):
            with st.spinner("Training compliance prediction model..."):
                try:
                    results = st.session_state.ml_trainer.train_compliance_predictor()
                    st.success(f"✅ Compliance predictor trained! Accuracy: {results.get('accuracy', 'N/A')}")
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    
    # Model performance
    st.markdown("#### 📈 Model Performance")
    try:
        performance_data = st.session_state.feedback_loop.get_performance_analytics()
        if performance_data and len(performance_data) > 0:
            df = pd.DataFrame(performance_data)
            fig = px.line(df, x='date', y='accuracy', title='Model Performance Over Time')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available yet. Train models and collect feedback to see analytics.")
    except Exception as e:
        st.info("Performance analytics will be available after model training and feedback collection.")
    
    # Feedback analytics
    st.markdown("#### 🔁 Feedback Analytics")
    try:
        feedback_trends = st.session_state.feedback_loop.get_feedback_trends()
        if feedback_trends:
            col1, col2 = st.columns(2)
            with col1:
                ratings_df = pd.DataFrame(feedback_trends['ratings'])
                if not ratings_df.empty:
                    fig = px.histogram(ratings_df, x='rating', title='User Ratings Distribution')
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                corrections_df = pd.DataFrame(feedback_trends['corrections'])
                if not corrections_df.empty:
                    fig = px.bar(corrections_df, x='field', y='correction_count', 
                                title='Field Corrections Frequency')
                    st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info("Feedback analytics will be available after collecting user feedback.")

def render_dataset_insights():
    """Render dataset insights and analytics page"""
    st.markdown("### 📈 Dataset Insights & Analytics")
    
    try:
        # Get dataset statistics
        history = st.session_state.dataset_manager.get_analysis_history()
        if not history:
            st.info("No analysis data available yet. Analyze some images to see insights.")
            return
        
        # Overview metrics
        st.markdown("#### 🏆 Compliance Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        total_analyses = len(history)
        compliant_count = sum(1 for h in history if h['compliance_score'] >= 80)
        avg_score = np.mean([h['compliance_score'] for h in history]) if history else 0
        
        with col1:
            st.metric("Total Analyses", total_analyses)
        with col2:
            st.metric("Compliant Products", compliant_count)
        with col3:
            st.metric("Compliance Rate", f"{(compliant_count/total_analyses*100):.1f}%")
        with col4:
            st.metric("Average Score", f"{avg_score:.1f}%")
        
        # Compliance score distribution
        st.markdown("#### 📉 Compliance Score Distribution")
        scores_df = pd.DataFrame({
            'analysis_id': [h['id'] for h in history],
            'compliance_score': [h['compliance_score'] for h in history],
            'timestamp': [h['timestamp'] for h in history]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(scores_df, x='compliance_score', nbins=20, 
                             title='Compliance Score Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(scores_df, x='timestamp', y='compliance_score', 
                         title='Compliance Scores Over Time')
            st.plotly_chart(fig, use_container_width=True)
        
        # Field-wise analysis
        st.markdown("#### 🔍 Field-wise Performance")
        field_stats = st.session_state.dataset_manager.get_field_statistics()
        if field_stats:
            field_df = pd.DataFrame([
                {'Field': field, 'Detection Rate': f"{stats['detection_rate']:.1f}%"}
                for field, stats in field_stats.items()
            ])
            st.dataframe(field_df, use_container_width=True)
        
        # Export capabilities
        st.markdown("#### 📥 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Analysis History", use_container_width=True):
                history_df = pd.DataFrame(history)
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="analysis_history.csv",
                    mime="text/csv"
                )
        
        with col2:
            feedback_data = st.session_state.dataset_manager.get_feedback_data()
            if feedback_data:
                if st.button("Export Feedback Data", use_container_width=True):
                    feedback_df = pd.DataFrame(feedback_data)
                    csv = feedback_df.to_csv(index=False)
                    st.download_button(
                        label="Download Feedback CSV",
                        data=csv,
                        file_name="feedback_data.csv",
                        mime="text/csv"
                    )
    
    except Exception as e:
        st.error(f"Error loading dataset insights: {str(e)}")

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

if __name__ == "__main__":
    main()
