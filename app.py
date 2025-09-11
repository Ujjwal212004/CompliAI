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
        st.session_state.ml_trainer = MLTrainer(st.session_state.dataset_manager)
    if 'feedback_loop' not in st.session_state:
        st.session_state.feedback_loop = FeedbackLoop(st.session_state.dataset_manager, st.session_state.ml_trainer)

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
        # Add a sample hash to the analysis results for feedback
        if 'sample_hash' not in st.session_state.analysis_results:
            import hashlib
            import time
            sample_data = f"{st.session_state.uploaded_image.name}_{time.time()}"
            st.session_state.analysis_results['sample_hash'] = hashlib.md5(sample_data.encode()).hexdigest()
        
        st.session_state.feedback_loop.render_feedback_interface(
            st.session_state.analysis_results
        )
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
    st.markdown("#### 🎧 Model Training")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Train Complete Pipeline", use_container_width=True):
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
        if st.button("📋 View Training Status", use_container_width=True):
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
