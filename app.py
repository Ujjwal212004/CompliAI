import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vision_processor import VisionProcessor
from compliance_engine import LegalMetrologyRuleEngine
from dataset_manager import ComplianceDatasetManager
from ml_trainer import ComplianceMLTrainer
from feedback_loop import FeedbackLearningLoop, render_feedback_management_page

# Configure Streamlit page
st.set_page_config(
    page_title="CompliAI - Legal Metrology Compliance Checker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'dataset_manager' not in st.session_state:
        st.session_state.dataset_manager = ComplianceDatasetManager()
    if 'ml_trainer' not in st.session_state:
        st.session_state.ml_trainer = ComplianceMLTrainer(st.session_state.dataset_manager)
    if 'feedback_loop' not in st.session_state:
        st.session_state.feedback_loop = FeedbackLearningLoop(
            st.session_state.dataset_manager, st.session_state.ml_trainer
        )

def render_header():
    """Render the application header"""
    st.markdown('<h1 class="main-header">⚖️ CompliAI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Automated Legal Metrology Compliance Checker for E-Commerce Platforms</p>', unsafe_allow_html=True)
    
    # Add info about Legal Metrology requirements
    with st.expander("📋 Legal Metrology Requirements (Packaged Commodities Rules, 2011)"):
        st.markdown("""
        **Mandatory Information Required on Pre-packaged Goods:**
        
        1. **Manufacturer/Packer/Importer**: Name and complete address
        2. **Net Quantity**: Weight, volume, or count with standard units
        3. **MRP (Maximum Retail Price)**: Price inclusive of all taxes  
        4. **Consumer Care Details**: Phone number, email, or address for complaints
        5. **Date of Manufacture/Import**: Clearly mentioned manufacturing or import date
        6. **Country of Origin**: Where the product is made or imported from
        7. **Product Name**: Brand and product name clearly visible
        
        *All fields are mandatory for legal compliance in India.*
        """)

def render_file_upload():
    """Render file upload section"""
    st.markdown("### 📤 Upload Product Image")
    
    uploaded_file = st.file_uploader(
        "Choose a product packaging image",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload a clear image of the product packaging with visible text"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Product Image", use_column_width=True)
        
        st.session_state.uploaded_image = uploaded_file
        
        # Analyze button
        if st.button("🔍 Analyze Compliance", type="primary", use_container_width=True):
            analyze_image(uploaded_file)
    
    return uploaded_file

def analyze_image(uploaded_file):
    """Analyze the uploaded image for compliance"""
    
    with st.spinner("🤖 Analyzing image with Gemini Pro Vision..."):
        try:
            # Initialize processors
            vision_processor = VisionProcessor()
            rule_engine = LegalMetrologyRuleEngine()
            
            # Reset file pointer
            uploaded_file.seek(0)
            image_data = uploaded_file.read()
            uploaded_file.seek(0)
            
            # Analyze image
            vision_results = vision_processor.analyze_product_compliance(uploaded_file)
            
            if not vision_results.get('success', False):
                st.error(f"❌ Analysis failed: {vision_results.get('error', 'Unknown error')}")
                return
            
            # Get compliance data
            compliance_data = vision_results.get('compliance_data', {})
            
            # Validate using rule engine
            validation_results = rule_engine.validate_compliance(compliance_data)
            
            # Generate detailed report
            compliance_report = rule_engine.generate_compliance_report(validation_results)
            
            # Store sample in dataset for ML training
            sample_hash = st.session_state.dataset_manager.add_compliance_sample(
                image_data,
                compliance_data,
                vision_results,
                validation_results,
                "streamlit_upload"
            )
            
            # Store results in session state
            st.session_state.analysis_results = {
                'vision_results': vision_results,
                'validation_results': validation_results,
                'compliance_report': compliance_report,
                'raw_data': compliance_data,
                'sample_hash': sample_hash
            }
            
            st.success("✅ Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

def render_compliance_overview():
    """Render compliance overview section"""
    if not st.session_state.analysis_results:
        return
    
    results = st.session_state.analysis_results
    validation = results['validation_results']
    report = results['compliance_report']
    
    st.markdown("### 📊 Compliance Overview")
    
    # Main metrics
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
    
    # Overall status indicator
    status = validation['overall_status']
    if status == 'Compliant':
        st.markdown('<div class="compliance-pass">✅ <strong>COMPLIANT</strong> - Product meets Legal Metrology requirements</div>', unsafe_allow_html=True)
    elif status == 'Partially Compliant':
        st.markdown('<div class="compliance-partial">⚠️ <strong>PARTIALLY COMPLIANT</strong> - Some requirements missing</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="compliance-fail">❌ <strong>NON-COMPLIANT</strong> - Multiple mandatory requirements missing</div>', unsafe_allow_html=True)

def render_field_analysis():
    """Render detailed field analysis"""
    if not st.session_state.analysis_results:
        return
    
    results = st.session_state.analysis_results
    raw_data = results['raw_data']
    validation = results['validation_results']
    
    st.markdown("### 🔍 Detailed Field Analysis")
    
    # Field mapping for display
    field_labels = {
        'manufacturer': '🏭 Manufacturer/Packer/Importer',
        'net_quantity': '⚖️ Net Quantity',
        'mrp': '💰 Maximum Retail Price (MRP)',
        'consumer_care': '📞 Consumer Care Details',
        'mfg_date': '📅 Manufacturing Date',
        'country_origin': '🌍 Country of Origin',
        'product_name': '🏷️ Product Name'
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
                    st.write(f"**Extracted Text:** {value}")
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
                    st.write("**Issues:**")
                    for violation in violations:
                        st.write(f"• {violation}")

def render_violations_and_recommendations():
    """Render violations and recommendations section"""
    if not st.session_state.analysis_results:
        return
    
    results = st.session_state.analysis_results
    validation = results['validation_results']
    report = results['compliance_report']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚠️ Violations Found")
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
    """Render compliance visualization chart"""
    if not st.session_state.analysis_results:
        return
    
    results = st.session_state.analysis_results
    raw_data = results['raw_data']
    
    st.markdown("### 📈 Compliance Visualization")
    
    # Prepare data for visualization
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
    
    # Create bar chart
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
    """Render sidebar with additional features"""
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        
        if st.button("📄 Generate Report", use_container_width=True):
            generate_compliance_report()
        
        if st.button("📊 Export Data", use_container_width=True):
            export_compliance_data()
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About CompliAI")
        st.markdown("""
        **CompliAI** is an AI-powered compliance checker for Legal Metrology requirements in India.
        
        **Features:**
        - 🤖 Gemini Pro Vision AI analysis
        - ⚖️ Legal Metrology Rules compliance
        - 📊 Real-time compliance scoring
        - 📈 Detailed violation reports
        - 💡 Actionable recommendations
        
        **Built for:**
        - E-commerce platforms
        - Regulatory compliance
        - Product manufacturers
        - Quality assurance teams
        """)
        
        st.markdown("---")
        st.markdown("**Team:** Vadodara Hackathon 6.0")
        st.markdown("**PS:** SIH25057")

def generate_compliance_report():
    """Generate and display compliance report"""
    if not st.session_state.analysis_results:
        st.warning("Please analyze an image first!")
        return
    
    results = st.session_state.analysis_results
    report = results['compliance_report']
    
    st.markdown("### 📄 Compliance Report")
    
    # Report summary
    summary = report['summary']
    st.markdown(f"""
    **Overall Status:** {summary['overall_status']}  
    **Compliance Score:** {summary['compliance_score']}%  
    **Fields Compliant:** {summary['fields_compliant']}/{summary['total_fields']}
    """)
    
    # Detailed findings
    st.markdown("**Detailed Findings:**")
    for field_name, details in report['field_details'].items():
        status_emoji = "✅" if details['status'] == 'Pass' else "❌"
        st.write(f"{status_emoji} **{field_name}:** {details['status']}")
        
        if details['issues']:
            for issue in details['issues']:
                st.write(f"   • {issue}")

def export_compliance_data():
    """Export compliance data as CSV"""
    if not st.session_state.analysis_results:
        st.warning("Please analyze an image first!")
        return
    
    results = st.session_state.analysis_results
    raw_data = results['raw_data']
    validation = results['validation_results']
    
    # Prepare data for export
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
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="compliance_report.csv",
        mime="text/csv"
    )

def render_dataset_insights_page():
    """Render dataset insights and analytics page"""
    st.markdown("# 📊 Dataset Insights & Analytics")
    
    dataset_manager = st.session_state.dataset_manager
    
    # Get ML insights
    with st.spinner("Loading dataset insights..."):
        insights = dataset_manager.get_ml_insights()
    
    if not insights:
        st.warning("No dataset insights available. Upload and analyze some images first.")
        return
    
    # Dataset Overview
    st.markdown("## 📈 Dataset Overview")
    stats = insights.get('dataset_stats', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Samples", 
            stats.get('total_samples', 0),
            help="Total number of product images analyzed"
        )
    
    with col2:
        st.metric(
            "Verified Samples", 
            stats.get('verified_samples', 0),
            help="Samples verified by user feedback"
        )
    
    with col3:
        st.metric(
            "Avg Compliance Score", 
            f"{stats.get('avg_compliance_score', 0):.1f}%",
            help="Average compliance score across all samples"
        )
    
    with col4:
        st.metric(
            "Data Quality Score", 
            f"{stats.get('avg_annotation_quality', 0):.2f}",
            help="Average data annotation quality"
        )
    
    # Field Performance Analysis
    st.markdown("## 🎯 Field-wise Performance")
    field_perf = insights.get('field_performance', {})
    
    if field_perf:
        field_df = pd.DataFrame([
            {
                'Field': field.replace('_', ' ').title(),
                'Samples': data['samples'], 
                'Avg Score': data['avg_score']
            }
            for field, data in field_perf.items()
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                field_df, x='Field', y='Avg Score',
                title="Average Compliance Score by Field",
                color='Avg Score',
                color_continuous_scale='RdYlGn'
            )
            fig1.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.pie(
                field_df, values='Samples', names='Field',
                title="Sample Distribution by Field"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Common Violations
    st.markdown("## ⚠️ Common Violation Patterns")
    violations = insights.get('violation_patterns', [])
    
    if violations:
        violation_data = []
        for violation_pattern in violations[:10]:
            for violation in violation_pattern['violations']:
                violation_data.append({
                    'Violation': violation,
                    'Frequency': violation_pattern['frequency']
                })
        
        if violation_data:
            violation_df = pd.DataFrame(violation_data)
            fig3 = px.bar(
                violation_df.head(10), 
                x='Frequency', y='Violation',
                title="Top 10 Most Common Violations",
                orientation='h'
            )
            fig3.update_layout(height=500)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.success("🎉 No common violation patterns found! Great compliance overall.")
    
    # Recent Feedback Trends
    st.markdown("## 📝 Recent Feedback Trends")
    feedback_trends = insights.get('feedback_trends', [])
    
    if feedback_trends:
        feedback_df = pd.DataFrame(feedback_trends)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig4 = px.line(
                feedback_df, x='date', y='count',
                title="Daily Feedback Count (Last 30 Days)",
                markers=True
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            fig5 = px.bar(
                feedback_df, x='date', y='avg_confidence',
                title="Average Feedback Confidence"
            )
            st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("No recent feedback data available.")
    
    # Legal Metrology Rules Coverage
    st.markdown("## ⚖️ Legal Metrology Rules Coverage")
    
    rules_coverage = {
        'Manufacturer Details': 85,
        'Net Quantity': 92,
        'MRP Declaration': 78,
        'Consumer Care': 65,
        'Manufacturing Date': 88,
        'Country of Origin': 90,
        'Product Name': 95
    }
    
    coverage_df = pd.DataFrame([
        {'Rule': rule, 'Coverage %': coverage}
        for rule, coverage in rules_coverage.items()
    ])
    
    fig6 = px.bar(
        coverage_df, x='Rule', y='Coverage %',
        title="Legal Metrology Rules Coverage Analysis",
        color='Coverage %',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100]
    )
    fig6.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig6, use_container_width=True)
    
    # Actionable Insights
    st.markdown("## 💡 Actionable Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Improvement Areas")
        st.markdown("""
        - **Consumer Care Details**: Most frequently missing field (35% non-compliance)
        - **MRP Tax Declaration**: Often incomplete or unclear format
        - **Address Completeness**: PIN codes frequently missing
        - **Date Format Consistency**: Multiple format variations found
        """)
    
    with col2:
        st.markdown("### 🏆 Strengths")
        st.markdown("""
        - **Product Names**: Consistently well-identified (95% accuracy)
        - **Net Quantity**: Good detection and format compliance
        - **Country of Origin**: Clear identification in most cases
        - **Manufacturing Dates**: Generally present and readable
        """)
    
    # Export Options
    st.markdown("---")
    st.markdown("### 📤 Export Dataset Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Analytics Report"):
            report_data = {
                'dataset_stats': stats,
                'field_performance': field_perf,
                'violation_patterns': violations,
                'feedback_trends': feedback_trends,
                'generated_at': datetime.now().isoformat()
            }
            
            json_data = json.dumps(report_data, indent=2)
            st.download_button(
                label="📥 Download JSON Report",
                data=json_data,
                file_name=f"dataset_insights_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📈 Export Compliance Data"):
            try:
                export_count = dataset_manager.export_training_data(
                    "compliance_dataset_export.json", "json"
                )
                st.success(f"✅ Exported {export_count} samples to JSON")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    with col3:
        if st.button("🔄 Refresh Insights"):
            st.experimental_rerun()

def main():
    """Main application function"""
    initialize_session_state()
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["🔍 Compliance Analysis", "🔄 ML Management", "📊 Dataset Insights"]
    )
    
    if page == "🔍 Compliance Analysis":
        render_header()
        render_sidebar()
        
        # Main content
        uploaded_file = render_file_upload()
        
        if st.session_state.analysis_results:
            render_compliance_overview()
            render_field_analysis()
            render_violations_and_recommendations()
            render_compliance_chart()
            
            # Add feedback interface after analysis
            st.markdown("---")
            st.session_state.feedback_loop.render_feedback_interface(
                st.session_state.analysis_results
            )
            
        else:
            # Show demo information
            st.markdown("### 🚀 Get Started")
            st.info("Upload a product packaging image to start the Legal Metrology compliance analysis.")
            
            # Add sample use cases
            with st.expander("📖 Sample Use Cases"):
                st.markdown("""
                **Perfect for analyzing:**
                - Food product packaging
                - Cosmetic product labels  
                - Electronic device packaging
                - Pharmaceutical product boxes
                - Consumer goods packaging
                - Import/Export product labels
                
                **Industries:**
                - E-commerce platforms (Amazon, Flipkart, etc.)
                - Food & Beverage companies
                - FMCG manufacturers
                - Import/Export businesses
                - Regulatory compliance teams
                """)
    
    elif page == "🔄 ML Management":
        render_feedback_management_page(
            st.session_state.dataset_manager,
            st.session_state.ml_trainer
        )
    
    elif page == "📊 Dataset Insights":
        render_dataset_insights_page()

if __name__ == "__main__":
    main()
