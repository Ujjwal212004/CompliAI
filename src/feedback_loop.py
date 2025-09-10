import streamlit as st
import pandas as pd
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import asyncio
import threading
import time

from dataset_manager import ComplianceDatasetManager
from ml_trainer import ComplianceMLTrainer

class FeedbackLearningLoop:
    """
    Feedback learning loop system for continuous model improvement
    Handles user feedback collection, data annotation, and automated retraining
    """
    
    def __init__(self, dataset_manager: ComplianceDatasetManager, ml_trainer: ComplianceMLTrainer):
        self.dataset_manager = dataset_manager
        self.ml_trainer = ml_trainer
        self.feedback_threshold = 50  # Retrain after 50 new feedbacks
        self.last_retrain_time = None
        self.retrain_interval = timedelta(days=7)  # Retrain weekly
        
    def collect_user_feedback(self, sample_hash: str, field_corrections: Dict, 
                            overall_rating: int, user_id: str = "anonymous") -> bool:
        """Collect and store user feedback"""
        
        # Store feedback in dataset manager
        success = self.dataset_manager.add_user_feedback(
            sample_hash, field_corrections, overall_rating, user_id
        )
        
        if success:
            # Check if retraining is needed
            self._check_retrain_trigger()
            return True
        
        return False
    
    def _check_retrain_trigger(self):
        """Check if model retraining should be triggered"""
        
        conn = sqlite3.connect(self.dataset_manager.db_path)
        cursor = conn.cursor()
        
        try:
            # Count recent feedback
            cursor.execute("""
            SELECT COUNT(*) FROM user_feedback 
            WHERE created_at > datetime('now', '-7 days')
            """)
            
            recent_feedback_count = cursor.fetchone()[0]
            
            # Check last retrain time
            cursor.execute("""
            SELECT MAX(created_at) FROM ml_training_history
            """)
            
            last_retrain = cursor.fetchone()[0]
            
            should_retrain = False
            
            # Trigger conditions
            if recent_feedback_count >= self.feedback_threshold:
                should_retrain = True
                reason = f"Feedback threshold reached: {recent_feedback_count} feedbacks"
            
            elif last_retrain:
                last_retrain_dt = datetime.fromisoformat(last_retrain)
                if datetime.now() - last_retrain_dt > self.retrain_interval:
                    should_retrain = True
                    reason = "Weekly retrain interval reached"
            
            elif not last_retrain:
                should_retrain = True
                reason = "No previous training found"
            
            if should_retrain:
                st.info(f"🤖 Automated retraining triggered: {reason}")
                self._trigger_background_retrain()
                
        except Exception as e:
            st.error(f"Error checking retrain trigger: {str(e)}")
        finally:
            conn.close()
    
    def _trigger_background_retrain(self):
        """Trigger background model retraining"""
        
        def retrain_worker():
            try:
                st.info("🔄 Starting background model retraining...")
                results = self.ml_trainer.train_complete_pipeline(min_samples=30)
                st.success(f"✅ Model retrained successfully! Version: {results['model_version']}")
                
                # Update retrain timestamp
                self.last_retrain_time = datetime.now()
                
            except Exception as e:
                st.error(f"❌ Model retraining failed: {str(e)}")
        
        # Start background thread for retraining
        thread = threading.Thread(target=retrain_worker)
        thread.daemon = True
        thread.start()
    
    def render_feedback_interface(self, analysis_results: Dict):
        """Render user feedback collection interface"""
        
        st.markdown("### 📝 Provide Feedback for Model Improvement")
        
        if not analysis_results:
            st.warning("No analysis results to provide feedback on.")
            return
        
        sample_hash = analysis_results.get('sample_hash')
        if not sample_hash:
            st.error("No sample hash found in analysis results.")
            return
        
        raw_data = analysis_results.get('raw_data', {})
        validation_results = analysis_results.get('validation_results', {})
        
        st.markdown("#### Overall Rating")
        overall_rating = st.slider(
            "How accurate was the overall analysis?",
            min_value=1, max_value=5, value=3,
            help="1 = Very Poor, 5 = Excellent"
        )
        
        st.markdown("#### Field-by-Field Corrections")
        
        field_corrections = {}
        
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
            with st.expander(f"Correct {label}", expanded=False):
                field_data = raw_data.get(field, {})
                field_validation = validation_results.get('field_validations', {}).get(field, {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Current Detection:**")
                    original_value = field_data.get('value', 'Not detected')
                    original_found = field_data.get('found', False)
                    original_compliance = field_data.get('compliance', 'Unknown')
                    
                    st.write(f"Found: {original_found}")
                    st.write(f"Value: {original_value}")
                    st.write(f"Compliance: {original_compliance}")
                
                with col2:
                    st.write("**Your Corrections:**")
                    
                    # Correction toggles
                    correct_detection = st.checkbox(
                        f"Correction needed for {field}",
                        key=f"correct_{field}"
                    )
                    
                    if correct_detection:
                        # Field found toggle
                        corrected_found = st.checkbox(
                            "Field is actually present",
                            value=original_found,
                            key=f"found_{field}"
                        )
                        
                        # Corrected value
                        corrected_value = st.text_area(
                            "Correct value:",
                            value=original_value if original_found else "",
                            height=60,
                            key=f"value_{field}"
                        )
                        
                        # Corrected compliance
                        corrected_compliance = st.selectbox(
                            "Correct compliance:",
                            options=["Pass", "Fail"],
                            index=0 if original_compliance == "Pass" else 1,
                            key=f"compliance_{field}"
                        )
                        
                        # Confidence in correction
                        correction_confidence = st.slider(
                            "How confident are you in this correction?",
                            min_value=0.1, max_value=1.0, value=0.9, step=0.1,
                            key=f"confidence_{field}"
                        )
                        
                        # Notes
                        correction_notes = st.text_area(
                            "Additional notes (optional):",
                            height=40,
                            key=f"notes_{field}"
                        )
                        
                        # Store correction data
                        field_corrections[field] = {
                            'original_value': original_value,
                            'corrected_value': corrected_value,
                            'original_found': original_found,
                            'corrected_found': corrected_found,
                            'original_compliance': original_compliance,
                            'corrected_compliance': corrected_compliance,
                            'confidence': correction_confidence,
                            'correction_type': 'user_correction',
                            'category': 'field_correction',
                            'notes': correction_notes
                        }
        
        # Additional feedback
        st.markdown("#### Additional Feedback")
        
        col1, col2 = st.columns(2)
        
        with col1:
            image_quality = st.selectbox(
                "Image Quality:",
                options=["Excellent", "Good", "Fair", "Poor"],
                index=1
            )
        
        with col2:
            detection_speed = st.selectbox(
                "Analysis Speed:",
                options=["Very Fast", "Fast", "Acceptable", "Slow"],
                index=1
            )
        
        general_comments = st.text_area(
            "General Comments (optional):",
            placeholder="Any other feedback about the analysis accuracy, speed, or suggestions for improvement...",
            height=80
        )
        
        # User identification (optional)
        user_id = st.text_input(
            "Your ID/Name (optional, for feedback tracking):",
            placeholder="Enter your name or ID if you want to track your contributions"
        )
        
        # Submit feedback button
        if st.button("📤 Submit Feedback", type="primary"):
            
            # Add metadata to corrections
            metadata = {
                'image_quality': image_quality,
                'detection_speed': detection_speed,
                'general_comments': general_comments,
                'feedback_timestamp': datetime.now().isoformat()
            }
            
            # Add metadata as a special field
            field_corrections['_metadata'] = {
                'correction_type': 'metadata',
                'category': 'user_feedback',
                'corrected_value': json.dumps(metadata),
                'confidence': 1.0,
                'notes': general_comments
            }
            
            # Submit feedback
            success = self.collect_user_feedback(
                sample_hash, field_corrections, overall_rating, 
                user_id if user_id else "anonymous"
            )
            
            if success:
                st.success("✅ Thank you! Your feedback has been submitted and will help improve the model.")
                st.balloons()
                
                # Show contribution stats
                self.show_contribution_stats(user_id if user_id else "anonymous")
                
            else:
                st.error("❌ Failed to submit feedback. Please try again.")
    
    def show_contribution_stats(self, user_id: str):
        """Show user contribution statistics"""
        
        conn = sqlite3.connect(self.dataset_manager.db_path)
        cursor = conn.cursor()
        
        try:
            # User's contribution count
            cursor.execute("""
            SELECT COUNT(*) FROM user_feedback WHERE user_id = ?
            """, (user_id,))
            
            user_contributions = cursor.fetchone()[0]
            
            # Total contributions
            cursor.execute("SELECT COUNT(*) FROM user_feedback")
            total_contributions = cursor.fetchone()[0]
            
            # User's average confidence
            cursor.execute("""
            SELECT AVG(confidence_score) FROM user_feedback WHERE user_id = ?
            """, (user_id,))
            
            avg_confidence = cursor.fetchone()[0] or 0
            
            st.markdown("#### 🎯 Your Contribution Impact")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Your Contributions", user_contributions)
            
            with col2:
                percentage = (user_contributions / max(total_contributions, 1)) * 100
                st.metric("Contribution %", f"{percentage:.1f}%")
            
            with col3:
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            
            if user_contributions >= 10:
                st.success("🏆 Expert Contributor! Your feedback is highly valued.")
            elif user_contributions >= 5:
                st.info("⭐ Active Contributor! Keep up the great work.")
            
        except Exception as e:
            st.warning(f"Could not load contribution stats: {str(e)}")
        finally:
            conn.close()
    
    def render_feedback_analytics(self):
        """Render feedback analytics dashboard"""
        
        st.markdown("### 📊 Feedback Analytics Dashboard")
        
        conn = sqlite3.connect(self.dataset_manager.db_path)
        
        try:
            # Feedback over time
            feedback_df = pd.read_sql_query("""
            SELECT 
                DATE(created_at) as feedback_date,
                COUNT(*) as feedback_count,
                AVG(confidence_score) as avg_confidence
            FROM user_feedback
            WHERE created_at >= date('now', '-30 days')
            GROUP BY DATE(created_at)
            ORDER BY feedback_date
            """, conn)
            
            if not feedback_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Feedback Volume (Last 30 Days)")
                    fig1 = px.line(feedback_df, x='feedback_date', y='feedback_count',
                                  title="Daily Feedback Count")
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.markdown("#### Average Confidence")
                    fig2 = px.bar(feedback_df, x='feedback_date', y='avg_confidence',
                                 title="Daily Average Confidence")
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Field-wise feedback
            field_feedback_df = pd.read_sql_query("""
            SELECT 
                field_name,
                COUNT(*) as correction_count,
                AVG(confidence_score) as avg_confidence
            FROM user_feedback
            WHERE field_name NOT LIKE '_%'
            GROUP BY field_name
            ORDER BY correction_count DESC
            """, conn)
            
            if not field_feedback_df.empty:
                st.markdown("#### Field-wise Feedback")
                fig3 = px.bar(field_feedback_df, x='field_name', y='correction_count',
                             title="Corrections by Field")
                st.plotly_chart(fig3, use_container_width=True)
            
            # Top contributors
            contributor_df = pd.read_sql_query("""
            SELECT 
                user_id,
                COUNT(*) as contribution_count,
                AVG(confidence_score) as avg_confidence
            FROM user_feedback
            GROUP BY user_id
            ORDER BY contribution_count DESC
            LIMIT 10
            """, conn)
            
            if not contributor_df.empty:
                st.markdown("#### Top Contributors")
                st.dataframe(contributor_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading feedback analytics: {str(e)}")
        finally:
            conn.close()
    
    def render_model_performance_dashboard(self):
        """Render model performance monitoring dashboard"""
        
        st.markdown("### 🎯 Model Performance Monitoring")
        
        conn = sqlite3.connect(self.dataset_manager.db_path)
        
        try:
            # Training history
            training_df = pd.read_sql_query("""
            SELECT 
                model_version,
                accuracy,
                f1_score,
                training_samples,
                created_at
            FROM ml_training_history
            ORDER BY created_at DESC
            LIMIT 10
            """, conn)
            
            if not training_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Model Accuracy Trend")
                    fig1 = px.line(training_df[::-1], x='created_at', y='accuracy',
                                  title="Model Accuracy Over Time",
                                  markers=True)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.markdown("#### F1 Score Trend")
                    fig2 = px.line(training_df[::-1], x='created_at', y='f1_score',
                                  title="Model F1 Score Over Time",
                                  markers=True)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Training samples vs performance
                st.markdown("#### Training Data vs Performance")
                fig3 = px.scatter(training_df, x='training_samples', y='accuracy',
                                 size='f1_score', hover_data=['model_version'],
                                 title="Training Samples vs Accuracy")
                st.plotly_chart(fig3, use_container_width=True)
            
            # Current model stats
            st.markdown("#### Current Active Model")
            
            active_model_df = pd.read_sql_query("""
            SELECT 
                model_version,
                accuracy,
                precision_score,
                recall_score,
                f1_score,
                training_samples,
                created_at
            FROM ml_training_history
            WHERE is_active = TRUE
            ORDER BY created_at DESC
            LIMIT 1
            """, conn)
            
            if not active_model_df.empty:
                model_info = active_model_df.iloc[0]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{model_info['accuracy']:.3f}")
                
                with col2:
                    st.metric("Precision", f"{model_info['precision_score']:.3f}")
                
                with col3:
                    st.metric("Recall", f"{model_info['recall_score']:.3f}")
                
                with col4:
                    st.metric("F1 Score", f"{model_info['f1_score']:.3f}")
                
                st.info(f"Model Version: {model_info['model_version']} | "
                       f"Training Samples: {model_info['training_samples']} | "
                       f"Created: {model_info['created_at'][:10]}")
            
        except Exception as e:
            st.error(f"Error loading model performance data: {str(e)}")
        finally:
            conn.close()
    
    def render_data_quality_dashboard(self):
        """Render data quality monitoring dashboard"""
        
        st.markdown("### 📈 Data Quality Monitoring")
        
        insights = self.dataset_manager.get_ml_insights()
        
        if insights:
            # Dataset statistics
            stats = insights.get('dataset_stats', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", stats.get('total_samples', 0))
            
            with col2:
                st.metric("Verified Samples", stats.get('verified_samples', 0))
            
            with col3:
                st.metric("Avg Compliance", f"{stats.get('avg_compliance_score', 0):.1f}%")
            
            with col4:
                st.metric("Avg Quality", f"{stats.get('avg_annotation_quality', 0):.2f}")
            
            # Field performance
            field_perf = insights.get('field_performance', {})
            if field_perf:
                st.markdown("#### Field-wise Performance")
                
                field_df = pd.DataFrame([
                    {'Field': field, 'Samples': data['samples'], 'Avg Score': data['avg_score']}
                    for field, data in field_perf.items()
                ])
                
                fig = px.bar(field_df, x='Field', y='Avg Score',
                           title="Average Compliance Score by Field")
                st.plotly_chart(fig, use_container_width=True)
            
            # Violation patterns
            violations = insights.get('violation_patterns', [])
            if violations:
                st.markdown("#### Common Violation Patterns")
                
                for i, violation_data in enumerate(violations[:5]):
                    with st.expander(f"Violation Pattern #{i+1} (Frequency: {violation_data['frequency']})"):
                        for violation in violation_data['violations']:
                            st.write(f"• {violation}")
            
            # Feedback trends
            feedback_trends = insights.get('feedback_trends', [])
            if feedback_trends:
                st.markdown("#### Recent Feedback Trends")
                
                feedback_df = pd.DataFrame(feedback_trends)
                fig = px.line(feedback_df, x='date', y='count',
                             title="Daily Feedback Count (Last 30 Days)")
                st.plotly_chart(fig, use_container_width=True)
    
    def trigger_manual_retrain(self):
        """Trigger manual model retraining"""
        
        st.markdown("### 🔄 Manual Model Retraining")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_samples = st.number_input(
                "Minimum Training Samples",
                min_value=10, max_value=1000, value=50,
                help="Minimum number of samples needed for training"
            )
        
        with col2:
            include_recent_only = st.checkbox(
                "Use only recent data (last 30 days)",
                help="Train only on recent feedback data"
            )
        
        if st.button("🚀 Start Manual Retraining", type="primary"):
            try:
                with st.spinner("Training new model... This may take a few minutes."):
                    results = self.ml_trainer.train_complete_pipeline(min_samples=min_samples)
                
                st.success(f"✅ Model retrained successfully!")
                st.json(results['dataset_info'])
                
                # Update UI to show new model
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"❌ Retraining failed: {str(e)}")
    
    def export_feedback_data(self):
        """Export feedback data for analysis"""
        
        st.markdown("### 📥 Export Feedback Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format",
                options=["CSV", "JSON", "Excel"],
                index=0
            )
        
        with col2:
            date_range = st.selectbox(
                "Date Range",
                options=["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
                index=1
            )
        
        if st.button("📤 Export Data"):
            try:
                # Get date filter
                date_filters = {
                    "Last 7 days": "date('now', '-7 days')",
                    "Last 30 days": "date('now', '-30 days')",
                    "Last 90 days": "date('now', '-90 days')",
                    "All time": "date('2020-01-01')"
                }
                
                date_filter = date_filters[date_range]
                
                conn = sqlite3.connect(self.dataset_manager.db_path)
                
                # Export feedback data
                feedback_df = pd.read_sql_query(f"""
                SELECT 
                    uf.*,
                    cs.compliance_score,
                    cs.created_at as sample_created_at
                FROM user_feedback uf
                JOIN compliance_samples cs ON uf.sample_id = cs.id
                WHERE uf.created_at >= {date_filter}
                ORDER BY uf.created_at DESC
                """, conn)
                
                conn.close()
                
                if not feedback_df.empty:
                    if export_format == "CSV":
                        csv_data = feedback_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"feedback_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    
                    elif export_format == "JSON":
                        json_data = feedback_df.to_json(orient='records', indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_data,
                            file_name=f"feedback_data_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json"
                        )
                    
                    st.success(f"✅ Exported {len(feedback_df)} feedback records")
                    st.dataframe(feedback_df.head(), use_container_width=True)
                
                else:
                    st.warning("No feedback data found for the selected date range.")
            
            except Exception as e:
                st.error(f"Export failed: {str(e)}")

def render_feedback_management_page(dataset_manager: ComplianceDatasetManager, 
                                  ml_trainer: ComplianceMLTrainer):
    """Render comprehensive feedback management page"""
    
    feedback_loop = FeedbackLearningLoop(dataset_manager, ml_trainer)
    
    st.markdown("# 🔄 Feedback & Learning Management")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analytics", "🎯 Performance", "📈 Data Quality", 
        "🔄 Retraining", "📤 Export"
    ])
    
    with tab1:
        feedback_loop.render_feedback_analytics()
    
    with tab2:
        feedback_loop.render_model_performance_dashboard()
    
    with tab3:
        feedback_loop.render_data_quality_dashboard()
    
    with tab4:
        feedback_loop.trigger_manual_retrain()
    
    with tab5:
        feedback_loop.export_feedback_data()
