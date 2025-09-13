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
import socket
import subprocess
import platform
import qrcode
from io import BytesIO
import base64



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
    /* Mobile-friendly styles */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem 0.5rem;
        }
        .site-title {
            font-size: 2rem;
            letter-spacing: 1px;
        }
        .site-subtitle {
            font-size: 1rem;
        }
        .metric-card {
            margin-bottom: 1rem;
        }
    }
    .mobile-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .qr-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .access-info {
        background: #e8f4fd;
        border: 1px solid #b3d7ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .camera-capture {
        width: 100%;
        max-width: 400px;
        margin: 1rem auto;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # Connect to a dummy address to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        # Fallback method
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
            return local_ip
        except Exception:
            return "127.0.0.1"

def get_network_interfaces():
    """Get all network interfaces and their IP addresses"""
    interfaces = []
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["ipconfig"], capture_output=True, text=True, shell=True)
            output = result.stdout
            lines = output.split('\n')
            current_adapter = ""
            for line in lines:
                line = line.strip()
                if "adapter" in line.lower() and ":" in line:
                    current_adapter = line.split(':')[0].strip()
                elif "IPv4 Address" in line and "." in line:
                    ip = line.split(':')[1].strip()
                    if not ip.startswith("127.") and current_adapter:
                        interfaces.append({"adapter": current_adapter, "ip": ip})
        else:
            # For Linux/macOS
            result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
            if result.returncode == 0:
                ips = result.stdout.strip().split()
                for ip in ips:
                    if not ip.startswith("127."):
                        interfaces.append({"adapter": "Network", "ip": ip})
    except Exception as e:
        st.error(f"Error getting network interfaces: {e}")
    
    return interfaces

def generate_qr_code(url):
    """Generate QR code for the given URL"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert PIL image to base64 string
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def render_mobile_access_info():
    """Render mobile access information and QR codes"""
    st.markdown("### 📱 Mobile Access")
    
    # Get local IP and port
    local_ip = get_local_ip()
    port = 8501  # Default Streamlit port
    
    # Check if we're running on all interfaces
    is_accessible = True
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌐 Network Access")
        
        # Show current access URL
        local_url = f"http://{local_ip}:{port}"
        st.markdown(f"**Local URL:** `{local_url}`")
        
        # Show network interfaces
        interfaces = get_network_interfaces()
        if interfaces:
            st.markdown("**Available Network Interfaces:**")
            for interface in interfaces:
                adapter_name = interface["adapter"]
                ip = interface["ip"]
                url = f"http://{ip}:{port}"
                st.markdown(f"• {adapter_name}: `{url}`")
        
        # Instructions for mobile access
        st.markdown("""**Mobile Access Steps:**
        1. Ensure phone and laptop are on same Wi-Fi
        2. Open browser on phone
        3. Enter the URL above or scan QR code
        4. Access CompliAI on your mobile device""")
        
        # Server configuration info
        with st.expander("🔧 Server Configuration"):
            st.markdown("""**To enable mobile access, run:**
            ```bash
            streamlit run app.py --server.address=0.0.0.0
            ```
            
            This binds Streamlit to all network interfaces, allowing mobile access.
            
            **Alternative with custom port:**
            ```bash
            streamlit run app.py --server.address=0.0.0.0 --server.port=8502
            ```""")
    
    with col2:
        st.markdown("#### 📲 QR Code Access")
        
        # Generate QR code for the main URL
        if local_ip and local_ip != "127.0.0.1":
            qr_img = generate_qr_code(local_url)
            st.markdown(
                f'<div class="qr-container"><img src="data:image/png;base64,{qr_img}" width="200" alt="QR Code"/><p><strong>Scan to access CompliAI</strong></p></div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("Unable to determine local IP. Please check network connection.")
        
        # Mobile browser compatibility note
        st.info("""**📝 Mobile Browser Note:**
        
        Some mobile browsers may block camera access on non-HTTPS pages. If camera doesn't work:
        
        1. Try different browsers (Chrome, Firefox, Safari)
        2. Use ngrok for HTTPS access (see setup below)
        3. Upload images from gallery instead""")
    
    # Ngrok setup section
    with st.expander("🔒 HTTPS Access with ngrok (Recommended for Camera)"):
        st.markdown("""**For HTTPS access (needed for mobile camera):**
        
        1. **Install ngrok:**
           - Download from [ngrok.com](https://ngrok.com/)
           - Extract and add to PATH
        
        2. **Run your app normally:**
           ```bash
           streamlit run app.py
           ```
        
        3. **In another terminal, run ngrok:**
           ```bash
           ngrok http 8501
           ```
        
        4. **Use the HTTPS URL provided by ngrok**
           - Example: `https://abc123.ngrok.io`
           - This URL works from anywhere with internet
        
        **Benefits:**
        - HTTPS enables mobile camera access
        - Works from anywhere (not just local network)
        - Secure tunnel to your local app
        """)

def detect_mobile():
    """Detect if the user is on a mobile device"""
    try:
        # This is a simple heuristic based on user agent
        # In Streamlit, we can't directly access user agent, so we use viewport width
        return False  # Placeholder - would need JavaScript for proper detection
    except:
        return False

class MobileCameraCapture:
    """Mobile camera capture component with live streaming"""
    
    def __init__(self):
        self.captured_image = None
        self.capture_lock = threading.Lock()
    
    def capture_frame(self, frame):
        """Capture frame from camera stream"""
        with self.capture_lock:
            img = frame.to_ndarray(format="bgr24")
            # Convert BGR to RGB
            img_rgb = img[:, :, ::-1]
            self.captured_image = Image.fromarray(img_rgb)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def render_live_camera_capture():
    """Render live camera capture interface using HTML5 Camera API"""
    st.markdown("#### 📷 Live Camera Capture")
    
    # HTML5 Camera Interface
    camera_html = """
    <div id="camera-container" style="text-align: center; margin: 20px 0;">
        <video id="camera-feed" width="100%" height="300" style="border: 2px solid #667eea; border-radius: 10px; display: none;"></video>
        <canvas id="capture-canvas" width="640" height="480" style="display: none;"></canvas>
        <img id="captured-image" style="max-width: 100%; border: 2px solid #667eea; border-radius: 10px; display: none;">
        
        <div style="margin: 20px 0;">
            <button id="start-camera" onclick="startCamera()" style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; border: none; padding: 12px 24px; border-radius: 8px;
                font-size: 16px; margin: 5px; cursor: pointer; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            ">📹 Start Camera</button>
            
            <button id="capture-photo" onclick="capturePhoto()" style="
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white; border: none; padding: 12px 24px; border-radius: 8px;
                font-size: 16px; margin: 5px; cursor: pointer; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                display: none;
            ">📸 Capture Photo</button>
            
            <button id="retake-photo" onclick="retakePhoto()" style="
                background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
                color: white; border: none; padding: 12px 24px; border-radius: 8px;
                font-size: 16px; margin: 5px; cursor: pointer; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                display: none;
            ">🔄 Retake</button>
            
            <button id="use-photo" onclick="usePhoto()" style="
                background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
                color: white; border: none; padding: 12px 24px; border-radius: 8px;
                font-size: 16px; margin: 5px; cursor: pointer; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                display: none;
            ">✔️ Use Photo</button>
        </div>
        
        <div id="camera-status" style="margin: 10px 0; padding: 10px; border-radius: 5px; display: none;"></div>
    </div>
    
    <script>
    let currentStream = null;
    let capturedImageData = null;
    
    async function startCamera() {
        try {
            // Request camera permission with back camera preference
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: { ideal: 'environment' } // Back camera for documents
                }
            };
            
            currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            const video = document.getElementById('camera-feed');
            const startBtn = document.getElementById('start-camera');
            const captureBtn = document.getElementById('capture-photo');
            const status = document.getElementById('camera-status');
            
            video.srcObject = currentStream;
            video.play();
            
            // Show/hide elements
            video.style.display = 'block';
            startBtn.style.display = 'none';
            captureBtn.style.display = 'inline-block';
            
            // Show success status
            status.innerHTML = '✅ Camera active! Position your product and tap Capture Photo.';
            status.style.display = 'block';
            status.style.backgroundColor = '#d4edda';
            status.style.color = '#155724';
            
        } catch (error) {
            console.error('Camera error:', error);
            const status = document.getElementById('camera-status');
            
            let errorMessage = '❌ Camera access failed. ';
            
            if (error.name === 'NotAllowedError') {
                errorMessage += 'Please allow camera permissions and try again.';
            } else if (error.name === 'NotFoundError') {
                errorMessage += 'No camera found on this device.';
            } else if (error.name === 'NotSupportedError') {
                errorMessage += 'Camera not supported in this browser.';
            } else {
                errorMessage += 'Error: ' + error.message;
            }
            
            status.innerHTML = errorMessage + '<br><small>Try using Gallery Upload instead.</small>';
            status.style.display = 'block';
            status.style.backgroundColor = '#f8d7da';
            status.style.color = '#721c24';
        }
    }
    
    function capturePhoto() {
        const video = document.getElementById('camera-feed');
        const canvas = document.getElementById('capture-canvas');
        const capturedImg = document.getElementById('captured-image');
        const captureBtn = document.getElementById('capture-photo');
        const retakeBtn = document.getElementById('retake-photo');
        const useBtn = document.getElementById('use-photo');
        const status = document.getElementById('camera-status');
        
        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Capture frame
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        // Convert to image data
        capturedImageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Show captured image
        capturedImg.src = capturedImageData;
        capturedImg.style.display = 'block';
        video.style.display = 'none';
        
        // Update buttons
        captureBtn.style.display = 'none';
        retakeBtn.style.display = 'inline-block';
        useBtn.style.display = 'inline-block';
        
        // Update status
        status.innerHTML = '📸 Photo captured! Review and tap "Use Photo" to analyze or "Retake" for a new photo.';
        status.style.backgroundColor = '#fff3cd';
        status.style.color = '#856404';
    }
    
    function retakePhoto() {
        const video = document.getElementById('camera-feed');
        const capturedImg = document.getElementById('captured-image');
        const captureBtn = document.getElementById('capture-photo');
        const retakeBtn = document.getElementById('retake-photo');
        const useBtn = document.getElementById('use-photo');
        const status = document.getElementById('camera-status');
        
        // Show video, hide image
        video.style.display = 'block';
        capturedImg.style.display = 'none';
        
        // Update buttons
        captureBtn.style.display = 'inline-block';
        retakeBtn.style.display = 'none';
        useBtn.style.display = 'none';
        
        // Reset status
        status.innerHTML = '✅ Camera active! Position your product and tap Capture Photo.';
        status.style.backgroundColor = '#d4edda';
        status.style.color = '#155724';
        
        capturedImageData = null;
    }
    
    function usePhoto() {
        if (capturedImageData) {
            // Convert base64 to blob
            const byteCharacters = atob(capturedImageData.split(',')[1]);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], {type: 'image/jpeg'});
            
            // Create file object
            const file = new File([blob], 'camera-capture.jpg', {type: 'image/jpeg'});
            
            // This is where we'd integrate with Streamlit's file uploader
            // For now, show success message
            const status = document.getElementById('camera-status');
            status.innerHTML = '✅ Photo ready! Please use the Gallery Upload option above to select this captured image.';
            status.style.backgroundColor = '#d1ecf1';
            status.style.color = '#0c5460';
            
            // Stop camera
            if (currentStream) {
                currentStream.getTracks().forEach(track => track.stop());
            }
        }
    }
    
    // Stop camera when leaving page
    window.addEventListener('beforeunload', function() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
    });
    </script>
    """
    
    st.markdown(camera_html, unsafe_allow_html=True)
    
    # Camera tips for mobile
    with st.expander("📱 Mobile Camera Tips", expanded=False):
        st.markdown("""
        **📷 For Best Results:**
        
        • **Allow camera permissions** when browser asks
        • **Use back camera** (automatically selected)
        • **Hold phone steady** with both hands
        • **Ensure good lighting** (natural light preferred)
        • **Position product clearly** in frame
        • **Tap to focus** on text if supported
        
        **📦 Product Positioning:**
        
        • **Fill 70-80% of frame** with product
        • **Keep packaging parallel** to screen
        • **Include all text** that needs analysis
        • **Avoid shadows and glare**
        • **Use plain background** if possible
        """)

def render_mobile_camera_upload():
    """Render enhanced mobile camera upload interface with live capture"""
    st.markdown("### 📷 Mobile Image Capture")
    
    # Camera capture method selector
    capture_method = st.radio(
        "Choose capture method:",
        ["Gallery Upload", "Live Camera (HTTPS Required)", "File Upload"],
        horizontal=True,
        help="Gallery Upload works on all devices. Live Camera needs HTTPS (use ngrok)."
    )
    
    uploaded_file = None
    
    if capture_method == "Gallery Upload":
        st.markdown("#### 🖼️ Upload from Gallery")
        
        # Mobile-optimized file uploader with camera access
        uploaded_file = st.file_uploader(
            "📱 Take Photo or Choose from Gallery",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="On mobile: Tap to choose 'Take Photo' or 'Photo Library'",
            accept_multiple_files=False,
            key="mobile_gallery_upload"
        )
        
        # Add HTML5 camera input for direct camera access
        st.markdown("""
        <div style="margin: 1rem 0;">
            <label for="camera-input" style="
                display: inline-block;
                padding: 0.5rem 1rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 5px;
                cursor: pointer;
                text-align: center;
                font-weight: bold;
            ">
                📷 Quick Camera Capture
            </label>
            <input 
                type="file" 
                id="camera-input" 
                accept="image/*" 
                capture="environment"
                style="display: none;"
                onchange="handleCameraCapture(this)"
            >
        </div>
        
        <script>
        function handleCameraCapture(input) {
            if (input.files && input.files[0]) {
                // This would integrate with Streamlit's file uploader
                // For now, show user to use the file uploader above
                alert('📷 Photo captured! Please use the file uploader above to select your captured image.');
            }
        }
        </script>
        """, unsafe_allow_html=True)
        
    elif capture_method == "Live Camera (HTTPS Required)":
        st.markdown("#### 🔴 Live Camera Capture")
        
        # Add the enhanced HTML5 camera interface
        render_live_camera_capture()
        
        st.info("ℹ️ **Live camera is now available!** This deployed version has HTTPS enabled, so the camera should work directly.")
    
    else:  # File Upload
        st.markdown("#### 📁 Standard File Upload")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Select image file from your device",
            key="standard_file_upload"
        )
    
    # Display uploaded image
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            
            # Mobile-optimized image display
            st.markdown("**🖼️ Captured Image:**")
            
            # Responsive image display
            col1, col2, col3 = st.columns([0.5, 2, 0.5])
            with col2:
                st.image(image, caption=f"Product Image ({uploaded_file.name})", use_column_width=True)
            
            # Image info
            with st.expander("📊 Image Information"):
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {uploaded_file.size:,} bytes")
                st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")
                st.write(f"**Format:** {image.format}")
                
                # Image quality assessment
                width, height = image.size
                total_pixels = width * height
                
                if total_pixels > 2000000:  # > 2MP
                    st.success("✅ High resolution - excellent for analysis")
                elif total_pixels > 1000000:  # > 1MP
                    st.info("ℹ️ Good resolution - suitable for analysis")
                else:
                    st.warning("⚠️ Low resolution - may affect accuracy")
            
            # Mobile-friendly analysis button
            st.markdown("---")
            col1, col2, col3 = st.columns([0.5, 2, 0.5])
            with col2:
                if st.button("🔍 Analyze Compliance Now", type="primary", use_container_width=True):
                    return uploaded_file
            
            # Quick preview options
            with st.expander("🔍 Quick Actions"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Rotate 90°", use_container_width=True):
                        # Rotate image (this would need additional logic)
                        st.info("Rotation feature coming soon!")
                with col2:
                    if st.button("🔍 Zoom to Text", use_container_width=True):
                        st.info("Auto-zoom feature coming soon!")
                        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            st.info("Please try uploading a different image file.")
    
    # Enhanced mobile photography tips
    with st.expander("📱 Mobile Photography Guide", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📷 Camera Tips", "📋 Positioning", "⚙️ Settings"])
        
        with tab1:
            st.markdown("""**📷 Camera & Lighting:**
            
            ✅ **DO:**
            - Use natural daylight when possible
            - Tap screen to focus on text
            - Hold phone steady (use both hands)
            - Clean camera lens before shooting
            - Take multiple shots from different angles
            
            ❌ **DON'T:**
            - Use flash (creates glare and shadows)
            - Shoot in very dim lighting
            - Rush - take time to compose shot
            - Ignore camera shake warnings
            """)
        
        with tab2:
            st.markdown("""**📋 Product Positioning:**
            
            ✅ **Optimal Setup:**
            - Place product on flat, clean surface
            - Ensure all text is visible in frame
            - Fill 70-80% of frame with product
            - Keep packaging parallel to camera
            - Avoid reflective surfaces underneath
            
            💯 **Pro Tips:**
            - Use white paper as background
            - Slightly angle product to avoid glare
            - Capture front and back if info is split
            - Include product edges in frame
            """)
        
        with tab3:
            st.markdown("""**⚙️ Mobile Settings:**
            
            **📱 Phone Settings:**
            - Enable HDR for better detail
            - Use highest resolution available
            - Turn off digital zoom (move closer instead)
            - Enable gridlines to help composition
            
            **🌐 App Settings:**
            - Connect to Wi-Fi for faster upload
            - Close background apps for better performance
            - Enable location services if prompted
            - Allow camera permissions in browser
            
            **🔋 Performance:**
            - Wait for image to fully load before analyzing
            - Don't switch apps during upload
            - Keep phone charged (analysis uses battery)
            """)
    
    return None

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

def detect_mobile_device():
    """Detect if user is on mobile device using JavaScript"""
    mobile_script = """
    <script>
    function detectMobile() {
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        const screenWidth = window.innerWidth;
        return isMobile || isTouch || screenWidth < 768;
    }
    
    if (detectMobile()) {
        document.body.classList.add('mobile-device');
        // Send mobile status to Streamlit (this would need additional integration)
    }
    </script>
    """
    st.markdown(mobile_script, unsafe_allow_html=True)
    
    # Simple fallback detection based on viewport
    return False  # Would need JavaScript integration for real detection

def render_file_upload():
    st.markdown("### 📤 Upload Product Image")
    
    # Mobile-friendly interface toggle
    interface_mode = st.radio(
        "Choose interface:",
        ["Auto-Detect", "Desktop Mode", "Mobile Mode"],
        index=0,
        horizontal=True,
        help="Auto-detect works best, but you can manually select interface type"
    )
    
    # Analysis method selector
<<<<<<< HEAD
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
                'cascading': '✅ Smart Analysis (Recommended)',
                'gemini_only': '🔧 Direct Gemini Only (Debug)'
            }[x],
            index=0,
            help="Smart Analysis uses our enhanced field extraction with real Gemini API integration. Choose this for accurate results from your uploaded images."
        )
        st.session_state.analysis_method = analysis_method
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1, 2, 1])
=======
    with st.container():
        col1, col2 = st.columns([3, 2] if interface_mode != "Mobile Mode" else [1, 1])
        
        with col1:
            st.markdown("**Choose Upload Method:**")
            
            if interface_mode == "Mobile Mode":
                # Mobile-optimized upload
                upload_method = st.selectbox(
                    "Upload Method",
                    ["Camera/Gallery", "File Browser"],
                    help="Camera/Gallery optimized for mobile devices"
                )
                
                if upload_method == "Camera/Gallery":
                    uploaded_file = st.file_uploader(
                        "📱 Take Photo or Choose from Gallery",
                        type=['png', 'jpg', 'jpeg', 'webp'],
                        help="On mobile: Choose 'Take Photo' or 'Photo Library'",
                        key="mobile_upload"
                    )
                else:
                    uploaded_file = st.file_uploader(
                        "Choose image file",
                        type=['png', 'jpg', 'jpeg', 'webp'],
                        help="Select image from file system",
                        key="file_browser_upload"
                    )
            else:
                # Desktop/Auto interface
                uploaded_file = st.file_uploader(
                    "Choose a product packaging image",
                    type=['png', 'jpg', 'jpeg', 'webp'],
                    help="Upload a clear image of the product packaging with visible text",
                    key="desktop_upload"
                )
                
                # Add mobile camera option for desktop too
                with st.expander("📱 Mobile Camera Options"):
                    st.info("📷 For mobile camera access, switch to 'Mobile Mode' above or visit the Mobile Access section.")
        
>>>>>>> 2edad0b (Prepare for cloud deployment: Clean app.py for mobile HTTPS support)
        with col2:
            st.markdown("**Analysis Method:**")
            analysis_method = st.radio(
                "Choose analysis approach",
                options=['cascading', 'gemini_only'],
                format_func=lambda x: {
                    'cascading': '🔄 Sequential Analysis' if interface_mode == "Mobile Mode" else '🔄 Sequential Analysis (Rule-based → ML → Gemini)',
                    'gemini_only': '🤖 Gemini Only' if interface_mode == "Mobile Mode" else '🤖 Gemini API Only (Original)'
                }[x],
                index=0,
                help="Sequential analysis is recommended for best accuracy and speed" if interface_mode == "Mobile Mode" else "Sequential analysis tries rule-based first (fast), then ML model if rule-based fails, finally Gemini API if both fail. Uses the first successful result for efficiency."
            )
            st.session_state.analysis_method = analysis_method
    
    # Display uploaded image and analysis button
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Mobile-optimized or desktop display
            if interface_mode == "Mobile Mode":
                # Mobile: Full width with image info
                st.markdown(f"**🖼️ Image Preview: {uploaded_file.name}**")
                st.image(image, caption="Product Image", use_column_width=True)
                
                # Mobile-specific image info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Width", f"{image.size[0]}px")
                with col2:
                    st.metric("Height", f"{image.size[1]}px")
                with col3:
                    st.metric("Size", f"{uploaded_file.size//1024}KB")
                
                # Mobile: Full-width button
                if st.button("🔍 Analyze Compliance", type="primary", use_container_width=True):
                    st.session_state.uploaded_image = uploaded_file
                    analyze_image(uploaded_file)
            else:
                # Desktop: Centered display
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image, caption="Uploaded Product Image", use_column_width=True)
                
                # Desktop: Centered button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔍 Analyze Compliance", type="primary", use_container_width=True):
                        st.session_state.uploaded_image = uploaded_file
                        analyze_image(uploaded_file)
        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            st.info("Please try uploading a different image file.")
    
    # Additional mobile features
    if interface_mode == "Mobile Mode":
        st.markdown("---")
        with st.expander("📱 Mobile Features", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🚀 Quick Actions:**")
                st.markdown("- Take photo directly")
                st.markdown("- Choose from gallery")
                st.markdown("- Auto-rotate detection")
                st.markdown("- Touch-friendly interface")
            
            with col2:
                st.markdown("**📈 Optimizations:**")
                st.markdown("- Compressed image upload")
                st.markdown("- Faster analysis")
                st.markdown("- Mobile-first design")
                st.markdown("- Offline capability")
    
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
                "Dataset Insights",
                "Mobile Access"
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

def render_mobile_access_page():
    """Render the mobile access configuration and setup page"""
    st.markdown("# 📱 Mobile Access - Cloud Deployed")
    
    # Show cloud deployment status
    st.markdown("### 🌐 Cloud Deployment Status")
    st.success("✅ **HTTPS Enabled** - CompliAI is deployed on Streamlit Cloud with full HTTPS support!")
    st.info("📷 **Mobile Camera Access**: Live camera capture is available on this HTTPS deployment.")
    
    # Mobile features for cloud deployment
    st.markdown("### 📱 Mobile Features Available")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📷 Camera Features:**")
        st.markdown("• Live camera capture")
        st.markdown("• Gallery photo upload")
        st.markdown("• Real-time preview")
        st.markdown("• Touch-friendly interface")
        
    with col2:
        st.markdown("**✨ Mobile Optimizations:**")
        st.markdown("• Responsive design")
        st.markdown("• Mobile-friendly UI")
        st.markdown("• HTTPS secure connection")
        st.markdown("• Cross-platform compatibility")
    
    st.markdown("---")
    st.markdown("### 📷 Test Mobile Camera")
    
    # Mobile camera test interface
    uploaded_file = render_mobile_camera_upload()
    if uploaded_file:
        analyze_image(uploaded_file)
    
    # Create tabs for additional configuration and tutorials
    tab1, tab2, tab3 = st.tabs(["📷 Camera Test", "⚙️ Advanced Config", "📚 Tutorials"])
    
    with tab1:
        st.markdown("### 📱 Mobile Camera Test Completed Above")
        st.info("The mobile camera interface is available above. Upload an image to test the full compliance analysis pipeline.")
    
    with tab2:
        st.markdown("### ⚙️ Advanced Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖥️ Server Configuration")
            
            # Current server status for cloud deployment
            st.info("**Cloud Deployment Settings:**\n" + 
                   "- Host: Streamlit Cloud\n" + 
                   "- Protocol: HTTPS\n" + 
                   "- Mobile Camera: Enabled")
            
            # Custom port option
            custom_port = st.number_input(
                "Custom Port", 
                min_value=1024, 
                max_value=65535, 
                value=8501,
                help="Change default port if 8501 is busy"
            )
            
            if custom_port != 8501:
                st.markdown(f"**Run with custom port:**")
                st.code(f"streamlit run app.py --server.address=0.0.0.0 --server.port={custom_port}")
        
        with col2:
            st.markdown("#### 🔒 Security Options")
            
            enable_auth = st.checkbox(
                "Enable Basic Authentication",
                help="Require password for access"
            )
            
            if enable_auth:
                auth_password = st.text_input(
                    "Access Password",
                    type="password",
                    help="Set password for mobile access"
                )
                st.info("Note: Basic auth requires additional configuration")
            
            st.markdown("**HTTPS Setup:**")
            st.markdown("For secure mobile camera access:")
            st.code("ngrok http 8501")
    
    with tab3:
        st.markdown("### 📚 Step-by-Step Tutorials")
        
        tutorial_option = st.selectbox(
            "Choose Tutorial",
            [
                "Windows Local IP Setup",
                "macOS Local IP Setup", 
                "Linux Local IP Setup",
                "Mobile Browser Setup",
                "Ngrok HTTPS Setup",
                "Troubleshooting Guide"
            ]
        )
        
        if tutorial_option == "Windows Local IP Setup":
            st.markdown("""#### 🪟 Windows Setup
            
            **Step 1: Find Your IP Address**
            1. Open Command Prompt (Win + R, type `cmd`)
            2. Type `ipconfig` and press Enter
            3. Look for "IPv4 Address" under your Wi-Fi adapter
            4. Note the IP (e.g., 192.168.1.5)
            
            **Step 2: Start Streamlit with Network Access**
            ```bash
            streamlit run app.py --server.address=0.0.0.0
            ```
            
            **Step 3: Access from Mobile**
            1. Connect phone to same Wi-Fi network
            2. Open browser on phone
            3. Go to: http://YOUR_IP:8501
            4. Example: http://192.168.1.5:8501
            """)
        
        elif tutorial_option == "Mobile Browser Setup":
            st.markdown("""#### 📱 Mobile Browser Configuration
            
            **Best Mobile Browsers for CompliAI:**
            - ✅ Chrome (recommended)
            - ✅ Firefox
            - ✅ Safari (iOS)
            - ⚠️ Edge (may have camera issues)
            
            **Camera Access:**
            1. Enable location services
            2. Allow camera permissions
            3. Use HTTPS for reliable camera access
            4. Clear browser cache if issues occur
            
            **Upload Tips:**
            - Use "Take Photo" option for best results
            - Ensure good lighting
            - Hold device steady
            - Tap to focus on text
            """)
        
        elif tutorial_option == "Ngrok HTTPS Setup":
            st.markdown("""#### 🔒 Ngrok HTTPS Setup
            
            **Why Use Ngrok?**
            - Enables HTTPS for mobile camera access
            - Works from anywhere (not just local network)
            - Secure tunnel to your app
            
            **Setup Steps:**
            
            1. **Install Ngrok:**
               - Go to [ngrok.com](https://ngrok.com/)
               - Download for your OS
               - Extract to a folder in PATH
            
            2. **Start CompliAI:**
               ```bash
               streamlit run app.py
               ```
            
            3. **Open New Terminal and Run:**
               ```bash
               ngrok http 8501
               ```
            
            4. **Copy HTTPS URL:**
               - Look for line like: https://abc123.ngrok.io
               - Use this URL on mobile device
               - Camera will work with HTTPS!
            
            **Pro Tips:**
            - Free ngrok has 8-hour session limit
            - URL changes each restart
            - Sign up for stable URLs
            """)
        
        elif tutorial_option == "Troubleshooting Guide":
            st.markdown("""#### 🔧 Common Issues & Solutions
            
            **❌ Cannot Access from Phone**
            - Check both devices on same Wi-Fi
            - Verify IP address is correct
            - Try restarting router
            - Disable VPN on either device
            
            **📷 Camera Not Working**
            - Use HTTPS (ngrok recommended)
            - Try different browser
            - Check camera permissions
            - Clear browser cache
            
            **🐌 Slow Performance**
            - Use Wi-Fi instead of cellular
            - Close other apps on phone
            - Reduce image size before upload
            - Check laptop performance
            
            **🔌 Connection Keeps Dropping**
            - Check Wi-Fi signal strength
            - Move closer to router
            - Restart Streamlit app
            - Use ethernet on laptop
            
            **🚨 Emergency Backup Plan**
            - Use laptop camera directly
            - Upload images via email/cloud
            - Use USB cable to transfer files
            - Switch to mobile hotspot
            """)
    
    # Connection test section
    st.markdown("---")
    st.markdown("### 🧪 Connection Test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Scan Network Interfaces", use_container_width=True):
            with st.spinner("Scanning network interfaces..."):
                interfaces = get_network_interfaces()
                if interfaces:
                    st.success(f"Found {len(interfaces)} network interfaces:")
                    for i, interface in enumerate(interfaces, 1):
                        st.write(f"{i}. {interface['adapter']}: {interface['ip']}")
                else:
                    st.warning("No network interfaces found")
    
    with col2:
        if st.button("📋 Copy Connection Info", use_container_width=True):
            connection_info = f"""CompliAI Mobile Access
            
Local IP: {local_ip}
Port: 8501
URL: http://{local_ip}:8501

Setup Command:
streamlit run app.py --server.address=0.0.0.0

For HTTPS (camera access):
ngrok http 8501
            """
            st.text_area("Connection Information", connection_info, height=200)
            st.info("📋 Copy the information above to share or save")

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
    elif active_page == "Mobile Access":
        render_mobile_access_page()
    
    # Render footer on all pages
    render_footer()

if __name__ == "__main__":
    main()
