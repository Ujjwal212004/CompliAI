# 📱 CompliAI Mobile Access Setup Guide

## Quick Start (Windows)

### Option 1: Using Startup Scripts
1. **Double-click `start_mobile.bat`** or **right-click `start_mobile.ps1` → Run with PowerShell**
2. The script will:
   - Find your local IP address
   - Display the mobile URL
   - Start CompliAI with network access
3. **Open the displayed URL on your phone** (e.g., `http://192.168.1.5:8501`)

### Option 2: Manual Setup
1. **Find Your IP Address:**
   ```cmd
   ipconfig
   ```
   Look for "IPv4 Address" under your Wi-Fi adapter

2. **Start CompliAI with Network Access:**
   ```cmd
   streamlit run app.py --server.address=0.0.0.0
   ```

3. **Access from Mobile:**
   - Connect phone to same Wi-Fi network
   - Open browser on phone
   - Go to: `http://YOUR_IP:8501`

## Mobile Camera Access (HTTPS)

For mobile camera functionality, you need HTTPS. Use ngrok:

1. **Install ngrok:**
   - Download from [ngrok.com](https://ngrok.com/)
   - Extract and add to PATH

2. **Start CompliAI normally:**
   ```cmd
   streamlit run app.py
   ```

3. **In another terminal, run ngrok:**
   ```cmd
   ngrok http 8501
   ```

4. **Use the HTTPS URL provided by ngrok** (e.g., `https://abc123.ngrok.io`)

## Using the Mobile Interface

### In-App Setup
1. Go to **Mobile Access** section in the app sidebar
2. Follow the **Quick Setup** tab for automatic configuration
3. Scan the **QR code** with your phone to access directly
4. Use the **Mobile-Optimized Upload** for camera functionality

### Camera Features

#### Gallery Upload (Recommended)
- **Take Photo**: Direct camera access
- **Choose from Gallery**: Select existing photos
- **Auto-optimization**: Image quality assessment
- **Touch-friendly**: Mobile-optimized interface

#### Live Camera (HTTPS Required)
- **Real-time Preview**: See camera feed before capture
- **Focus Control**: Tap to focus on text
- **Multiple Captures**: Take several shots
- **Back Camera**: Optimized for document scanning

**Note**: Live camera requires HTTPS. Use ngrok for mobile access.

#### Upload Methods
1. **Gallery Upload**: Best for most users
   - Works on all mobile browsers
   - Direct camera and gallery access
   - Automatic image optimization

2. **Live Camera**: For advanced users
   - Requires HTTPS connection
   - Real-time camera preview
   - Professional document scanning

3. **File Upload**: Traditional method
   - Standard file browser
   - Works on all devices
   - No camera permissions needed

### Mobile Tips
- **Best Browsers:** Chrome, Firefox, Safari
- **Camera Access:** Requires HTTPS (use ngrok)
- **Photo Tips:** Good lighting, steady hands, clear text visibility
- **Performance:** Use Wi-Fi, close other apps, stable connection

## Troubleshooting

### Cannot Access from Phone
- ✅ Both devices on same Wi-Fi?
- ✅ IP address correct?
- ✅ Firewall blocking port 8501?
- ✅ Try restarting router

### Camera Not Working
- ✅ Using HTTPS (ngrok)?
- ✅ Browser permissions enabled?
- ✅ Try different browser
- ✅ Clear browser cache

### Slow Performance
- ✅ Strong Wi-Fi signal?
- ✅ Close other phone apps
- ✅ Use smaller image files
- ✅ Check laptop performance

### Connection Drops
- ✅ Move closer to router
- ✅ Restart CompliAI app
- ✅ Check power saving settings
- ✅ Use ethernet on laptop

## Network Configuration

### Windows Firewall
If connection fails, allow through Windows Firewall:
1. Windows Security → Firewall & network protection
2. Allow an app through firewall
3. Add Python/Streamlit if needed

### Router Settings
- Ensure both devices get IP addresses
- Check AP isolation is disabled
- Verify port 8501 is not blocked

### Alternative Ports
If port 8501 is busy:
```cmd
streamlit run app.py --server.address=0.0.0.0 --server.port=8502
```

## Advanced Features

### QR Code Access
- Automatically generated in Mobile Access section
- Instant phone access without typing URLs
- Updated in real-time

### Mobile-Optimized Interface
- Touch-friendly controls
- Responsive design
- Camera integration
- Offline-capable analysis

### Security Options
- Basic authentication available
- HTTPS via ngrok
- Local network only by default
- No data leaves your device

---

**Ready to revolutionize mobile compliance checking!** 🚀

For more help, check the **Tutorials** tab in the Mobile Access section of the app.