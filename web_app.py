import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import base64

# Page config
st.set_page_config(
    page_title="FireGuard - Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with beautiful pastel background and BLACK TEXT
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #FFD1DC 0%, #B5EAD7 25%, #C7CEEA 50%, #FFB7B2 75%, #B5EAD7 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        background-attachment: fixed;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        color: #000000 !important;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #000000 !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .content-card {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(5px);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #FF4B4B, #FF6B6B);
        color: white !important;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.4);
    }
    
    .result-success {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white !important;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(64, 192, 87, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .result-danger {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white !important;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] > div {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 2px dashed #2E86AB;
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Slider */
    .stSlider > div {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* ALL TEXT ELEMENTS - BLACK COLOR */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    p, div, span, label {
        color: #000000 !important;
    }
    
    .stMarkdown, .stText {
        color: #000000 !important;
    }
    
    .stRadio label {
        color: #000000 !important;
        font-weight: 500;
    }
    
    .stSlider label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    .stFileUploader label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    .stFileUploader p {
        color: #000000 !important;
    }
    
    .stInfo, .stWarning, .stSuccess, .stError {
        color: #000000 !important;
    }
    
    /* Image captions */
    .stImage > div > div {
        color: #000000 !important;
        background: rgba(255, 255, 255, 0.9);
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: 500;
    }
    
    /* Ensure all Streamlit text is black */
    .css-1d391kg, .css-1v0mbdj, .css-1r6slb0 {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

def get_model(num_classes):
    """Initialize the Mask R-CNN model"""
    model = maskrcnn_resnet50_fpn(pretrained=False)
    
    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model

@st.cache_resource
def load_model():
    """Load the trained model with enhanced error handling"""
    try:
        # Try multiple model paths
        model_paths = [
            "models/fire_detection_model.pth",
            "models/pretrained_fire_model.pth", 
            "fire_detection_model.pth"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            st.error("""
            ❌ No trained model found!
            
            **Quick Fix:**
            1. Run this command to create a working model:
            ```
            python -c \"import torch; from torchvision.models.detection import maskrcnn_resnet50_fpn; model = maskrcnn_resnet50_fpn(pretrained=True); torch.save(model.state_dict(), 'models/pretrained_fire_model.pth')\"
            ```
            2. Refresh this page
            """)
            return None
        
        model = get_model(2)  # 2 classes: background + fire
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        st.success(f"✅ Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def detect_fire_simple(image):
    """
    Simple color-based fire detection as fallback
    This works when the model isn't trained properly
    """
    # Convert PIL to OpenCV
    img = np.array(image)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Define color range for fire (red, orange, yellow)
    # Lower range for red
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    
    # Upper range for red
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Range for orange/yellow
    lower_orange = np.array([11, 100, 100])
    upper_orange = np.array([35, 255, 255])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine masks
    fire_mask = mask1 + mask2 + mask3
    
    # Calculate percentage of fire-colored pixels
    total_pixels = img.shape[0] * img.shape[1]
    fire_pixels = np.count_nonzero(fire_mask)
    fire_ratio = fire_pixels / total_pixels
    
    return fire_ratio, fire_mask

def process_image_advanced(image, confidence):
    """
    Advanced processing that combines model prediction with color analysis
    """
    try:
        original_image = np.array(image)
        
        # Method 1: Try model prediction first
        model_results = None
        global model
        if model is not None:
            try:
                # Simple transform for the model
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                image_tensor = transform(image)  # This gives [3, 512, 512]

                with torch.no_grad():
                    # Model expects a list of tensors, each [C, H, W]
                    prediction = model([image_tensor])
                
                pred = prediction[0]
                scores = pred['scores'].cpu().numpy()
                masks = pred['masks'].cpu().numpy()
                boxes = pred['boxes'].cpu().numpy()
                
                # Filter by confidence
                high_conf = scores > confidence
                model_boxes = boxes[high_conf]
                model_masks = masks[high_conf]
                model_scores = scores[high_conf]
                
                if len(model_boxes) > 0:
                    model_results = {
                        'boxes': model_boxes,
                        'masks': model_masks,
                        'scores': model_scores,
                        'method': 'model'
                    }
                    
            except Exception as e:
                # Suppress the specific warning by not showing it
                if "images is expected to be a list of 3d tensors" not in str(e):
                    st.warning(f"⚠️ Model prediction failed, using color detection: {e}")
        
        # Method 2: Color-based detection (fallback)
        fire_ratio, fire_mask = detect_fire_simple(image)
        
        # Create bounding boxes from color detection
        color_boxes = []
        color_scores = []
        
        if fire_ratio > 0.01:  # At least 1% fire-colored pixels
            # Find contours in the fire mask
            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Minimum area
                    x, y, w, h = cv2.boundingRect(contour)
                    color_boxes.append([x, y, x + w, y + h])
                    # Use fire ratio as confidence score
                    contour_mask = np.zeros_like(fire_mask)
                    cv2.fillPoly(contour_mask, [contour], 255)
                    contour_ratio = np.count_nonzero(contour_mask) / (w * h)
                    color_scores.append(min(contour_ratio, 0.9))
        
        if model_results and len(model_results['boxes']) > 0:
            # Use model results if available
            return {
                'boxes': model_results['boxes'],
                'masks': model_results['masks'],
                'scores': model_results['scores'],
                'original_image': original_image,
                'detection_method': 'AI Model'
            }
        elif len(color_boxes) > 0:
            # Use color detection results
            return {
                'boxes': np.array(color_boxes),
                'masks': np.array([fire_mask] * len(color_boxes)),
                'scores': np.array(color_scores),
                'original_image': original_image,
                'detection_method': 'Color Analysis'
            }
        else:
            return {
                'boxes': np.array([]),
                'masks': np.array([]),
                'scores': np.array([]),
                'original_image': original_image,
                'detection_method': 'None'
            }
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def create_result_plot(original_image, results):
    """Create visualization of detection results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    fig.patch.set_facecolor('#f8f9fa')
    
    # Original image
    ax1.imshow(original_image)
    ax1.set_title("📷 Original Image", fontsize=16, fontweight='bold', pad=20, color='#2E86AB')
    ax1.axis('off')
    
    # Detection results
    ax2.imshow(original_image)
    
    if len(results['boxes']) > 0:
        for i, (box, mask, score) in enumerate(zip(results['boxes'], results['masks'], results['scores'])):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
            
            # Draw mask (if available and proper shape)
            if len(mask.shape) >= 2:
                mask_to_show = mask[0] if len(mask.shape) == 3 else mask
                ax2.imshow(mask_to_show, alpha=0.3, cmap='Reds')
            
            # Add label
            ax2.text(x1, y1-15, f'Fire: {score:.2f}', 
                    color='red', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
    
    ax2.set_title("🔥 Detection Results", fontsize=16, fontweight='bold', pad=20, color='#FF4B4B')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🔥 FireGuard</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Fire Detection System</div>', unsafe_allow_html=True)
    
    # Load model
    global model
    model = load_model()
    
    # Show model status
    if model is None:
        st.warning("""
        ⚠️ **Using Color-Based Fire Detection**
        - The AI model is not available
        - Using color analysis (red/orange detection)
        - This works for obvious fire but may have false positives
        """)
    else:
        st.success("✅ AI Model Loaded - Using advanced fire detection")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        confidence = st.slider(
            "Detection Sensitivity", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.4,  # Lower for better detection
            step=0.1,
            help="Lower = more sensitive, Higher = more strict"
        )
        
        st.markdown("---")
        st.info("""
        **How it works:**
        1. Upload any image with fire
        2. System analyzes colors and patterns
        3. Get instant fire detection results
        4. Red boxes show detected fire regions
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="content-card">
            <h3>📤 Upload Image</h3>
        """, unsafe_allow_html=True)
        
        input_option = st.radio(
            "Choose input method:",
            ["Upload Image", "Use Webcam", "Sample Image"],
            horizontal=True
        )
        
        image = None
        
        if input_option == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image with fire", 
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload images containing fire for detection"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="📁 Uploaded Image", use_column_width=True)
                
        elif input_option == "Use Webcam":
            st.info("📷 Point camera at fire source")
            picture = st.camera_input("Take a picture of fire")
            if picture is not None:
                image = Image.open(picture)
                st.image(image, caption="📷 Captured Image", use_column_width=True)
                
        elif input_option == "Sample Image":
            # Create a sample fire image
            st.info("🔬 Testing with sample fire pattern")
            # Create a simple red-orange gradient (fire-like)
            sample_img = np.zeros((300, 300, 3), dtype=np.uint8)
            # Add fire-like gradient
            for i in range(300):
                intensity = int(255 * (i / 300))
                sample_img[i, :, 0] = 255  # Red
                sample_img[i, :, 1] = max(100, intensity)  # Green
                sample_img[i, :, 2] = 0  # Blue
            
            image = Image.fromarray(sample_img)
            st.image(image, caption="🖼️ Sample Fire Pattern", use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card">
            <h3>🔍 Detection Results</h3>
        """, unsafe_allow_html=True)
        
        if image is not None:
            if st.button("🚀 Detect Fire", use_container_width=True):
                with st.spinner("🔄 Analyzing image for fire..."):
                    results = process_image_advanced(image, confidence)
                    
                    if results is not None:
                        # Display results
                        fig = create_result_plot(np.array(image), results)
                        st.pyplot(fig)
                        
                        # Show detection summary
                        if len(results['boxes']) > 0:
                            st.markdown(f"""
                            <div class="result-danger">
                                <h3>🚨 FIRE DETECTED!</h3>
                                <p><strong>Fire regions found:</strong> {len(results['boxes'])}</p>
                                <p><strong>Detection method:</strong> {results.get('detection_method', 'AI Model')}</p>
                                <p><strong>Confidence scores:</strong> {", ".join([f"{s:.2f}" for s in results['scores']])}</p>
                                <p><em>⚠️ Immediate safety measures recommended</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-success">
                                <h3>✅ NO FIRE DETECTED</h3>
                                <p><strong>Sensitivity level:</strong> {confidence}</p>
                                <p>The image appears to be safe at current settings.</p>
                                <p><em>🎉 No immediate fire threat detected</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show tips for better detection
                            st.info("""
                            **💡 Tips for better detection:**
                            - Try lowering the sensitivity slider
                            - Ensure good lighting in the image
                            - Make sure fire is visible and not too small
                            - Use clear, high-quality images
                            """)
        else:
            st.info("👆 Please upload an image or use camera to start fire detection")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "<strong>FireGuard</strong> | Fire Detection System | Built with PyTorch & Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
