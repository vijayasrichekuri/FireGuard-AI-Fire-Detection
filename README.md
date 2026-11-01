# 🔥 FireGuard – AI-Powered Fire Detection System

An intelligent computer vision application that detects fire in images using **PyTorch, Mask R-CNN and OpenCV**, with an interactive web interface built using **Streamlit**.

## 📌 Project Overview

Fire incidents can cause significant damage when they are not identified early. FireGuard explores an AI-based approach to identifying visible fire regions in images.

The application analyzes uploaded images, webcam captures, or sample images and displays detected regions with bounding boxes, segmentation masks, and confidence scores.

## ✨ Key Features

- **AI-Based Fire Detection:** Uses a Mask R-CNN architecture with a ResNet-50 FPN backbone.
- **Object Localization:** Identifies potential fire regions using bounding boxes.
- **Instance Segmentation:** Displays predicted fire masks.
- **Confidence Scores:** Shows prediction confidence for detected regions.
- **Image Upload:** Supports JPG, JPEG, PNG, and BMP images.
- **Webcam Input:** Capture images directly through the application.
- **Sample Image Testing:** Includes a sample fire-pattern option.
- **Fallback Detection:** Uses HSV-based color analysis when the AI model is unavailable or does not return detections.
- **Interactive Interface:** Built with Streamlit.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Object Detection | Mask R-CNN |
| Backbone | ResNet-50 FPN |
| Image Processing | NumPy, Pillow |
| Visualization | Matplotlib |
| Web Framework | Streamlit |

## ⚙️ System Architecture

```text
       User Input
           |
    Image / Webcam
           |
    Image Preprocessing
           |
      Fire Detection
           |
     +-----+------+
     |            |
  AI Model    Color Analysis
  Mask R-CNN    Fallback
     |            |
     +-----+------+
           |
   Detection Results
           |
  Bounding Boxes and Masks
           |
   Confidence Scores
           |
    Streamlit Dashboard
```

## 📂 Project Structure

```text
FireDetectionProject/
│
├── models/
│   └── fire_detection_model.pth
│
├── utils/
│
├── web_app.py
├── inference.py
├── user_inference.py
├── train.py
├── sounds.py
├── check_model.py
├── create_sample_data.py
│
├── test_fire_detection.py
├── test_install.py
├── test_new_image.py
├── test_packages.py
│
├── requirements.txt
├── packages.txt
├── .gitignore
└── README.md
```

*Note: Model weights and datasets are not included in this GitHub repository. Obtain the required model checkpoint separately and place it in the `models/` directory.*

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vijayasrichekuri/FireGuard-AI-Fire-Detection.git
```

### 2. Navigate to the project directory

```bash
cd FireGuard-AI-Fire-Detection
```

### 3. Create a virtual environment

```bash
python -m venv fireguard-env
```

Activate it on Windows:

```bash
fireguard-env\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the model checkpoint

Place the compatible model checkpoint at:

```text
models/fire_detection_model.pth
```

The model architecture and checkpoint must be compatible with the application.

### 6. Run the application

```bash
python -m streamlit run web_app.py
```

Open the local URL displayed in your terminal.

## 🖥️ How to Use

1. Launch the FireGuard application.
2. Select an input method: Upload Image, Use Webcam, or Sample Image.
3. Provide an image for analysis.
4. Adjust the detection sensitivity using the sidebar.
5. Click **Detect Fire**.
6. View the detection visualization and results.

## 🧠 Detection Approach

### AI-Based Detection

FireGuard uses a Mask R-CNN model configured with two classes:

- Background
- Fire

The model generates bounding boxes, segmentation masks, and confidence scores for detected regions.

### Color-Based Fallback

When model inference is unavailable or produces no detections, the application uses HSV color analysis to identify red, orange, and yellow regions that may resemble fire.

**Important:** Color-based detection is a heuristic and can produce false positives. It is not a replacement for a trained fire-detection model.

## ⚠️ Limitations

- Detection performance depends on the quality and compatibility of the model checkpoint.
- Color-based detection can confuse fire-like colors with other objects.
- Image-based detection cannot guarantee real-time fire safety or replace dedicated fire alarms.
- Performance depends on available hardware and image resolution.

## 🔮 Future Enhancements

- Real-time video-stream fire detection.
- Improved model accuracy with diverse training datasets.
- Smoke detection.
- Alert and notification integration.
- Model optimization for faster inference.
- Cloud deployment and monitoring.

## 👩‍💻 Author

**Vijaya Sri Chekuri**

GitHub: [@vijayasrichekuri](https://github.com/vijayasrichekuri)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Disclaimer:** FireGuard is an experimental computer vision project intended for demonstration and educational purposes. It should not be used as a certified fire-safety system.
