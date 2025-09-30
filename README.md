# 🚀 ALPR-TPS: Automatic License Plate Recognition for Terminal Petikemas Surabaya

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-orange.svg)
![YOLO](https://img.shields.io/badge/YOLOv8-ULTRAlytics-green.svg)
![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen.svg)

**Real-time License Plate Recognition system** built for PT Terminal Petikemas Surabaya. Detects and recognizes Indonesian license plates with high accuracy using YOLOv8 and EasyOCR.

## 📋 Table of Contents
- [Features](#-features)
- [Workflow](#-workflow)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#configuration)
- [Results](#-results)
- [Contributing](#-contributing)

## 🎯 Features

- **Real-time Detection**: Live video processing with bounding boxes
- **Smart OCR**: Intelligent text selection to filter out noise
- **Indonesian Plate Focus**: Optimized for local license plate formats
- **3-Panel Dashboard**: Professional live preview interface
- **Auto Logging**: CSV export with timestamps and confidence scores
- **GPU/CPU Support**: Automatic device detection for optimal performance
- **Resizable UI**: Flexible window with maintained aspect ratios

## 🔄 Workflow
Live Camera Feed → YOLO Plate Detection → Smart Text Selection → EasyOCR Processing → Result Display & Logging

### 🧠 Algorithm Innovations:
1. **Smart Text Selection** - Height clustering to distinguish main plate vs secondary text
2. **Dynamic Confidence Adjustment** - Context-aware scoring system
3. **Largest Plate Filtering** - Area-based selection in multi-vehicle environments

## 💻 Installation

### Prerequisites
- Python 3.11.9
- Webcam or camera source
- 4GB+ RAM recommended

### Step-by-Step Setup

1. **Clone Repository**
```
git clone https://github.com/your-username/ALPR-TPS.git
cd ALPR-TPS
```

2. **Create Virtual Environment (Recommended)**
```
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install Dependencies**
```
pip install -r requirements.txt
```

4. **Download Model Files**
Place your YOLO model in models/plate_detection/best.pt
EasyOCR models will auto-download on first run

## 🚀 Usage
#### Live Detection Mode
```
# Run on terminal where alpr.py is exist
python alpr_live.py
```

#### Controls
- C - Capture and recognize current plate
- ESC - Exit application
- Window X - Close program

#### Single Image Processing
```
python main.py
Place test images in images/ folder
```

## 📁 Project Structure
```
ALPR-TPS/
├── alpr_live.py          # 🎥 Main live detection application
├── main.py               # 🖼️ Single image processor
├── config.py             # ⚙️ Configuration settings
├── requirements.txt      # 📦 Dependencies
├── models/
│   └── plate_detection/
│       └── best.pt      # 🔧 YOLO model (add your own)
├── utils/
│   ├── plate_detector.py # 🎯 YOLO plate detection
│   └── ocr_reader.py     # 🔤 EasyOCR text recognition
├── images/               # 📸 Test images folder
├── output/               # 💾 Results & logs
│   └── log.csv          # 📝 Auto-generated capture log
└── README.md            # 📖 This file
```

## Configuration
Edit config.py to customize:
```
DETECTION_CONFIDENCE_THRESHOLD = 0.5    # Detection sensitivity
OCR_CONFIDENCE_THRESHOLD = 0.6          # OCR accuracy threshold
MIN_PLATE_TEXT_LENGTH = 3               # Minimum characters for valid plate
```

## 📊 Results
- Real-time Performance: 15-30 FPS (depending on hardware)
- Accuracy: >85% on Indonesian plates
- Output: CSV logs with timestamps, plate numbers, and confidence scores

Sample CSV Output:
```
timestamp,license_plate,confidence,filename
2024-09-30 14:25:30,B1234XYZ,0.85,captured_plate_2024-09-30_14-25-30.jpg
```

## 🤝 Contributing
This project was developed for PT Terminal Petikemas Surabaya as part of internship and academic research.

Potential Research Extensions:
- Multi-frame verification systems
- Indonesian plate format classification
- Adverse condition handling algorithms

## 📄 License
This project is for academic and research purposes.

## 👨‍💻 Developer
Your Name - PT Terminal Petikemas Surabaya Intern
Version: 1.0.0 | Last Update: September 2025

## 🎯 Quick Start
```
# Clone, install, and run!
git clone https://github.com/your-username/ALPR-TPS.git
cd ALPR-TPS
pip install -r requirements.txt
python alpr_live.py
```
