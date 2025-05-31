# Mu'een (مُعين) - Sign Language Hospital Assistant

An intelligent web application designed to assist deaf and mute patients in hospital settings by recognizing Arabic sign language and providing real-time answers to their queries.

## 🎯 Project Overview

**Mu'een** is a specialized system developed for hospital reception areas that bridges the communication gap between deaf/mute patients and healthcare staff. The system uses computer vision to detect sign language gestures and provides intelligent responses through a RAG (Retrieval-Augmented Generation) system powered by OpenAI's GPT-4.

### Key Features

- **Real-time Sign Language Detection**: Uses YOLOv11 model trained on custom dataset
- **Arabic Language Support**: Full Arabic interface and responses
- **Intelligent Question-Answer System**: RAG-based system for accurate responses
- **Web-Based Interface**: Easy-to-use browser interface
- **Live Camera Feed**: Real-time video processing and detection

## 🏥 Target Use Case

Specifically designed for **King Salman Hospital** reception area to help deaf and mute patients with:
- Booking appointments
- Getting lab results
- Finding pharmacy locations
- Emergency procedures

## 🗂️ Project Structure

```
mu'een/
├── app.py                          # Main Flask application
├── sign_language_rag.py           # RAG system and LLM integration
├── SignLanguage_YOLO11_Final.ipynb # Model training notebook
├── qa.txt                         # Knowledge base Q&A pairs
├── data.yaml                      # Dataset configuration
├── run_YOLO_cam.py               # Standalone camera testing
├── resize.py                     # Image preprocessing utility
├── creating_images.py            # Dataset creation tool
├── change_image_names.py         # Dataset organization utility
├── yolov11_run/weights/best.pt   # Trained YOLO model weights
└── templates/
    └── index.html                # Web interface
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera device
- OpenAI API key
- Git


### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Environment Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 3: Download Model Weights

Ensure the trained YOLO model is placed at:
```
yolov11_run/weights/best.pt
```

## 🚀 Running the Application

### Start the Web Application

```bash
python app.py
```

The application will be available at: `http://localhost:5000`


## 📊 Supported Sign Language Words

The system currently recognizes 8 medical-related sign language gestures:

| **English** | **Arabic** | **Context** |
|-------------|------------|-------------|
| Analysis    | تحليل      | Lab tests |
| Appointment | موعد       | Booking appointments |
| Complaint   | شكوى       | Filing complaints |
| Dispensing  | صرف        | Pharmacy services |
| Emergency   | طوارئ      | Emergency situations |
| Fasting     | صيام       | Pre-test fasting |
| Medicine    | دواء       | Medication queries |
| Result      | نتيجة      | Test results |

## 🧠 How It Works

1. **Sign Detection**: Camera captures real-time video feed
2. **YOLO Processing**: Custom YOLOv11 model detects sign language gestures
3. **Translation**: English predictions are translated to Arabic
4. **Sentence Building**: Multiple signs are combined into meaningful sentences
5. **RAG System**: Sentence is processed through retrieval system to find relevant information
6. **LLM Response**: OpenAI GPT-4 generates contextual Arabic responses
7. **Display**: Results are shown in real-time web interface

## 🔧 Technical Architecture

### Core Components

- **Flask Web Server**: Handles HTTP requests and serves the web interface
- **YOLOv11 Model**: Custom-trained computer vision model for sign detection
- **FAISS Vector Database**: Efficient similarity search for question matching
- **OpenAI GPT-4**: Natural language generation for responses
- **Sentence Transformers**: Multilingual embeddings for semantic search


## 🎯 Custom Dataset

The sign language dataset was **self-created** specifically for this project:
- **8 medical sign language gestures**
- **Custom annotations** for hospital context
- **YOLOv11 format** for object detection
- **Arabic medical terminology** focus


## 📋 Requirements

```
flask
ultralytics
opencv-python
sentence-transformers
faiss-cpu
openai
python-dotenv
numpy
torch
torchvision
```


## 👥 Team members

This project was developed by a team of five students:

Shaden Alturki

Abdullah Aljubran

Jana Almalki

Shahad Albalawi

Raghad Alqahtani



**Mu'een (مُعين)** - Making healthcare accessible for everyone through technology.
