# Programming Project 1: Assignment 1

## Fake Image / Video / Audio Detector

*(Addressing Challenges of AI-Generated Content)*

---

## Team Members

* Mohammed Awad (s3946625)
* Khalid Dayib (s3947672)
* Kane Fuller (s4002534)
* Hiten Verma (s4040528)
* Aditya Ajay (s4041761)

---

## Supervisor

Vic Ciesielski

---

## Project Overview

This project is a full-stack web application designed to detect whether media content (images, videos, or audio files) is real or AI-generated.

The system combines deep learning models with a modern web interface to provide fast, accessible, and user-friendly detection of synthetic media. Users can upload media files and receive a prediction, confidence score, and probability breakdown indicating whether the content is authentic or AI-generated.

---

## Features

* Image deepfake detection using EfficientNet-B0
* Video deepfake detection using frame-based analysis
* Audio deepfake detection using spectrogram classification
* Firebase user authentication
* Scan history tracking
* Dashboard analytics and visualisations
* FastAPI backend and Next.js frontend
* Confidence score reporting for all predictions

---

## Project Structure

```text
Fake-Image-Video-Audio-Detector/
├── backend/          # FastAPI backend
├── frontend/         # Next.js frontend
├── README.md
└── .gitignore
```

---

## Requirements

### Backend

* Python 3.10+
* pip

### Frontend

* Node.js (v18+ recommended)
* npm

---

## Setup Instructions

### Clone Repository

```bash
git clone https://github.com/Addddiii/Fake-Image-Video-Audio-Detector.git
cd Fake-Image-Video-Audio-Detector
```

---

## Backend Setup

Install backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Run the backend:

```bash
py -3.12 -m uvicorn app.main:app --reload
```

Backend URL:

```text
http://localhost:8000
```

---

## Frontend Setup

Install frontend dependencies:

```bash
cd frontend
npm install
```

Run the frontend:

```bash
npm run dev
```

Frontend URL:

```text
http://localhost:3000
```

---

## Environment Variables

Create the following file:

```text
frontend/.env.local
```

Add:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Model Files

Place trained model files inside:

```text
backend/models/
```

Required files:

* image_model.pth
* video_model.pth
* audio_model.pth

---

## API Endpoints

### Image Detection

```http
POST /predict/image
```

### Video Detection

```http
POST /predict/video
```

### Audio Detection

```http
POST /predict/audio
```

### Authentication

```http
POST /auth/verify
GET  /auth/me
```

### Health Check

```http
GET /health
```

---

## Firebase Setup

Place the Firebase credentials file inside:

```text
backend/firebase-credentials.json
```

The credentials file is used by the backend to verify authenticated user tokens.

**Important:** Do not commit Firebase credentials to GitHub.

---

## FFmpeg

FFmpeg may be required on the backend machine for certain audio and video formats such as `.m4a`, `.mov`, `.mkv`, and `.webm`.

End users accessing the website do not need to install FFmpeg.

---

## Tech Stack

### Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS
* Firebase Authentication
* Recharts

### Backend

* FastAPI
* Python
* PyTorch
* Torchvision
* OpenCV
* Librosa

### Machine Learning

* EfficientNet-B0
* Image classification model
* Video frame analysis model
* Audio spectrogram classification model

---

## Version Information

* Version: 1.0
* Course: Programming Project 1
* Year: 2026

---

## Summary

This project demonstrates a practical application of machine learning for detecting AI-generated media. By integrating image, video, and audio analysis into a single platform, the system provides users with an accessible tool for identifying potentially manipulated content and exploring the challenges posed by modern generative AI technologies.
