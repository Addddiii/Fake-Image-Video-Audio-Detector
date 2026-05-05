# Programming Project 1: Assignment 1

## Fake Image / Video / Audio Detector  
*(Addressing challenges of AI-generated content)*

---

## Team Members

- Mohammed Awad (s3946625)  
- Khalid Dayib (s3947672)  
- Kane Fuller (s4002534)  
- Hiten Verma (s4040528)  
- Aditya Ajay (s4041761)  

---

## Supervisor

Vic Ciesielski  

---

## Project Overview

This project is a full-stack web application designed to detect whether media content (image, video, or audio) is real or AI-generated.

It combines deep learning models with a web interface to provide fast and accessible detection for modern AI-generated media.

---

## Features

- Image deepfake detection (EfficientNet-B0)
- Video deepfake detection using frame-based analysis
- Audio deepfake detection using spectrogram classification
- Firebase authentication (token verification)
- FastAPI backend + Next.js frontend

---

## Project Structure

FAKE-IMAGE-VIDEO-AUDIO-DETECTOR/
├── backend/              # FastAPI backend
├── frontend/             # Next.js frontend
├── training/             # Model training scripts
├── preprocess_scripts/   # Data preprocessing
├── videos/               # Dataset (optional)
├── README.md

---

## Requirements

### Backend
- Python 3.10+
- pip

### Frontend
- Node.js (v18+ recommended)
- npm

---

## Setup Instructions

### Clone Repository

git clone https://github.com/Addddiii/Fake-Image-Video-Audio-Detector.git  
cd Fake-Image-Video-Audio-Detector

---

## Backend Setup

cd backend  
pip install -r requirements.txt  

Run backend:

py -3.12 -m uvicorn main:app --reload  

Backend URL:  
http://localhost:8000  

---

## Frontend Setup

cd frontend  
npm install  

Run frontend:

npm run dev  

Frontend URL:  
http://localhost:3000  

---

## Environment Variables

Create:

frontend/.env.local  

Add:

NEXT_PUBLIC_API_URL=http://localhost:8000  

---

## Model Files

Place trained models inside:

backend/models/

Required:
- image_model.pth  
- video_model.pth  
- audio_model.pth  

---

## API Endpoint

POST /upload  

Upload:
- image  
- video  
- audio  

Returns prediction (fake or real) with confidence.

---

## Optional: Firebase Setup

Place credentials file:

backend/firebase-credentials.json  

Used for verifying user login tokens.

---

## Tech Stack

- FastAPI (Backend API)
- PyTorch (Machine Learning)
- Next.js (Frontend)
- Firebase (Authentication)
- OpenCV (Video Processing)
- Librosa (Audio Processing)

---

## Version Information

- Version: 1.0  
- Date: 05/05/2026  
- Course: VC-683 (Project Groups)  

---

## Summary

This system demonstrates a practical application of machine learning for detecting AI-generated media, integrating image, video, and audio analysis into a single unified platform.