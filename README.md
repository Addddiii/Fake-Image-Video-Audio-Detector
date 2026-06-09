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

This project is a full-stack web application designed to detect whether media content (images, videos, or audio files) is authentic or AI-generated.

The system combines deep learning models with a modern web interface to provide fast and accessible detection of synthetic media. Users can upload media files and receive a prediction, confidence score, and probability breakdown indicating whether the content is real or AI-generated.

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
├── backend/
├── frontend/
├── README.md
└── .gitignore
```

---

## Requirements

### Backend

* Python 3.10+
* pip

### Frontend

* Node.js 18+
* npm

---

## Installation Instructions

### Clone Repository

```bash
git clone https://github.com/Addddiii/Fake-Image-Video-Audio-Detector.git
cd Fake-Image-Video-Audio-Detector
```

### Backend Installation

```bash
cd backend
pip install -r requirements.txt
```

### Frontend Installation

```bash
cd frontend
npm install
```

---

## Running Instructions

### Start Backend

```bash
cd backend
py -3.12 -m uvicorn app.main:app --reload
```

Backend URL:

```text
http://localhost:8000
```

### Start Frontend

```bash
cd frontend
npm run dev
```

Frontend URL:

```text
http://localhost:3000
```

---

## Environment Variables

Create:

```text
frontend/.env.local
```

Add:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Authentication

The application uses Firebase Authentication for secure user login and account management.

---

## GitHub Repository

https://github.com/Addddiii/Fake-Image-Video-Audio-Detector

---

## Deployed Application

The application is not currently deployed online. It can be run locally using the backend and frontend setup instructions above.

---

## Credential Information

The application uses Firebase Authentication for user login and account management.

Firebase credentials are not included in this submission for security reasons. To configure Firebase independently, create a Firebase project and place the Firebase service account credentials file in:

backend/firebase-credentials.json

---

## Data Management Information

### Database Type

No external database is used.

### Data Storage

User scan history is stored in browser local storage.

### Authentication

Firebase Authentication.

### Cloud Database

Not applicable.

### Connection Information

Not applicable.

---

## Release Notes

### Version 1.0

* Implemented image deepfake detection
* Implemented video deepfake detection
* Implemented audio deepfake detection
* Integrated Firebase authentication
* Added scan history functionality
* Added dashboard analytics and visualisations
* Completed FastAPI backend integration
* Completed Next.js frontend interface

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

This project demonstrates a practical application of machine learning for detecting AI-generated media. By integrating image, video, and audio analysis into a single platform, the system provides users with an accessible tool for identifying manipulated content and understanding the challenges posed by modern generative AI technologies.
