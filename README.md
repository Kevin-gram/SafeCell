# 🧬 SafeCell: AI-Powered Malaria Diagnosis Dashboard

**SafeCell** is an AI-powered dashboard designed to assist healthcare workers — especially in rural and underserved communities — in the early and accurate detection of malaria using blood smear images. The system also helps visualize regional malaria trends and collect real-time user feedback, enabling data-driven public health responses.

---

## 📸 App Interface Previews

| **Mobile View** | **Tablet View** | **Desktop View** |
|-----------------|------------------|-------------------|
| ![safeCell_mobile](https://github.com/user-attachments/assets/63ab94aa-9288-4368-845c-0e54687b3e6c) | ![safeCell_tablet](https://github.com/user-attachments/assets/837ecd70-32d5-431c-be9c-f4bd2015859a) | ![SafeCell_desktop](https://github.com/user-attachments/assets/b069a230-6fbc-4929-a979-aa7d3d14b8ee) |

---

## 🎥 Video Demo

Watch the full 5-minute demonstration below:  
📽️ [![Watch the Demo](https://img.shields.io/badge/Video-Demo-blue)](https://www.loom.com/share/8d04a2a22d084b9bb3e4a35b773a14e6?sid=e011f1a4-1ec1-4cda-a528-c44d8b593622)

---

## 🚀 Features

- 🧠 **Malaria Detection**  
  Upload and analyze blood smear images with a deep learning model.

- 📊 **Detection Statistics**  
  Visual dashboards showing trends and infection patterns (powered by Chart.js).

- 🗺️ **Location-Based Insights**  
  Interactive map integration for geospatial case tracking.

- 📱 **Responsive Design**  
  Fully optimized for all screen sizes — mobile, tablet, and desktop.

- 🌗 **Dark Mode Support**  
  User-friendly theme toggling between light and dark modes.

- 🌍 **Multilingual Interface**  
  Supports multiple languages for inclusivity.

- 💬 **Community Feedback Collection**  
  Allows healthcare workers to provide feedback for continuous system improvement.

- 🔐 **Login (Mock Session)**  
  Simulated login system with placeholder credentials for testing.

☁️ Deployment Plan
SafeCell is deployed across three platforms using free-tier services:

🌐 Frontend: Netlify
Platform: Netlify

Technology: React.js (Vite)

Build Command: npm run build

Publish Directory: dist

Deployment Method: GitHub integration with automatic deployment on push to main branch

Live URL: To be added after deployment

⚙️ Backend API: Render
Platform: Render

Technology: FastAPI / Flask (Python)

Start Command: gunicorn app:app (or equivalent)

Deployment Method: GitHub auto-deploy from backend subdirectory

Live URL: To be added after deployment

🧠 AI Model: Hugging Face Spaces
Platform: Hugging Face Spaces

Technology: Gradio or FastAPI

Deployment Method: Upload model weights and inference script

Usage: Public interface and REST API for image classification

Live URL: To be added after deployment
---

## ⚙️ Setup Instructions

Follow the steps below to get the project running locally:

```bash
# 1. Clone the repository
git clone https://github.com/Kevin-gram/SafeCell.git
cd SafeCell

# 2. Install dependencies
npm install

# 3. Start the development server
npm run dev
