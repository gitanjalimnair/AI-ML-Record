# 🎓 Student Placement Prediction System

A Machine Learning web application that predicts whether a student is likely to be placed based on academic and professional attributes. The application is built using Flask, Scikit-learn, and Docker for easy deployment.

---

## Features

- Predicts placement status using Machine Learning
- Random Forest Classifier
- Modern Glassmorphism User Interface
- Dockerized application
- Real-time prediction
- Confidence score using predict_proba()

---

## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- Docker

---

## Project Structure

```
Student-Placement-Predictor/
│
├── app.py
├── train.py
├── placement_model.pkl
├── requirements.txt
├── Dockerfile
├── README.md
│
├── dataset/
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    ├── style.css
    └── images/
```

---

## Machine Learning Model

Algorithm Used:

- Random Forest Classifier

Target Variable:

- Placement (Yes / No)

Features Used:

- IQ
- Previous Semester Result
- CGPA
- Academic Performance
- Internship Experience
- Extra Curricular Score
- Communication Skills
- Projects Completed

---

## Run Locally

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train.py
```

Run Flask

```bash
python app.py
```

---

## Docker

Build

```bash
docker build -t student-placement-predictor .
```

Run

```bash
docker run -p 5000:5000 student-placement-predictor
```

---

## Author

Gitanjali M Nair