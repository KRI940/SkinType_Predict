
# SkinSaathi (Flask)

SkinSaathi is a Flask-based web application that predicts skin type using a trained ML model and provides personalized skincare ingredient recommendations with remedies.

---

## Features
- **Landing Page** – Welcomes users and explains the app.
- **User Authentication** – Register and login pages.
- **Skin Type Prediction** – Predicts skin type using `skin_model.pkl`.
- **Ingredient Recommendations** – Grid view with remedies list.
- **Lightweight Flask Frontend** – Accessible from any browser.

---

## 1) Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/<your-username>/SkinType_Predict.git
cd SkinType_Predict

python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
