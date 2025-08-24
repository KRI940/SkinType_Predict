# 🌸 SkinSaathi

SkinSaathi is a **Machine Learning project** that predicts a user's **skin type** (e.g., Oily, Dry, Normal, Sensitive) based on various input parameters.  
The project comes with a **Flask-based frontend** for user interaction, making it simple and intuitive to use.

---

## 🚀 Features
- 📊 **Machine Learning Model** to classify skin type.  
- 🌐 **Flask Web Application** frontend.  
- 🧑‍💻 User-friendly interface to input skin-related parameters.  
- 🔮 Provides **instant predictions** for skin type.
- 💡 Shows **skincare recommendations** based on predicted skin type.
- ⚡ Lightweight and easy to deploy.  

---

## 🛠️ Tech Stack
- **Python**
- **Scikit-learn / Pandas / NumPy** (for ML model)
- **Flask** (for web frontend)
- **HTML, CSS, JS** (for UI)

---

## 📂 Project Structure
```
SkinSaathi/
│── static/                # CSS, JS, images
│── templates/             # HTML templates (Flask frontend)
│── model/                 # Trained ML model files
│── app.py                 # Main Flask application
│── train_model.py         # Script to train the ML model
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
```

---

## ⚙️ Installation & Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SkinSaathi.git
   cd SkinSaathi
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**
   ```bash
   python app.py
   ```

5. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```




