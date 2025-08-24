
# SkinSaathi (Flask)

A Flask frontend that mirrors your Tkinter GUI: Landing page, Register, Login, Main App with prediction form, and a Recommendations page with ingredient grid + remedies list.

## 1) Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Put your model files

Copy these into the project folder (or set `MODEL_DIR` env var to their folder):

- `skin_model.pkl`
- `target_encoder.pkl`
- `scaler.pkl` (recommended, since the model was trained on scaled features)
- `*_encoder.pkl` for each categorical: gender, weather, oiliness, acne, tightness_after_wash, makeup_usage, flaking, redness_itchiness.

## 3) Put your images (optional)

Replace placeholders in `static/img/`:
- `logo1.png`, `img2.jpg`, `img3.jpg`
- Ingredient images named as in `app.py` (e.g. `aloe_vera.jpg`, `multani_mitti.jpg`)

## 4) Run

```bash
# Optionally point to a different folder containing your model files
# set MODEL_DIR=K:\Kritika\Internship Project\AI Skin type Predictor\skin  (Windows PowerShell: $env:MODEL_DIR="...")
# export MODEL_DIR="/path/to/skin" (macOS/Linux)

python app.py
# open http://127.0.0.1:5000
```

## Notes

- Credentials are stored in a simple CSV file `credentials.csv` for parity with your Tkinter app.
- To reset accounts, delete `credentials.csv`.
- The UI color scheme, layout, and page flow replicate the Tkinter version.
