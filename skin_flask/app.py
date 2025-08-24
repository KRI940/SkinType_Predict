import os
import csv
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash

# ---- Colors to mirror Tkinter theme ----
TURQUOISE = "#48D1CC"
GRAY90 = "#F7F7F7"
TEAL = "#008080"
DARKBLUE = "#191970"
HOVER_BG = "#D3D3D3"

# ---- File paths ----
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "credentials.csv")

# Optionally set MODEL_DIR via environment variable. Defaults to project root.
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.dirname(__file__))

def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None

# -----------------------------
# Load ML assets (graceful if missing)
# -----------------------------
model = safe_load(os.path.join(MODEL_DIR, "skin_model.pkl"))
xgb_model = safe_load(os.path.join(MODEL_DIR, "xgb_model.pkl"))  # unused but loaded if present
scaler = safe_load(os.path.join(MODEL_DIR, "scaler.pkl"))
target_encoder = safe_load(os.path.join(MODEL_DIR, "target_encoder.pkl"))

def load_encoder(name):
    enc = safe_load(os.path.join(MODEL_DIR, f"{name}_encoder.pkl"))
    if enc is None:
        # Fallback classes matching gui2.py defaults
        defaults = {
            "gender": ["Male", "Female"],
            "weather": ["Hot", "Cold", "Moderate","Humid", "Dry"],
            "oiliness": ["Low", "Medium", "High"],
            "acne": ["No", "Yes"],
            "tightness_after_wash": ["No", "Yes"],
            "makeup_usage": ["Frequent", "Rare", "Never"],
            "flaking": ["No", "Yes"],
            "redness_itchiness": ["No","Yes"],
        }
        class SimpleEnc:
            def __init__(self, classes):
                self.classes_ = np.array(classes)
                self._map = {c:i for i,c in enumerate(classes)}
            def transform(self, arr):
                return np.array([self._map.get(a, 0) for a in arr])
        return SimpleEnc(defaults[name])
    return enc

encoders = {
    "gender": load_encoder("gender"),
    "weather": load_encoder("weather"),
    "oiliness": load_encoder("oiliness"),
    "acne": load_encoder("acne"),
    "tightness_after_wash": load_encoder("tightness_after_wash"),
    "makeup_usage": load_encoder("makeup_usage"),
    "flaking": load_encoder("flaking"),
    "redness_itchiness": load_encoder("redness_itchiness"),
}

# -----------------------------
# Domain data from gui2.py
# -----------------------------
skin_care_ingredients = {
    "oily": [
        "Raw milk", "Turmeric", "Sandalwood", "Honey", "Aloe vera",
        "Lemon", "Ice", "Neem", "Rose water", "Cucumber extract", "Multani mitti"
    ],
    "dry": [
        "Honey", "Avocado oil", "Coconut oil", "Yogurt", "Aloe vera",
        "Cucumber extract", "Almond oil", "Shea butter", "Castor oil", "Oatmeal"
    ],
    "sensitive": [
        "Aloe vera", "Green tea", "Chamomile", "Niacinamides",
        "Apple cider vinegar", "Honey", "Centella asiatica", "Spirulina",
        "Raw milk", "Turmeric", "Oat extract", "Sandalwood", "Neem", "Tea tree oil"
    ],
    "normal": [
        "Raw milk", "Honey", "Yogurt", "Multani mitti", "Aloe vera",
        "Castor oil", "Coconut oil", "Ice", "Rose water", "Cucumber", "Lemon"
    ],
}

skin_care_remedies = {
    "oily": [
        "Turmeric & Multani Mitti Mask – Mix 2 tsp multani mitti + ½ tsp turmeric + rose water. Apply 15 mins.",
        "Neem & Cucumber Pack – Grind neem leaves + 2 tbsp cucumber juice. Apply 20 mins.",
        "Lemon-Honey Cleanser – Mix 1 tsp lemon juice + 1 tsp honey. Massage, then rinse.",
        "Aloe Vera & Ice Rub – Apply aloe vera gel, rub ice cube for 2 mins.",
        "Turmeric + Honey Spot Treatment → Dab on pimples, wash after 10 mins.",
        "Multani Mitti + Lemon Pack → Oil control + brightening.",
        "Neem + Aloe Vera Gel → Apply overnight for acne.",
        "Rose Water + Ice Cube Rub → Tightens pores."
        ],
    "dry": [
        "Honey & Yogurt Mask – Mix 2 tsp honey + 1 tsp yogurt. Apply 20 mins.",
        "Avocado Oil & Almond Oil Massage – Mix equal parts. Massage before bed.",
        "Oatmeal & Aloe Vera Pack – Blend 2 tbsp oatmeal + 1 tbsp aloe vera. Apply 15 mins.",
        "Shea Butter & Coconut Oil Cream – Whip shea butter + coconut oil. Apply daily.",
        "Oatmeal + Honey Scrub → Gentle exfoliation + hydration.",
        "Aloe Vera + Coconut Oil Mask → Intense moisturization.",
        "Avocado Oil + Yogurt Pack → Restores suppleness.",
        "Shea Butter Night Cream → Locks in moisture overnight."
        ],
    "sensitive": [
        "Chamomile & Aloe Vera Gel – Brew chamomile tea + aloe vera. Apply 15 mins.",
        "Green Tea & Honey Mask – Brew green tea, mix with honey. Apply 15 mins.",
        "Oat Extract & Spirulina Pack – Mix oat powder + spirulina + rose water. Apply 20 mins.",
        "Neem & Sandalwood Paste – Mix neem + sandalwood + raw milk. Apply 10–15 mins.",
        "Diluted Apple Cider Vinegar – Mix 1 part ACV + 3 parts water. Use as toner.",
        "Chamomile + Aloe Vera Pack → Soothes irritation.",
        "Green Tea + Honey Mask → Anti-redness + mild glow.",
        "Neem + Sandalwood Paste → Gentle acne control.",
        "Spirulina + Oat Extract Pack → Calms inflammation."
        ],
    "normal": [
        "Raw Milk & Honey Cleanser – Mix raw milk + honey. Massage & rinse.",
        "Aloe Vera & Cucumber Pack – Mix aloe vera gel + cucumber juice. Apply 15 mins.",
        "Multani Mitti & Rose Water Pack – Mix multani mitti + rose water. Apply 20 mins.",
        "Coconut Oil Night Massage – Apply coconut oil before bed.",
        "Ice & Lemon Rub – Freeze lemon juice + water into cubes, rub on face weekly.",
        "Milk + Lemon Cleanser → Brightens skin.",
        "Aloe Vera + Cucumber Gel → Refreshing hydration.",
        "Multani Mitti + Rose Water Pack → Detox & glow.",
        "Coconut Oil Massage (weekly) → Maintains balance."
        ]
    }

# Image filenames (place images in static/img with these filenames)
ingredient_images = {
    "Raw milk": "raw_milk.jpg",
    "Turmeric": "turmeric.jpg",
    "Sandalwood": "sandalwood.jpg",
    "Honey": "honey.jpg",
    "Aloe vera": "aloe_vera.jpg",
    "Lemon": "lemon.jpg",
    "Ice": "ice.jpg",
    "Neem": "neem.jpg",
    "Rose water": "rose_water.jpg",
    "Cucumber extract": "cucumber_extract.jpg",
    "Multani mitti": "multani_mitti.jpg",
    "Avocado oil": "avocado_oil.jpg",
    "Coconut oil": "coconut_oil.jpg",
    "Yogurt": "yogurt.jpg",
    "Almond oil": "almond_oil.jpg",
    "Shea butter": "shea_butter.jpg",
    "Castor oil": "castor_oil.jpg",
    "Oatmeal": "oatmeal.jpg",
    "Green tea": "green_tea.jpg",
    "Chamomile": "chamomile.jpg",
    "Niacinamides": "niacinamides.jpg",
    "Apple cider vinegar": "apple_cider_vinegar.jpg",
    "Centella asiatica": "centella_asiatica.jpg",
    "Spirulina": "spirulina.jpg",
    "Oat extract": "oat_extract.jpg",
    "Tea tree oil": "tea_tree_oil.jpg",
    "Cucumber": "cucumber_extract.jpg"
}

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "replace-with-a-secret-key"

def users_exist():
    return os.path.exists(CREDENTIALS_FILE) and os.path.getsize(CREDENTIALS_FILE) > 0

@app.context_processor
def inject_theme():
    return dict(TURQUOISE=TURQUOISE, GRAY90=GRAY90, TEAL=TEAL, DARKBLUE=DARKBLUE, HOVER_BG=HOVER_BG)

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not os.path.exists(CREDENTIALS_FILE):
            flash("No users registered yet!", "error")
            return redirect(url_for("login"))
        with open(CREDENTIALS_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["username"] == username and row["password"] == password:
                    session["user"] = username
                    flash("Logged in successfully.", "ok")
                    return redirect(url_for("main_app"))
        flash("Invalid Username or Password", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "ok")
    return redirect(url_for("landing"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm = request.form.get("confirm", "").strip()

        if not all([name, username, password, confirm]):
            flash("All fields are required!", "error")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match!", "error")
            return redirect(url_for("register"))

        file_exists = os.path.exists(CREDENTIALS_FILE)
        if file_exists:
            with open(CREDENTIALS_FILE, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["username"] == username:
                        flash("Username already exists!", "error")
                        return redirect(url_for("register"))
        with open(CREDENTIALS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["name", "username", "password"])
            writer.writerow([name, username, password])
        flash("Account created successfully! Please log in.", "ok")
        return redirect(url_for("login"))

    return render_template("register.html")

def require_login():
    if "user" not in session:
        flash("Please log in first to access the main app.", "error")
        return False
    return True

@app.route("/main", methods=["GET", "POST"])
def main_app():
    if not require_login():
        return redirect(url_for("login"))

    # Build choices from encoders
    def classes_of(enc):
        arr = getattr(enc, "classes_", [])
        try:
            return list(arr.tolist())
        except Exception:
            return list(arr)

    choices = {
        "Gender": classes_of(encoders["gender"]),
        "Weather": classes_of(encoders["weather"]),
        "Oiliness": classes_of(encoders["oiliness"]),
        "Acne": classes_of(encoders["acne"]),
        "Tightness After Wash": classes_of(encoders["tightness_after_wash"]),
        "Makeup Usage": classes_of(encoders["makeup_usage"]),
        "Flaking": classes_of(encoders["flaking"]),
        "Redness/Itchiness": classes_of(encoders["redness_itchiness"]),
    }

    predicted = session.get("predicted_skin_type", "")
    return render_template("main.html", choices=choices, predicted=predicted)

@app.route("/predict", methods=["POST"])
def predict():
    if not require_login():
        return redirect(url_for("login"))

    if model is None or target_encoder is None:
        flash("Model files not loaded. Please place your *.pkl files in the project folder or set MODEL_DIR.", "error")
        return redirect(url_for("main_app"))

    try:
        # Parse inputs
        age = int(request.form.get("Age"))
        water = float(request.form.get("Water Intake (liters)"))
        gender = encoders["gender"].transform([request.form.get("Gender")])[0]
        weather = encoders["weather"].transform([request.form.get("Weather")])[0]
        oiliness = encoders["oiliness"].transform([request.form.get("Oiliness")])[0]
        acne = encoders["acne"].transform([request.form.get("Acne")])[0]
        tight = encoders["tightness_after_wash"].transform([request.form.get("Tightness After Wash")])[0]
        makeup = encoders["makeup_usage"].transform([request.form.get("Makeup Usage")])[0]
        flaking = encoders["flaking"].transform([request.form.get("Flaking")])[0]
        redness = encoders["redness_itchiness"].transform([request.form.get("Redness/Itchiness")])[0]

        X = np.array([[age, gender, water, weather, oiliness, acne, tight, makeup, flaking, redness]], dtype=float)
        if scaler is not None:
            X = scaler.transform(X)

        pred = model.predict(X)
        skin_type = target_encoder.inverse_transform(pred)[0].lower()
        session["predicted_skin_type"] = skin_type
        flash(f"Predicted Skin Type: {skin_type.capitalize()}", "ok")
    except Exception as e:
        flash(f"Error during prediction: {e}", "error")

    return redirect(url_for("main_app"))

@app.route("/recommendations")
def recommendations():
    if not require_login():
        return redirect(url_for("login"))
    stype = session.get("predicted_skin_type", "")
    if not stype:
        flash("Please predict skin type first.", "error")
        return redirect(url_for("main_app"))

    ingredients = skin_care_ingredients.get(stype, [])
    remedies = skin_care_remedies.get(stype, [])
    images = [(ing, ingredient_images.get(ing, "placeholder.jpg")) for ing in ingredients]
    return render_template("recommendations.html", skin_type=stype, images=images, remedies=remedies)

# Placeholder pages
@app.route("/features")
def features():
    return render_template("placeholder.html", title="Features", message="Features Page Placeholder")

@app.route("/risks")
def risks():
    return render_template("placeholder.html", title="Risk Detection", message="Risk Detection Page Placeholder")

@app.route("/how-it-works")
def how_it_works():
    return render_template("placeholder.html", title="How it Works", message="How it Works Page Placeholder")

@app.route("/faq")
def faq():
    return render_template("placeholder.html", title="FAQ", message="FAQ Page Placeholder")

if __name__ == "__main__":
    # For local testing
    app.run(debug=True, port=5000)
