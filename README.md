# 🏏 IPL Winning Prediction

A machine learning web application that predicts the winning team of an IPL (Indian Premier League) match based on historical data such as venue, toss decision, and teams.  
The project uses a **Random Forest Classifier** and is deployed using **Flask**.

---

## 📂 Project Structure


📦 IPL_WINNING
├── Dataset/
│ └── matches.csv # Dataset used for training
├── static/
│ ├── bg-dots.png # UI background assets
│ ├── bg-tech.png
│ ├── result.css # CSS for result page
│ └── style.css # CSS for main page
├── templates/
│ ├── index.html # Home page (user input form)
│ └── result.html # Prediction result page
├── app.py # Flask application
├── model_building.py # Model training script
├── ipl_model.pkl # Trained RandomForest model
├── team_encoder.pkl # Label encoder for teams
├── toss_encoder.pkl # Label encoder for toss decision
├── venue_encoder.pkl # Label encoder for venue
├── winner_encoder.pkl # Label encoder for winners
├── requirements.txt # Python dependencies
├── runtime.txt # Python runtime version
└── README.md # Project documentation


---

## ⚙️ Tech Stack

- **Frontend**: HTML, CSS (Custom UI with `style.css` and `result.css`)
- **Backend**: Flask (Python web framework)
- **Machine Learning**: RandomForestClassifier (from scikit-learn)
- **Model Serialization**: joblib
- **Deployment**: Render / Localhost

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/IPL_WINNING.git
cd IPL_WINNING


2️⃣ Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows


3️⃣ Install dependencies
pip install -r requirements.txt


4️⃣ Run the Flask app
python app.py



5️⃣ Open in browser

Go to: http://127.0.0.1:5000

🧠 Model Training

The training script is available in model_building.py.

It trains a RandomForestClassifier on IPL match data (matches.csv).

Encoders (team_encoder.pkl, toss_encoder.pkl, venue_encoder.pkl, winner_encoder.pkl) are created to handle categorical variables.

The trained model is stored in ipl_model.pkl.

📊 Example Input & Output

Input (via web form):

Venue: Wankhede Stadium

Team 1: Mumbai Indians

Team 2: Chennai Super Kings

Toss Winner: Mumbai Indians

Toss Decision: Bat

Output:
👉 Predicted Winner: Mumbai Indians


