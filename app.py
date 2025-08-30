from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


rf_model = joblib.load("ipl_model.pkl")
team_encoder = joblib.load("team_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")
toss_encoder = joblib.load("toss_encoder.pkl")
winner_encoder = joblib.load("winner_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    team1 = request.form["team1"]
    team2 = request.form["team2"]
    venue = request.form["venue"]
    toss_winner = request.form["toss_winner"]

    team1_encoded = team_encoder.transform([team1])[0]
    team2_encoded = team_encoder.transform([team2])[0]
    venue_encoded = venue_encoder.transform([venue])[0]
    toss_encoded = toss_encoder.transform([toss_winner])[0]

    pred = rf_model.predict([[team1_encoded, team2_encoded, venue_encoded, toss_encoded]])[0]
    winner = winner_encoder.inverse_transform([pred])[0]

    return render_template("result.html", prediction_text=f"🏆 Predicted Winner: {winner}")

if __name__ == "__main__":
    app.run(debug=True)
