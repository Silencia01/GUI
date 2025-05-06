from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import flask-cors
import random
import torch
import joblib
import numpy as np
from torch import nn

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This will allow all domains to access your API


# --- Define your model ---
class BinaryMLP(nn.Module):
    def __init__(self, input_dim):
        super(BinaryMLP, self).__init__()
        self.shared1 = nn.Linear(input_dim, 384)
        self.shared2 = nn.Linear(384, 192)
        self.output = nn.Linear(192, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = x * torch.sigmoid(x)
        x = self.dropout(self.shared1(x))
        x = x * torch.sigmoid(x)
        x = self.dropout(self.shared2(x))
        return self.output(x)


# --- Load model and scaler ---
INPUT_DIM = 10
model = BinaryMLP(INPUT_DIM)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

scaler = joblib.load("scaler.pkl")

# SAVED THE PERCENTILE SCORE FROM THE ACTUAL DATASET AND LOADED
percentile_thresholds = joblib.load("percentile_thresholds.pkl")
print("Loaded percentile thresholds:", percentile_thresholds)


# NEW CHANGE MADE
# --- Helper function to assign percentile category ---
def map_to_percentile(score, thresholds):
    if score <= thresholds["p10"]:
        return 0  # Low
    elif score <= thresholds["p90"]:
        return 1  # Average
    else:
        return 2  # High


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_sequence", methods=["POST"])
def get_sequence():
    span = int(request.json.get("span", 5))
    grid_size = 4
    total_boxes = grid_size * grid_size
    sequence = random.sample(range(total_boxes), span)
    return jsonify({"sequence": sequence})


@app.route("/check_sequence", methods=["POST"])
def check_sequence():
    user_seq = request.json.get("user_sequence", [])
    correct_seq = request.json.get("correct_sequence", [])
    result = user_seq == correct_seq
    score = 1 if result else 0
    return jsonify({"correct": result, "score": score})


@app.route("/digit-symbol")
def digit_symbol():
    return render_template("digit_symbol.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    age = data["age"]
    gender = data["gender"]
    education = data["education"]
    memory_score = data["memory_score"]
    executive = data.get("executive_score", 50.0)
    reasoning = data.get("reasoning_score", 50.0)

    pct_map = {"Low": 0, "Average": 1, "High": 2}
    memory_pct = pct_map.get(data.get("memory_percentile", "Average"), 1)
    executive_pct = pct_map.get(data.get("executive_percentile", "Average"), 1)
    reasoning_pct = pct_map.get(data.get("reasoning_percentile", "Average"), 1)

    gender_male = 1 if gender == "Male" else 0
    gender_female = 1 - gender_male

    raw_input = np.array([[age, memory_score, executive, reasoning, gender_female, gender_male, education]])
    scaled_input = scaler.transform(raw_input)
    final_input = np.concatenate([scaled_input, [[memory_pct, executive_pct, reasoning_pct]]], axis=1)
    input_tensor = torch.tensor(final_input, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor).squeeze()
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob >= 0.5 else 0

    return jsonify({"prediction": prediction, "probability": prob})


@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json()

    # Extract participant info
    participant = data.get("participant", {})
    scores = data.get("scores", {})
    age = int(participant.get("age", 30))  # Ensure it's an integer
    gender = participant.get("gender", "Female")  # Default to Female
    education = participant.get("education", "high_school")  # Default to "high_school"

    # One-hot encode gender
    gender_male = 1 if gender == "Male" else 0
    gender_female = 1 - gender_male

    # Education encoding
    education_map = {"high_school": 0, "undergrad": 1, "postgrad": 2}
    education_level = education_map.get(education, 0)  # Default to "high_school"

    # CHANGES MADE HERE
    # Extract cognitive scores
    memory_score = scores.get("memory_score", 0)
    executive_score = scores.get("executive_score", 0)
    reasoning_score = scores.get("reasoning_score", 0)

    # Assign percentile classes dynamically
    memory_pct = map_to_percentile(memory_score, percentile_thresholds["memory"])
    executive_pct = map_to_percentile(executive_score, percentile_thresholds["executive"])
    reasoning_pct = map_to_percentile(reasoning_score, percentile_thresholds["reasoning"])

    # Add interpretability labels
    pct_label_map = {0: "Low", 1: "Average", 2: "High"}
    memory_label = pct_label_map[memory_pct]
    executive_label = pct_label_map[executive_pct]
    reasoning_label = pct_label_map[reasoning_pct]

    # Prepare the input for the model
    raw_input = np.array([[age, memory_score, executive_score, reasoning_score,
                           gender_female, gender_male, education_level]])
    scaled_input = scaler.transform(raw_input)
    final_input = np.concatenate([scaled_input, [[memory_pct, executive_pct, reasoning_pct]]], axis=1)
    input_tensor = torch.tensor(final_input, dtype=torch.float32)

    # Run the prediction
    with torch.no_grad():
        output = model(input_tensor).squeeze()
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob >= 0.5 else 0

    return jsonify({
        "status": "success",
        "prediction": prediction,
        "probability": prob,
        "memory_percentile": memory_label,
        "executive_percentile": executive_label,
        "reasoning_percentile": reasoning_label
    })


if __name__ == "__main__":
    app.run(debug=True)