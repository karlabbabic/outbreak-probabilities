import math
from js import document


def by_id(element_id):
    return document.getElementById(element_id)


def set_text(element_id, text):
    by_id(element_id).textContent = text


def createWeekInputs(event=None):
    num_weeks = int(by_id("numWeeks").value)
    container = by_id("weekInputs")

    html = []
    for i in range(1, num_weeks + 1):
        html.append(
            f"""
            <div class="week-row">
              <label for="week{i}">Week {i}:</label>
              <input type="number" id="week{i}" step="any" min="0">
            </div>
            """
        )

    container.innerHTML = "".join(html)


def on_model_change(event=None):
    # Kept for symmetry with py-input; no special action needed.
    return None


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def predict_probability(model, weeks):
    """
    Lightweight in-browser PMO-style predictor.

    Replace this function later if you want to load exported RF/GB model
    artifacts from your notebook workflow.
    """
    total = sum(weeks)
    recent = weeks[-1] if weeks else 0.0
    first = weeks[0] if weeks else 0.0
    zeros = sum(1 for x in weeks if x == 0)

    trend = recent - first
    slope = (weeks[-1] - weeks[0]) / max(1, len(weeks) - 1) if len(weeks) > 1 else 0.0

    if model == "RF":
        score = (
            -1.10
            + 0.38 * total
            + 0.22 * recent
            + 0.12 * max(0.0, trend)
            - 0.10 * zeros
            + 0.08 * abs(slope)
        )
    else:  # GB
        score = (
            -0.95
            + 0.42 * total
            + 0.25 * recent
            + 0.16 * max(0.0, trend)
            - 0.08 * zeros
            + 0.10 * abs(slope)
        )

    pmo = sigmoid(score)
    predicted_label = "Major outbreak" if pmo >= 0.5 else "No major outbreak"
    return predicted_label, pmo


def getPrediction(event=None):
    model = by_id("model").value
    num_weeks = int(by_id("numWeeks").value)

    weeks = []
    for i in range(1, num_weeks + 1):
        el = by_id(f"week{i}")
        if el is None:
            set_text("result", f"Please enter a value for Week {i}.")
            return

        raw = el.value.strip()
        if raw == "":
            set_text("result", f"Please enter a value for Week {i}.")
            return

        try:
            value = float(raw)
        except ValueError:
            set_text("result", f"Week {i} must be a number.")
            return

        weeks.append(value)

    predicted_label, pmo = predict_probability(model, weeks)
    probability = pmo

    set_text(
        "result",
        f"Prediction: {predicted_label} | Probability: {probability:.4f} | PMO: {pmo:.4f}"
    )


createWeekInputs()
