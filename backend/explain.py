def generate_explanation(prediction, adv):

    label = prediction["label"]
    score = adv["score"]

    consistency = adv.get("consistency", 1.0)
    confidence_drop = adv.get("confidence_drop", 0.0)
    entropy = adv.get("entropy_increase", 0.0)
    kl = adv.get("kl_divergence", 0.0)

    explanation = f"The model predicted '{label}'. "

    # risk level
    if score < 0.3:
        explanation += "The prediction is stable under perturbations. "
    elif score < 0.6:
        explanation += "The prediction shows moderate instability. "
    else:
        explanation += "The prediction is highly unstable and may be adversarial. "

    # reasoning
    if consistency < 0.7:
        explanation += "Predictions changed across transformations. "

    if confidence_drop > 0.2:
        explanation += "Model confidence dropped significantly. "

    if entropy > 0.1:
        explanation += "Prediction uncertainty increased. "

    if kl > 0.1:
        explanation += "Probability distribution shifted across variants. "

    return explanation