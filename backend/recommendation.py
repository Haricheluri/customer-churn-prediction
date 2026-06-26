def get_recommendation(prediction, probability):

    if prediction == 1:

        if probability >= 0.90:
            return {
                "risk": "Very High",
                "action": "Assign retention manager, offer premium discount, schedule immediate follow-up."
            }

        elif probability >= 0.75:
            return {
                "risk": "High",
                "action": "Send personalized retention offer and discount coupon."
            }

        elif probability >= 0.50:
            return {
                "risk": "Medium",
                "action": "Send promotional email and loyalty rewards."
            }

        else:
            return {
                "risk": "Low",
                "action": "Monitor customer engagement."
            }

    return {
        "risk": "Very Low",
        "action": "No immediate action required."
    }