def generate_response(user_message, latest_detection, latest_summary):
    """Generate a rule-based response from detections and summary context."""
    user_message = user_message.lower().strip()

    if user_message in ["help", "/help"]:
        return (
            "I can provide information about the current safety situation. You can ask me about:\n"
            "• Number of people detected (e.g., 'How many people are there?')\n"
            "• PPE compliance (e.g., 'What's the hardhat compliance rate?')\n"
            "• Overall safety summary (e.g., 'Give me a safety summary')\n"
            "• Current recommendations (e.g., 'What actions should we take?')\n"
            "• Specific PPE counts (e.g., 'How many people are wearing safety vests?')"
        )

    elif any(word in user_message for word in ["people", "person", "workers"]):
        people_count = latest_detection.get("Person", 0)
        return f"Currently, there are {people_count} {'person' if people_count == 1 else 'people'} detected in the monitored area."

    elif "hardhat" in user_message or "hard hat" in user_message:
        hardhat_count = latest_detection.get("Hardhat", 0)
        no_hardhat_count = latest_detection.get("NO-Hardhat", 0)
        total = hardhat_count + no_hardhat_count
        if total > 0:
            compliance_rate = (hardhat_count / total) * 100
            return (
                f"Hardhat status:\n"
                f"• Wearing hardhats: {hardhat_count}\n"
                f"• Not wearing hardhats: {no_hardhat_count}\n"
                f"• Compliance rate: {compliance_rate:.2f}%"
            )
        else:
            return "No hardhat data is available from the latest detection."

    elif "safety vest" in user_message or "vest" in user_message:
        vest_count = latest_detection.get("Safety Vest", 0)
        no_vest_count = latest_detection.get("NO-Safety Vest", 0)
        total = vest_count + no_vest_count
        if total > 0:
            compliance_rate = (vest_count / total) * 100
            return (
                f"Safety vest status:\n"
                f"• Wearing safety vests: {vest_count}\n"
                f"• Not wearing safety vests: {no_vest_count}\n"
                f"• Compliance rate: {compliance_rate:.2f}%"
            )
        else:
            return "No safety vest data is available from the latest detection."

    elif any(word in user_message for word in ["summary", "overview", "situation"]):
        return f"Here's the current safety situation:\n\n{latest_summary}"

    elif any(
        word in user_message
        for word in ["recommendation", "action", "what should we do"]
    ):
        if "Recommendations:" in latest_summary:
            recommendations = latest_summary.split("Recommendations:")[-1].strip()
            return f"Based on the current situation, here are the recommendations:\n{recommendations}"
        else:
            return "I'm sorry, but I don't have any specific recommendations available at the moment."

    elif "compliance" in user_message:
        hardhat_compliance = (
            latest_detection.get("Hardhat", 0)
            / (
                latest_detection.get("Hardhat", 0)
                + latest_detection.get("NO-Hardhat", 0)
            )
            * 100
            if (
                latest_detection.get("Hardhat", 0)
                + latest_detection.get("NO-Hardhat", 0)
            )
            > 0
            else 0
        )
        vest_compliance = (
            latest_detection.get("Safety Vest", 0)
            / (
                latest_detection.get("Safety Vest", 0)
                + latest_detection.get("NO-Safety Vest", 0)
            )
            * 100
            if (
                latest_detection.get("Safety Vest", 0)
                + latest_detection.get("NO-Safety Vest", 0)
            )
            > 0
            else 0
        )
        overall_compliance = (hardhat_compliance + vest_compliance) / 2
        return (
            f"Current PPE compliance rates:\n"
            f"• Hardhat compliance: {hardhat_compliance:.2f}%\n"
            f"• Safety vest compliance: {vest_compliance:.2f}%\n"
            f"• Overall PPE compliance: {overall_compliance:.2f}%"
        )

    else:
        return (
            "I'm not sure how to answer that. You can ask me about the number of people detected, "
            "PPE compliance rates, overall safety summary, or current recommendations. "
            "Type 'help' for more information on what I can do."
        )
