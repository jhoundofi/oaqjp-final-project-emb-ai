"""
This module implements a Flask web application for emotion detection.

It serves an HTML interface and provides an API endpoint to analyze text
for emotions using the Watson NLP library.
"""

from flask import Flask, render_template, request

# Import the emotion_detector function from your package
# pylint: disable=E0401, C0413
# E0401: This can sometimes occur if Pylint doesn't correctly
# understand the package path in a non-installed state.
# C0413: Placed for local import, typically after standard library.
from EmotionDetection.emotion_detection import emotion_detector


app = Flask("Emotion Detector")

@app.route("/emotionDetector")
def emotion_detector_route():
    """
    Handles the /emotionDetector route.

    Retrieves 'textToAnalyze' from request arguments, calls the
    emotion_detector function, and formats the output as a string.
    Includes error handling for blank or invalid text input.

    Returns:
        str: A formatted string displaying emotion analysis results
             or an error message for invalid input.
    """
    text_to_analyze = request.args.get('textToAnalyze')
    
    response_data = emotion_detector(text_to_analyze)

    # Check for general errors or specific blank entry handling from emotion_detector
    if response_data is None or response_data.get('dominant_emotion') is None:
        return "Invalid text! Please try again!"

    # Extract individual emotion scores. Using .get() for robustness.
    anger = response_data.get('anger', 0.0)
    disgust = response_data.get('disgust', 0.0)
    fear = response_data.get('fear', 0.0)
    joy = response_data.get('joy', 0.0)
    sadness = response_data.get('sadness', 0.0)
    dominant_emotion = response_data.get('dominant_emotion', 'None')

    # Format the output string as requested
    formatted_output = (
        f"For the given statement, the system response is 'anger': {anger}, "
        f"'disgust': {disgust}, 'fear': {fear}, 'joy': {joy} and 'sadness': {sadness}. "
        f"The dominant emotion is {dominant_emotion}."
    )
    
    return formatted_output

@app.route("/")
def render_index_page():
    """
    Renders the main index.html page for the web application.

    Returns:
        str: The rendered HTML content of index.html.
    """
    return render_template('index.html')

if __name__ == "__main__":
    # The host '0.0.0.0' makes the server accessible from any IP address
    # on the network, which is useful in containerized environments.
    # For local development, '127.0.0.1' or 'localhost' also works.
    app.run(host="0.0.0.0", port=5000)