import requests
import json

def emotion_detector(text_to_analyze):
    """
    Detects emotions in the given text using the Watson NLP Emotion Predict function
    and formats the output to include specific emotions and the dominant emotion.
    Incorporates error handling for blank entries (status_code 400).

    Args:
        text_to_analyze (str): The text to be analyzed for emotions.

    Returns:
        dict or None: A dictionary containing the scores for anger, disgust, fear, joy, sadness,
                      and the dominant emotion, or None if a general error occurs.
                      If status_code is 400, returns a dictionary with all None values.
    """
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    input_json = { "raw_document": { "text": text_to_analyze } }

    # Initialize scores to None for the 400 status code case
    anger_score = None
    disgust_score = None
    fear_score = None
    joy_score = None
    sadness_score = None
    dominant_emotion = None

    try:
        response = requests.post(url, headers=headers, json=input_json)

        # Check for status code 400 specifically for blank entries
        if response.status_code == 400:
            # Return a dictionary with None values as requested for status_code 400
            return {
                'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': None
            }

        # For other non-200 status codes, raise an HTTPError
        response.raise_for_status()

        response_json = response.json()

        # Proceed with normal emotion extraction if status code is 200 (OK)
        if 'emotionPredictions' in response_json and response_json['emotionPredictions']:
            if 'emotions' in response_json['emotionPredictions'][0]:
                emotions_data = response_json['emotionPredictions'][0]['emotions']
                
                anger_score = emotions_data.get('anger', 0.0)
                disgust_score = emotions_data.get('disgust', 0.0)
                fear_score = emotions_data.get('fear', 0.0)
                joy_score = emotions_data.get('joy', 0.0)
                sadness_score = emotions_data.get('sadness', 0.0)

                emotion_scores = {
                    'anger': anger_score,
                    'disgust': disgust_score,
                    'fear': fear_score,
                    'joy': joy_score,
                    'sadness': sadness_score
                }

                non_zero_scores = {k: v for k, v in emotion_scores.items() if v is not None and v > 0}
                if non_zero_scores:
                    dominant_emotion = max(non_zero_scores, key=non_zero_scores.get)
                else:
                    dominant_emotion = "No dominant emotion (all scores zero or None)"
            else:
                # Fallback for other structures (e.g., list of individual emotion objects)
                max_score = -1.0
                for prediction in response_json['emotionPredictions']:
                    emotion_name = prediction.get('emotion')
                    confidence = prediction.get('confidence', 0.0)

                    if emotion_name == 'anger':
                        anger_score = confidence
                    elif emotion_name == 'disgust':
                        disgust_score = confidence
                    elif emotion_name == 'fear':
                        fear_score = confidence
                    elif emotion_name == 'joy':
                        joy_score = confidence
                    elif emotion_name == 'sadness':
                        sadness_score = confidence

                    if confidence > max_score and emotion_name in ['anger', 'disgust', 'fear', 'joy', 'sadness']:
                        max_score = confidence
                        dominant_emotion = emotion_name
        
        return {
            'anger': anger_score,
            'disgust': disgust_score,
            'fear': fear_score,
            'joy': joy_score,
            'sadness': sadness_score,
            'dominant_emotion': dominant_emotion
        }

    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        return None # Return None for general request errors
    except json.JSONDecodeError:
        print("Error decoding JSON response.")
        return None # Return None for JSON decoding errors
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None # Return None for any other unexpected errors