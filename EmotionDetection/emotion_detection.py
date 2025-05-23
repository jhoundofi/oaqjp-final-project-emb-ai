import requests
import json

def emotion_detector(text_to_analyze):
    """
    Detects emotions in the given text using the Watson NLP Emotion Predict function
    and formats the output to include specific emotions and the dominant emotion.

    Args:
        text_to_analyze (str): The text to be analyzed for emotions.

    Returns:
        dict or None: A dictionary containing the scores for anger, disgust, fear, joy, sadness,
                      and the dominant emotion, or None if an error occurs or no emotions are detected.
    """
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    input_json = { "raw_document": { "text": text_to_analyze } }

    # Initialize scores
    anger_score = 0.0
    disgust_score = 0.0
    fear_score = 0.0
    joy_score = 0.0
    sadness_score = 0.0
    dominant_emotion = None

    try:
        response = requests.post(url, headers=headers, json=input_json)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        response_json = response.json()

        # Check if 'emotionPredictions' key exists and is not empty
        if 'emotionPredictions' in response_json and response_json['emotionPredictions']:
            # The API response structure can vary. We need to extract the relevant scores.
            # Assuming 'emotionPredictions' contains a list of dictionaries, each with 'emotion' and 'confidence'.
            # And within each 'emotion' object, there might be 'text' and 'emotions' as nested objects.
            # Based on typical Watson NLP output, it often looks like:
            # {'emotionPredictions': [{'emotion': {'anger': x, 'disgust': y, ...}, 'text': '...'}]}
            
            # Let's handle the case where 'emotion' directly contains scores or is a list of individual emotion objects.
            
            # Scenario 1: 'emotion' key directly contains scores (more common for aggregated models)
            # Example: {'emotionPredictions': [{'emotion': {'anger': 0.1, 'disgust': 0.05, ...}, 'text': '...'}]}
            
            # Scenario 2: 'emotion' key contains a list of individual emotion predictions
            # Example: {'emotionPredictions': [{'emotion': 'joy', 'confidence': 0.9}, {'emotion': 'anger', 'confidence': 0.1}]}

            # We need to adapt based on the *exact* structure the endpoint returns.
            # Given the previous context, 'emotion' and 'confidence' were top-level within each prediction.
            # Let's assume the previous simulation was closer: `{'emotion': 'joy', 'confidence': 0.9997123}`
            
            max_score = -1.0
            
            # It's more likely that the aggregated workflow directly gives the scores for each emotion.
            # Let's simulate parsing a common structure for an aggregated model:
            # response_json = {
            #     "emotionPredictions": [
            #         {
            #             "span": {"begin": 0, "end": 28},
            #             "emotions": {
            #                 "anger": 0.005,
            #                 "disgust": 0.002,
            #                 "fear": 0.003,
            #                 "joy": 0.98,
            #                 "sadness": 0.001
            #             }
            #         }
            #     ]
            # }

            # Accessing the emotions from the first prediction's 'emotions' object (assuming a single span prediction)
            if 'emotions' in response_json['emotionPredictions'][0]:
                emotions_data = response_json['emotionPredictions'][0]['emotions']
                
                anger_score = emotions_data.get('anger', 0.0)
                disgust_score = emotions_data.get('disgust', 0.0)
                fear_score = emotions_data.get('fear', 0.0)
                joy_score = emotions_data.get('joy', 0.0)
                sadness_score = emotions_data.get('sadness', 0.0)

                # Determine dominant emotion
                emotion_scores = {
                    'anger': anger_score,
                    'disgust': disgust_score,
                    'fear': fear_score,
                    'joy': joy_score,
                    'sadness': sadness_score
                }

                if emotion_scores: # Ensure there are scores to compare
                    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            else: # Fallback if 'emotions' key is not directly present, e.g., if it's a list of individual emotion objects
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

                    if confidence > max_score:
                        max_score = confidence
                        dominant_emotion = emotion_name
        
        # Format the output as required
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
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON response.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == '__main__':
    # Example usage with the required output format
    text1 = "I love this new technology."
    emotions1 = emotion_detector(text1)
    if emotions1:
        print(f"Emotions for '{text1}': {emotions1}")

    text2 = "I am so sad to hear that."
    emotions2 = emotion_detector(text2)
    if emotions2:
        print(f"Emotions for '{text2}': {emotions2}")

    text3 = "This is a neutral statement." # May yield low scores across all
    emotions3 = emotion_detector(text3)
    if emotions3:
        print(f"Emotions for '{text3}': {emotions3}")

    text4 = "I am extremely angry about this situation."
    emotions4 = emotion_detector(text4)
    if emotions4:
        print(f"Emotions for '{text4}': {emotions4}")

    text5 = "I am absolutely terrified."
    emotions5 = emotion_detector(text5)
    if emotions5:
        print(f"Emotions for '{text5}': {emotions5}")

    text6 = "" # Test with empty string
    emotions6 = emotion_detector(text6)
    if emotions6:
        print(f"Emotions for '{text6}': {emotions6}")
    else:
        print(f"No emotion detected for empty string: {emotions6}")