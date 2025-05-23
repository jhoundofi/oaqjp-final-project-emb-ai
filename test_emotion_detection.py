import unittest
# Import the emotion_detector function from your package
from EmotionDetection.emotion_detection import emotion_detector

class TestEmotionDetection(unittest.TestCase):
    def test_joy(self):
        text = "I am glad this happened"
        result = emotion_detector(text)
        self.assertIsNotNone(result, "API call should return a result")
        self.assertIn('dominant_emotion', result, "Result should contain dominant_emotion")
        self.assertEqual(result['dominant_emotion'], 'joy', f"Expected 'joy', but got {result['dominant_emotion']}")

    def test_anger(self):
        text = "I am really mad about this"
        result = emotion_detector(text)
        self.assertIsNotNone(result, "API call should return a result")
        self.assertIn('dominant_emotion', result, "Result should contain dominant_emotion")
        self.assertEqual(result['dominant_emotion'], 'anger', f"Expected 'anger', but got {result['dominant_emotion']}")

    def test_disgust(self):
        text = "I feel disgusted just hearing about this"
        result = emotion_detector(text)
        self.assertIsNotNone(result, "API call should return a result")
        self.assertIn('dominant_emotion', result, "Result should contain dominant_emotion")
        self.assertEqual(result['dominant_emotion'], 'disgust', f"Expected 'disgust', but got {result['dominant_emotion']}")

    def test_sadness(self):
        text = "I am so sad about this"
        result = emotion_detector(text)
        self.assertIsNotNone(result, "API call should return a result")
        self.assertIn('dominant_emotion', result, "Result should contain dominant_emotion")
        self.assertEqual(result['dominant_emotion'], 'sadness', f"Expected 'sadness', but got {result['dominant_emotion']}")

    def test_fear(self):
        text = "I am really afraid that this will happen"
        result = emotion_detector(text)
        self.assertIsNotNone(result, "API call should return a result")
        self.assertIn('dominant_emotion', result, "Result should contain dominant_emotion")
        self.assertEqual(result['dominant_emotion'], 'fear', f"Expected 'fear', but got {result['dominant_emotion']}")

if __name__ == '__main__':
    unittest.main()