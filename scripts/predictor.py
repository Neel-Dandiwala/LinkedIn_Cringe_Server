from .preprocessor import Preprocessor
import pickle
import numpy as np
from .extractor import Extractor
class Predictor:
    def __init__(self, model_path="models/model.pkl"):

        self.extractor = Extractor()
        self.preprocessor = Preprocessor()

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.lgb_model = model_data["model"]
            self.feature_columns = model_data["feature_columns"]

    def predict(self, text):
        """Predict if the text is cringe"""

        traditional_features = self.preprocessor.extract_traditional_features(text)
        bert_features = self.extractor.extract_bert_features(text)

        combined_features = np.concatenate([bert_features, [traditional_features[col] for col in self.feature_columns]])

        cringe_score = self.lgb_model.predict([combined_features])[0]

        return cringe_score

def main():
    predictor = Predictor()
    sample_text = """🚀 Excited to announce that I've been nominated for the "Most Influential Thought Leader in Synergy Innovation" award! 
    Feeling blessed and humbled by this recognition. 
    Remember: success is not just about crushing it 24/7, it's about leveraging your inner ninja to disrupt the paradigm! 
    #Blessed #Hustlelife #Leadership"""

    sample_text_2 = """I am a human. I can sing songs for you."""

    try: 
        score = predictor.predict(sample_text)
        normalized_score = (score) * 100
        if normalized_score < 20:
            rating = "Not cringe"
        elif normalized_score < 40:
            rating = "Somewhat cringe"
        elif normalized_score < 60:
            rating = "Cringe"
        elif normalized_score < 80:
            rating = "Very cringe"
        else:
            rating = "Extremely cringe"
        print(f"Cringe score: {normalized_score}/100 ({rating})")
    except Exception as e:
        print(f"Error predicting cringe score: {e}")

    try: 
        score = predictor.predict(sample_text_2)
        normalized_score = (score) * 100
        print(f"Cringe score: {normalized_score}/100")
    except Exception as e:
        print(f"Error predicting cringe score: {e}")

if __name__ == "__main__":
    main()

