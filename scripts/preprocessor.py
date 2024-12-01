import numpy as np
import pandas as pd
from textblob import TextBlob
from bs4 import BeautifulSoup
import re
import emoji
import unicodedata
import spacy
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    def __init__(self):
        self.corporate_buzzwords = set([ 'b2b sales',
            'synergy', 'leverage', 'saas', 'paradigm', 'disrupt', 'innovation',
            'thought leadership', 'bandwidth', 'circle back', 'deep dive',
            'rockstar', 'ninja', 'guru', 'crushing it', 'hustle', 'grind',
            'game changer', 'thought leader', 'influencer', 'visionary',
            'revolutionary', 'disruptive', 'innovative', 'cutting-edge',
            'best in class', 'world class', 'next level', 'paradigm shift',
            'blockchain', 'ai', 'machine learning', 'crypto', 'web3',
            'scale', 'scalable', 'mission critical', 'value add', 'roi',
            'growth hack', 'unicorn', 'rockstar', '10x', 'crushing it'
        ])
        
        self.humble_brag_phrases = set([
            'god', 'humble', 'blessed', 'honored', 'grateful', 'thankful',
            'privileged', 'excited to announce', 'proud to share', 'exciting news',
            'humbled', 'blessed to', 'grateful for', 'thankful for', 'honored to',
            'proud to', 'thrilled to', 'excited to', 'fortunate to', 'lucky to',
            'amazing opportunity', 'incredible journey', 'dream come true',
            'milestone', 'achievement', 'success story', 'journey'
        ])

        self.self_promotion_phrases = set([
            'thought leader', 'expert', 'guru', 'visionary', 'influencer',
            'entrepreneur', 'founder', 'ceo', 'leader', 'pioneer',
            'innovator', 'specialist', 'authority', 'professional',
            'consultant', 'advisor', 'mentor', 'coach'
        ])

        self.motivation_words = set(['success', 'achieve', 'accomplish', 'succeed', 'overcome', 'thrive', 'flourish', 'excel', 'surpass', 'transcend'])

        self.call_to_action_patterns = set(['like' , 'follow', 'connect', 'message', 'learn more', 'click', 'visit', 'check out', 'explore', 'discover', 'get in touch', 'get started', 'join', 'sign up', 'start', 'begin', 'get involved', 'engage', 'participate', 'support', 'donate', 'contribute', 'participate', 'engage', 'support', 'agree'])

        self.viral_patterns = set(['thread', 'must read', 'dont forget to', 'make sure to', 'remember to'])

        self.pattern = {
            'hashtag_spam': re.compile(r'#\w+'),
            'exclamation_spam': re.compile(r'!'),
            'question_spam': re.compile(r'\?'),
            'caps_spam': re.compile(r'\b[A-Z]{2,}\b'),
            'numerical_list': re.compile(r'\d+\.\s'),
            'punctuation_spam': re.compile(r'[^\w\s]'),
            'money_reference': re.compile(r'\b\d+(?:\.\d{1,2})?\b'),
        }

        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])

    def clean_text(self, text):
        """Clean the text by removing emojis and URLs"""
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = unicodedata.normalize('NFKD', text)
        emojis = [c for c in text if c in emoji.EMOJI_DATA]
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # URLs
        text = re.sub(r'\S+@\S+', '', text) # Emails
        text = re.sub(r'[^\x00-\x7F]+', '', text) # Non-ASCII
        doc = self.nlp(text)

        cleaned_tokens = []
        for token in doc:
            if token.like_num or token.text in ['.', '!', '?']:
                cleaned_tokens.append(token.text)
            elif not token.is_punct and not token.is_space:
                cleaned_tokens.append(token.text.lower())

        text = ' '.join(cleaned_tokens)

        ui_elements = ['Following', 'Connect', '• \d+', 'comments?', 'likes?', 'Repost', 'Send', 'View profile']
        for element in ui_elements:
            text = re.sub(element, '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\b[a-zA-Z]\b', '', text)
        text = str(TextBlob(text).correct())
        text = text + ' ' + ' '.join(emojis)
        text = ' '.join(text.split())

        return text

    def extract_traditional_features(self, text):
        doc = self.nlp(text)
        blob = TextBlob(text)
        features = {
            'text': text,
            'length': len(text),
            'emoji_count': len([c for c in text if emoji.is_emoji(c)]),
            'hashtag_count': len(self.pattern['hashtag_spam'].findall(text)),
            'exclamation_count': len(self.pattern['exclamation_spam'].findall(text)),
            'question_count': len(self.pattern['question_spam'].findall(text)),
            'capitalized_word_count': len([word for word in text.split() if word[0].isupper()]),
            'sentence_count': len(list(doc.sents)),
            'word_count': len(text.split()),
            'avg_word_length': sum(len(token.text) for token in doc if token.is_alpha) / len([token for token in doc if token.is_alpha]),
            'verb_count': len([token for token in doc if token.pos_ == 'VERB']),
            'noun_count': len([token for token in doc if token.pos_ == 'NOUN']),
            'adjective_count': len([token for token in doc if token.pos_ == 'ADJ']),
            'entity_count': len(doc.ents),
            'has_numerical_list': 1 if self.pattern['numerical_list'].search(text) else 0,
            'self_promotion_count': sum(1 for phrase in self.self_promotion_phrases if phrase in text.lower()),
            'buzzword_count': sum(1 for word in text.lower().split() if word in self.corporate_buzzwords),
            'buzzword_ratio': sum(1 for word in text.lower().split() if word in self.corporate_buzzwords) / max(1, len(text.split())),
            'humble_brag_count': sum(1 for phrase in self.humble_brag_phrases if phrase in text.lower()),
            'personal_pronouns': len([token for token in doc if token.pos_ == 'PRON' and token.tag_ == 'PRP']),
            'motivation_score': sum(1 for word in text.lower().split() if word in self.motivation_words),
            'excessive_punctuation': len(self.pattern['punctuation_spam'].findall(text)),
            'has_call_to_action': sum(1 for word in text.lower().split() if word in self.call_to_action_patterns),
            'has_viral_pattern': sum(1 for word in text.lower().split() if word in self.viral_patterns),
            'starts_with_number': 1 if re.match(r'^\d', text.strip()) else 0,
            'contains_money_reference': 1 if self.pattern['money_reference'].search(text) else 0,
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity,
            'is_extemely_positive': blob.sentiment.polarity > 0.8,
            'is_extremely_negative': blob.sentiment.polarity < -0.8,
        }
        return features

    def calculate_cringe_score(self, row):
        """Calculate the cringe score for a given text"""
        features = self.extract_traditional_features(row['text'])

        content_score = (
            features['buzzword_ratio'] * 0.3 +
            features['humble_brag_count'] * 0.25 +
            features['self_promotion_count'] * 0.25 +
            features['motivation_score'] * 0.2         
        ) * 0.5

        engagement_score = (
            features['has_viral_pattern'] * 0.4 +
            features['has_call_to_action'] * 0.3 +
            (features['exclamation_count'] + features['question_count']) * 0.2 +
            features['hashtag_count'] * 0.4 +
            features['emoji_count'] * 0.2
        ) * 0.3

        social_score = min(1.0, (np.log1p(row['score']) / 10) * 0.6 + 
                         (np.log1p(row['num_comments']) / 5) * 0.4) * 0.2

        controversy_factor = 1 + ((1 - row['upvote_ratio']) * 0.5)
        raw_score =  (content_score + engagement_score + social_score) * controversy_factor

        return min(1.0, raw_score)

    def preprocess_dataset(self, csv_path):
        """Preprocess the entire dataset"""
        df = pd.read_csv(csv_path)
        
        features_list = []
        for i, text in enumerate(df['text']):
            print(f"Processing post {i+1} of {len(df)}")
            cleaned_text = self.clean_text(text)
            features = self.extract_traditional_features(cleaned_text)
            features_list.append(features)
            
        features_df = pd.DataFrame(features_list)
        
        features_df['cringe_score'] = df.apply(self.calculate_cringe_score, axis=1)
        scaler = MinMaxScaler()
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])
        
        features_df = features_df.fillna(0)
        
        features_df.to_csv('data/processed_features.csv', index=False)
        print(f"Processed {len(features_df)} posts!")
        
        return features_df

def main():
    preprocessor = Preprocessor()
    features_df = preprocessor.preprocess_dataset('data/linkedin_posts.csv')
    
    print("\nFeature statistics:")
    print(features_df.describe())
    
    print("\nFeature correlations with cringe score:")
    #correlations = features_df.corr()['cringe_score'].sort_values(ascending=False)
    #print(correlations)


if __name__ == "__main__":
    main()