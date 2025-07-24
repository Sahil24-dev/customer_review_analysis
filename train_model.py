import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from utils import preprocess_text


df = pd.read_csv("sample_reviews.csv")
df['cleaned'] = df['review'].apply(preprocess_text)

X = df['cleaned']
y = df['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))


joblib.dump(pipeline, 'sentiment_model.pkl')
print("Model saved as sentiment_model.pkl")
