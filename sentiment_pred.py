import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
import re
from fastapi import FastAPI    
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

#load data :) 
numRows = 60000
data = pd.read_csv("IMDB Dataset.csv", header=0, nrows=numRows)



nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    text = text.lower()  
    text = re.sub(r'<br />', '', text)  
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'[^\w\s]', '', text)  
    text = ' '.join([word for word in text.split() if word not in stop_words]) 
    return text

data['clean_review'] = data['review'].apply(clean_text)
data.drop(columns='review', inplace=True)
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

X = data['clean_review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Display the classification report
y_pred = model.predict(X_train_vec)
report = classification_report(y_train, y_pred, target_names=['Negative', 'Positive'])
print(report)

#***********************************FastAPI :)***************************************************

app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Review(BaseModel):
    text: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to Sentemint Analyisi movie :) !"}

@app.post("/predict/")
async def predict_sentiment(review: Review):
    
    review_vec = vectorizer.transform([review.text])
    prediction = model.predict(review_vec)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    
    # Customize messages based on sentiment
    if sentiment == "Positive":
        message = "Glad you enjoyed the movie! Thanks for sharing your positive thoughts."
        recommendation = "We recommend checking out similar feel-good movies!"
    else:
        message = "Thanks for your feedback! Even negative experiences can guide you to better films."
        recommendation = "Consider exploring different genres for a better experience!"
    
    return {
        "sentiment": sentiment,
        "message": message,
        "recommendation": recommendation
    }





