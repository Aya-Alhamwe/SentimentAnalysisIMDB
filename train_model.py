import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  
from sklearn.metrics import classification_report
import nltk
import re
from nltk.stem import WordNetLemmatizer

numRows = 9000
data = pd.read_csv("IMDB Dataset.csv", header=0, nrows=numRows)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(nltk.corpus.stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower() 
    text = re.sub(r'<br />', '', text)  
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  
    return text

data['clean_review'] = data['review'].apply(clean_text)
data.drop(columns='review', inplace=True) 
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})  

X = data['clean_review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english')  
X_train_vec = vectorizer.fit_transform(X_train)  
X_test_vec = vectorizer.transform(X_test) 

model = SVC(kernel='linear', class_weight='balanced')  
model.fit(X_train_vec, y_train)

y_pred_train = model.predict(X_train_vec)
report_train = classification_report(y_train, y_pred_train, target_names=['Negative', 'Positive'])
print("Training Set Classification Report:\n", report_train)

y_pred_test = model.predict(X_test_vec)
report_test = classification_report(y_test, y_pred_test, target_names=['Negative', 'Positive'])
print("Test Set Classification Report:\n", report_test)

joblib.dump(model, 'train_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
