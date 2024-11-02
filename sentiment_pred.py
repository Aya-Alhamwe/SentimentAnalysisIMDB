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
from fastapi.responses import HTMLResponse

# Load the dataset :)
numRows = 9000
data = pd.read_csv("IMDB Dataset.csv", header=0, nrows=numRows)


nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))


def clean_text(text):
    text = text.lower() 
    text = re.sub(r'<br />', '', text)  
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words]) 
    return text


data['clean_review'] = data['review'].apply(clean_text)
data.drop(columns='review', inplace=True)
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})


X = data['clean_review']
y = data['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


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

# Root endpoint that returns an HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis</title>
    <style>
        /* Styles for the HTML page */
        body {
            font-family: 'Arial', sans-serif;
            background: #696da7;
            margin: 0;
            padding: 20px;
            color: #e0e0e0;
        }

        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #00ffcc;
            font-size: 2.5em;
            text-shadow: 0 4px 10px rgba(0, 255, 204, 0.4);
        }

        .container {
            max-width: 600px;
            margin: auto;
            background: rgba(40, 44, 52, 0.9);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.2);
            text-align: center;
        }

        .movie-image {
            width: 100%;
            max-width: 400px;
            height: auto;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0, 102, 255, 0.5);
        }

        .button-group {
            display: flex;
            justify-content: space-around;
            margin-top: 15px;
        }

        button {
            padding: 10px 15px;
            background: linear-gradient(90deg, #00ffcc, #0066ff);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: bold;
            transition: background 0.3s, transform 0.2s, box-shadow 0.3s;
            box-shadow: 0 4px 12px rgba(0, 102, 255, 0.3);
        }

        button:hover {
            background: linear-gradient(90deg, #0066ff, #00ffcc);
            transform: scale(1.02);
            box-shadow: 0 6px 15px rgba(0, 102, 255, 0.5);
        }

        textarea {
            width: 100%;
            max-width: 100%;
            height: 150px;
            margin: 15px auto;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 12px;
            background: #2b2d42;
            color: #d4d4dc;
            font-size: 1.1em;
            transition: border-color 0.3s, box-shadow 0.3s;
            display: block;
        }

        .result {
            margin-top: 20px;
            padding: 20px;
            background: rgba(0, 102, 255, 0.08);
            border-radius: 8px;
            border-left: 4px solid #00ffcc;
            color: #a8e6cf;
            font-size: 1.1em;
            animation: fadeIn 0.6s ease-in-out;
        }

        .result strong {
            color: #ffffff;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }

        footer {
            text-align: center;
            margin-top: 30px;
            font-size: 0.9em;
            color: #81dfd6;
        }
    </style>
</head>

<body>

    <div class="container">
        <h1>Sentiment Analysis for Movies</h1>

        <img id="movie-image" class="movie-image" src="https://raw.githubusercontent.com/Aya-Alhamwe/SentimentAnalysisIMDB/main/p1.jpg" alt="Movie Poster">

        <div class="button-group">
            <button onclick="previousImage()">Previous Movie</button>
            <button onclick="nextImage()">Next Movie</button>
        </div>

        <textarea id="review" placeholder="Enter your movie review here..."></textarea>
        <button onclick="submitReview()">Analyze Sentiment</button>

        <div id="result" class="result" style="display:none;"></div>
    </div>

    <footer>
        <p>Sentiment Analysis Project (Aya Alhamwi)</p>
    </footer>

    <script>
        // Array of movie images
        const movieImages = [
            "https://raw.githubusercontent.com/Aya-Alhamwe/SentimentAnalysisIMDB/main/p1.jpg",
            "https://raw.githubusercontent.com/Aya-Alhamwe/SentimentAnalysisIMDB/main/p2.jpg",
            "https://raw.githubusercontent.com/Aya-Alhamwe/SentimentAnalysisIMDB/main/p3.jpg"
        ];

        let currentImageIndex = 0;


        function nextImage() {
            currentImageIndex = (currentImageIndex + 1) % movieImages.length;
            document.getElementById("movie-image").src = movieImages[currentImageIndex];
        }

       
        function previousImage() {
            currentImageIndex = (currentImageIndex - 1 + movieImages.length) % movieImages.length;
            document.getElementById("movie-image").src = movieImages[currentImageIndex];
        }

     
        async function submitReview() {
            const reviewText = document.getElementById("review").value;
            const resultDiv = document.getElementById("result");

            // Send POST request to the predict endpoint
           const response = await fetch("/predict/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text: reviewText })
            });

            const data = await response.json();

            // Set the sentiment color based on prediction
            let sentimentColor;
            if (data.sentiment === "Positive") {
                sentimentColor = "#4caf50"; // Green for positive
            } else {
                sentimentColor = "#ff5252"; // Red for negative
            }

            // Display the result
            resultDiv.style.display = "block";
            resultDiv.innerHTML = `<strong>Sentiment:</strong> <span style="color: ${sentimentColor};">${data.sentiment}</span>`; 
            <strong>Message:</strong> ${data.message}<br>
            <strong>Recommendation:</strong> ${data.recommendation}
        }
    </script>
</body>

</html>
    """


@app.post("/predict/")
async def predict(review: Review):
    
    cleaned_text = clean_text(review.text)
    vectorized_text = vectorizer.transform([cleaned_text])  
    prediction = model.predict(vectorized_text) 

  
    sentiment = "Positive" if prediction[0] == 1 else "Negative"

    
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
