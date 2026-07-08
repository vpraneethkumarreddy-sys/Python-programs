import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = pd.DataFrame({
    "label": [0, 1, 0, 1, 0, 1],
    "message": [
        "Hi, how are you?",
        "Congratulations! You won a prize. Claim now!",
        "Let's meet tomorrow.",
        "Free gift waiting for you. Click here.",
        "See you at the office.",
        "You have won a free iPhone."
    ]
})

# Features and target
X = data["message"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numerical features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test with custom email
email = ["Congratulations! You have won a free iPhone. Click here to claim."]
email_vec = vectorizer.transform(email)

if model.predict(email_vec)[0] == 1:
    print("Spam Email")
else:
    print("Not Spam")
