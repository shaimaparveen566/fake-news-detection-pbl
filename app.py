import pandas as pd
import numpy as np
import cv2
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# Dummy Dataset (For Trial)
data = {
    "text": [
        "Breaking shocking election fraud exposed",
        "Government launches new education policy",
        "Celebrity scandal viral with edited image",
        "Scientists confirm new discovery"
    ],
    "image": [
        "img1.jpg",
        "img2.jpg",
        "img3.jpg",
        "img4.jpg"
    ],
    "label": [
        "FAKE",
        "REAL",
        "FAKE",
        "REAL"
    ]
}

df = pd.DataFrame(data)

df['label'] = df['label'].map({'REAL': 1, 'FAKE': 0})

# Text Feature Extraction
tfidf = TfidfVectorizer(stop_words='english')
X_text = tfidf.fit_transform(df['text'])

# Image Feature Extraction
def extract_image_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return [0, 0, 0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    height, width = gray.shape
    resolution = height * width
    contrast = np.std(gray)

    return [brightness, resolution, contrast]

image_features = []

for img_name in df['image']:
    img_path = os.path.join("images", img_name)
    features = extract_image_features(img_path)
    image_features.append(features)

X_image = np.array(image_features)


# Combine Features
X_combined = hstack([X_text, X_image])
y = df['label']


# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
