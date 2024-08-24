from flask import Flask, request, render_template, redirect, url_for
import cv2
import nltk
from transformers import pipeline
import pytesseract
import os

app = Flask(__name__)

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Update this path if necessary

# Summarization model
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

# NER model
ner = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')

# Download the Punkt tokenizer
nltk.download('punkt')

def imageToText(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        return 'No image found'
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract to extract text from the image
    text = pytesseract.image_to_string(gray)
    
    return text

import re

def extract_entities(text):
    # Regular expressions to find title and date
    title_regex = r'^[A-Z][a-z]+(?: [A-Z][a-z]+)*'
    date_regex = r'\b(?:\d{1,2} [A-Z][a-z]+ \d{4}|\d{4})\b'
    
    # Find title
    title_match = re.search(title_regex, text)
    title = title_match.group(0) if title_match else "Title not found"
    
    # Find date
    date_match = re.search(date_regex, text)
    date = date_match.group(0) if date_match else "Date not found"
    
    return title, date

def process_captured_frame(image_path):
    text = imageToText(image_path)

    # Extract entities
    title, date = extract_entities(text)

    # Text summarization
    summary = summarizer(text, max_new_tokens=200, min_new_tokens=130, do_sample=False)[0]['summary_text']

    return {
        'title': title,
        'date': date,
        'summary': summary
    }

def process_captured_frame(image_path):

    # Testing 
    image_path = 'uploads/sample2.jpg'

    text = imageToText(image_path)

    # Extract entities
    title, date = extract_entities(text)

    # Text summarization
    summary = summarizer(text, max_new_tokens=200, min_new_tokens=130, do_sample=False)[0]['summary_text']

    return {
        'title': title,
        'date': date,
        'summary': summary
    }

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # file_path = os.path.join('uploads', file.filename)
            file_path = os.path.join('uploads', 'image.jpg')

            file.save(file_path)
            summary = process_captured_frame(file_path)
            return render_template('result.html', summary=summary)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)