import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from io import StringIO
from docx import Document
import PyPDF2

# Load pre-trained models and vectorizer
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
model = pickle.load(open('best_model.pkl', 'rb'))
le = LabelEncoder()

# Streamlit UI Setup
st.title("Resume Screening App")
st.sidebar.title("Upload Your Resume")

# Upload File Section
uploaded_file = st.sidebar.file_uploader("Choose a file (PDF, Word, Text)", type=["pdf", "docx", "txt"])

# Function to clean and preprocess the resume text
def clean_resume(text):
    cleantxt = re.sub('http\S+\s', ' ', text)
    cleantxt = re.sub('RT|cc', ' ', cleantxt)
    cleantxt = re.sub('#\S+\s', ' ', cleantxt)
    cleantxt = re.sub('@\S+', ' ', cleantxt)
    cleantxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', cleantxt)
    cleantxt = re.sub(r'[^\x00-\x7f]', ' ', cleantxt)
    cleantxt = re.sub('\s+', ' ', cleantxt)
    return cleantxt.strip()

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to extract text from a Word document
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ''
    for para in doc.paragraphs:
        text += para.text
    return text

# Function to extract text from a text file
def extract_text_from_txt(txt_file):
    return txt_file.getvalue().decode("utf-8")

# Process uploaded file
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
        text = ""

    # Clean the extracted resume text
    cleaned_text = clean_resume(text)

    if cleaned_text:
        # Show the cleaned text
        st.subheader("Resume Text")
        st.text_area("Resume Content", cleaned_text, height=300)

        # Vectorize the cleaned text using the loaded TF-IDF vectorizer
        text_vectorized = tfidf.transform([cleaned_text])

        # Predict the category of the resume using the trained model
        prediction = model.predict(text_vectorized)
        predicted_category = le.inverse_transform(prediction)[0]

        # Display the predicted category
        st.subheader("Predicted Category")
        st.write(predicted_category)

        # Classification report for more details
        st.subheader("Model Performance")
        y_test = np.array([predicted_category])
        y_pred = prediction
        st.text_area("Classification Report", classification_report(y_test, y_pred, target_names=[predicted_category]), height=300)

    else:
        st.error("No text extracted from the resume. Please check the file format and content.")
