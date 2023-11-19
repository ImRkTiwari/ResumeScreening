# app.py
from flask import Flask, render_template, request
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Loading Models
KNClf = pickle.load(open('KNClf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Map category ID to category name
category_mapping = {
    0: "Advocate",
    1: "Arts",
    2: "Automation Testing",
    3: "Blockchain",
    4: "Business Analyst",
    5: "Civil Engineer",
    6: "Data Science",
    7: "Database",
    8: "DevOps Engineer",
    9: "DotNet Developer",
    10: "ETL Developer",
    11: "Electrical Engineering",
    12: "HR",
    13: "Hadoop",
    14: "Health and Fitness",
    15: "Java Developer",
    16: "Mechanical Engineer",
    17: "Nework Security Engineer",
    18: "Operation Manager",
    19: "PMO",
    20: "Python Developer",
    21: "SAP Developer",
    22: "Sales",
    23: "Testing",    
    24: "Web Designing"
}

# Function to clean the input resume


def clean_resume(txt):
    cleantxt = re.sub('http\S+\s', ' ', txt)
    cleantxt = re.sub('RT|cc', ' ', cleantxt)
    cleantxt = re.sub('#\S+\s', ' ', cleantxt)
    cleantxt = re.sub('@\S+', ' ', cleantxt)
    cleantxt = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', cleantxt)
    cleantxt = re.sub(r'[^\x00-\x7f]', ' ', cleantxt)
    cleantxt = re.sub('\s+', ' ', cleantxt)
    return cleantxt

# Flask route for the home page


@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# Flask route for handling resume submission and prediction


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        upload_file = request.files['file']
        if upload_file:
            try:
                resume_bytes = upload_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try decoding with 'latin-1'
                resume_text = resume_bytes.decode('latin-1')

            cleaned_resume = clean_resume(resume_text)
            input_features = tfidf.transform([cleaned_resume])
            prediction_id = KNClf.predict(input_features)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")
            return render_template('index.html', prediction=category_name)


#if __name__ == '__main__':
#    app.run(debug=True)
