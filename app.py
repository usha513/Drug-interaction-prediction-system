from flask import Flask, render_template, request, redirect, flash, url_for, session
from flaskext.mysql import MySQL
from flask_session import Session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from joblib import load
from flask import flash, redirect, url_for
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from flask import flash, redirect, url_for

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False

# MySQL Configuration
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'myproject'
app.config['MYSQL_DATABASE_HOST'] = '127.0.0.1'

mysql = MySQL(app)
Session(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.get_db().cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()

        if user:
            session['user_id'] = user[0]
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Please check your credentials.', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']

        cursor = mysql.get_db().cursor()
        cursor.execute("INSERT INTO users (username, password, email, phone, address) VALUES (%s, %s, %s, %s, %s)",
                       (username, password, email, phone, address))
        mysql.get_db().commit()
        flash('Registration successful. Please log in.', 'success')

    return render_template('register.html')

@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('home.html')
    else:
        flash('Please log in to access the home page.', 'warning')
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))


def load_dataset():
    dataset_path = 'Drug.csv'  # Replace with your actual dataset path
    try:
        df = pd.read_csv(dataset_path)
        return df
    except Exception as e:
        return str(e)

@app.route('/view_data')
def view_data():
    dataset = load_dataset()
    if isinstance(dataset, pd.DataFrame):
        data_html = dataset.to_html(classes='table table-bordered table-striped', escape=False)
    else:
        data_html = f"Error loading dataset: {dataset}"

    return render_template('home.html', data_html=data_html)

def preprocess_data(df):
    if df is not None:
        # Perform data preprocessing here (e.g., cleaning, feature engineering)
        # Replace this with your actual preprocessing code

        # For this example, let's just return the first 5 rows of the dataset with an additional calculated column
        df['condition'] = df['drugName'] * 2
        preprocessed_data = df.head()
        return preprocessed_data
    else:
        return None

# Create a route to load, preprocess, and display the dataset
@app.route('/preprocess', methods=['POST'])
def preprocess():
    # Load the dataset
    dataset = load_dataset()

    # Preprocess the dataset
    preprocessed_data = preprocess_data(dataset)

    if preprocessed_data is not None:
        # Store the preprocessed data in the session
        session['preprocessed_data'] = preprocessed_data.to_html(classes='table table-bordered table-striped', escape=False)
        flash('Data preprocessing successful!', 'success')
    else:
        flash('Error preprocessing dataset.', 'danger')

    return redirect(url_for('home'))

def load_and_preprocess_data():
    dataset_path = 'Drug.csv'  # Replace with your actual dataset path

    try:
        df = pd.read_csv(dataset_path)

        # Check for the required columns and preprocess them as needed
        required_columns = ['rating', 'condition', 'drugName']
        if not all(column in df.columns for column in required_columns):
            raise ValueError("One or more required columns are missing")

        # Example preprocessing: Convert 'rating' to numeric and handle missing values
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df.fillna({'rating': df['rating'].mean()}, inplace=True)

        # Add more preprocessing steps here as needed

        return df
    except Exception as e:
        return f"Error in load_and_preprocess_data: {e}"

from joblib import dump, load

def train_svm_model(X, y):
    model = SVC(kernel='linear')
    model.fit(X, y)
    return model

@app.route('/train_model', methods=['POST'])
def train_model_route():
    # Load and preprocess your dataset
    df = load_and_preprocess_data()

    if isinstance(df, pd.DataFrame):
        # Define features (X) and target variable (y)
        # Assuming 'rating' and 'condition' are the features, and 'drugName' is the target
        X = df[['rating', 'condition']].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = df['drugName']

        # Train an SVM model
        model = train_svm_model(X, y)

        # Save the trained model
        dump(model, 'svm_model.pkl')
        flash('Model trained and saved successfully!', 'success')
    else:
        flash('Error loading or preprocessing dataset.', 'danger')

    return redirect(url_for('home'))

import matplotlib.pyplot as plt
import os

@app.route('/apply_algorithm', methods=['POST'])
def apply_algorithm():
    try:
        model = load('svm_model.pkl')
        df = load_and_preprocess_data()
        X = df[['rating', 'condition']].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = df['drugName']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        accuracy = model.score(X_test, y_test)

        # Generate and save the graph as an image
        plt.figure()
        plt.bar(['Accuracy'], [accuracy])
        plt.ylim(0, 1)
        plt.savefig('static/accuracy_graph.png')  # Save in 'static' directory
        plt.close()

        flash(f'SVM model applied. Accuracy: {accuracy:.2f}', 'success')
    except Exception as e:
        flash(f'Error applying SVM model: {e}', 'danger')

    return redirect(url_for('home'))


# @app.route('/result', methods=['POST'])
# def result():
#     # Implement your algorithm application code here
#     flash('Result successfully!', 'success')
#     return redirect(url_for('home'))


# @app.route('/recommend_drugs', methods=['POST'])
# def recommend_drugs():
#     df = pd.read_csv('Drug.csv')
#     input_condition = request.form.get('condition')

#     # Perform a medical condition-based filter on the dataset
#     filtered_drugs = df[df['condition'] == input_condition]['drugName'].unique()

#     return render_template('result.html', recommended_drugs=filtered_drugs)

# ... [existing Flask setup and routes] ...

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df = pd.read_csv('Drug.csv')

# Assuming 'text_column' is the name of the column with text data
text_data = df['condition'].astype(str).tolist()  # Convert to list

# Create and fit the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(text_data)

# Save the trained vectorizer for later use
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
# Placeholder for your actual text-to-number conversion logic
def convert_text_to_numbers(text):
    # Load the trained TF-IDF vectorizer
    loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return loaded_vectorizer.transform([text])

def preprocess_condition(condition):
    # Convert the condition using the defined conversion method
    return convert_text_to_numbers(condition)

@app.route('/predict_and_recommend', methods=['POST'])
def predict_and_recommend():
    try:
        input_condition = request.form.get('condition')
        if not input_condition:
            flash('No condition provided', 'danger')
            return redirect(url_for('home'))

        # Load the trained model
        model = joblib.load('svm_model.pkl')

        # Load the dataset for drug recommendation
        df_drugs = pd.read_csv('Drug.csv')

        # Filter drugs based on the input condition
        # Make sure 'input_condition' matches how you're using it in the model
        recommended_drugs = df_drugs[df_drugs['medical_condition'] == input_condition][['condition', 'drugName']].drop_duplicates().values.tolist()

        return render_template('result.html', recommended_drugs=recommended_drugs,condition=input_condition)
    except FileNotFoundError as e:
        flash(f'File not found: {e}', 'danger')
    except Exception as e:
        flash(f'Error during prediction and recommendation: {e}', 'danger')

    return redirect(url_for('home'))



@app.route('/recommend_drugs', methods=['POST'])
def recommend_drugs():
    try:
        input_medical_condition = request.form.get('medical_condition')
        if not input_medical_condition:
            flash('No medical condition provided', 'danger')
            return redirect(url_for('home'))

        # Load the dataset
        df_drugs = pd.read_csv('Drug.csv')

        # Filter the DataFrame for the given medical condition and get unique drug names
        # recommended_drugs = df_drugs[df_drugs['medical_condition'] == input_medical_condition]['condition'].unique()
        

        recommended_drugs = df_drugs[df_drugs['medical_condition'] == input_medical_condition][['condition', 'drugName']].drop_duplicates().values.tolist()

        return render_template('home.html', recommended_drugs=recommended_drugs, condition=input_medical_condition)
    except FileNotFoundError as e:
        flash(f'File not found: {e}', 'danger')
    except Exception as e:
        flash(f'Error: {e}', 'danger')

    return redirect(url_for('home'))






import pandas as pd

@app.route('/predict_drug', methods=['POST'])
def predict_drug():
    input_medical_condition = request.form.get('medical_condition')
    if not input_medical_condition:
        flash('No medical condition provided', 'danger')
        return redirect(url_for('home'))

    try:
        df_drugs = pd.read_csv('Drug.csv')
        # Filter drugs based on Dose_mg being 1 or 2
        filtered_drugs = df_drugs[(df_drugs['Dose_mg'].isin([1, 2])) & (df_drugs['medical_condition'] == input_medical_condition)]

        if not filtered_drugs.empty:
            # Extracting medical conditions and doses
            drugs_info = filtered_drugs[['medical_condition', 'drugName','Dose_mg']].values.tolist()
            return render_template('drug_result.html', drugs_info=drugs_info, input_medical_condition=input_medical_condition)
        else:
            flash('No matching drugs found for the condition with specified doses', 'info')

    except Exception as e:
        flash(f'Error: {e}', 'danger')

    return redirect(url_for('home'))




if __name__ == '__main__':
    app.run(debug=True)