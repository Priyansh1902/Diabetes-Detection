from flask import Flask, render_template, request
import joblib
import numpy as np


# Load models
knn = joblib.load(f'C:/Users/priya/OneDrive/Desktop/Diabetes/ML Project/knn_model.joblib')
svm = joblib.load(f'C:/Users/priya/OneDrive/Desktop/Diabetes/ML Project/svm_model.joblib')
dtree = joblib.load(f'C:/Users/priya/OneDrive/Desktop/Diabetes/ML Project/decision_tree_model.joblib')
rforest = joblib.load(f'C:/Users/priya/OneDrive/Desktop/Diabetes/ML Project/random_forest_model.joblib')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/About')
def about():
    return render_template('About.html')


@app.route('/predict', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Process the form data here
        try:
            age = float(request.form['age'])
            gender = float(request.form['gender'])  # Assuming gender should be converted to a numeric value
            bmi = float(request.form['bmi'])
            blood_pressure = float(request.form['bloodPressure'])
            fbs = float(request.form['fbs'])
            hba1c = float(request.form['hba1c'])
            family_history = float(request.form['familyHistory'])
            smoking = float(request.form['smoking'])
            diet = float(request.form['diet'])
            exercise = float(request.form['exercise'])
            ml_model = request.form['mlModel']

            features = np.array([[age, gender, bmi, blood_pressure, fbs, hba1c, family_history, smoking, diet, exercise]])

            result = None
            

            if ml_model == "SVM":
                result = svm.predict(features)
            elif ml_model == "KNN":
                result = knn.predict(features)
            elif ml_model == "DecisionTree":
                result = dtree.predict(features)
            elif ml_model == "RandomForest":
                result = rforest.predict(features)

            # Process the result as needed
            return render_template('Result.html', prediction = result[0])

        except ValueError as e:
            # Handle the ValueError (e.g., invalid input data)
            error_message = str(e) 
            return render_template('Result.html', prediction = "Sorry" + error_message)

if __name__ == '__main__':
    app.run(debug=True)


