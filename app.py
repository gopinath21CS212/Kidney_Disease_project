from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('Classification_Model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = [float(x) for x in request.form.values()]
    final_features = [np.array(input_data)]
    
    # Make a prediction using the loaded model
    prediction = model.predict(final_features)
    output = np.around(prediction)

    if output == 1:
        output = "Yes"
    elif output == 0:
        output = "No"

    # Pass the prediction value to the template
    return render_template('index.html', prediction_text='Kidney Disease: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
