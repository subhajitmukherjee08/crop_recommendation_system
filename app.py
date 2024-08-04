from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the saved classification model
model = pickle.load(open('classifier_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Use the model to make crop recommendation
        prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Decode the predicted crop
        crop_decode = {
            1: 'RICE', 2: 'MAIZE', 3: 'JUTE', 4: 'COTTON', 5: 'COCONUT', 6: 'PAPAYA',
            7: 'ORANGE', 8: 'APPLE', 9: 'MUSKMELON', 10: 'WATERMELON', 11: 'GRAPES',
            12: 'MANGO', 13: 'BANANA', 14: 'POMEGRANATE', 15: 'LENTIL', 16: 'BLACKGRAM',
            17: 'MUNGBEAN', 18: 'MOTHBEANS', 19: 'PIGEONPEAS', 20: 'KIDNEYBEANS',
            21: 'CHICKPEA', 22: 'COFFEE'
        }

        recommended_crop = crop_decode[prediction[0]]
        
        # Return the predicted crop name as JSON response
        return jsonify({'crop': recommended_crop})

if __name__ == '__main__':
    app.run(debug=True)
