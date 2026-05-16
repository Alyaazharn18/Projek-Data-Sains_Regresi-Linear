from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Membuat aplikasi Flask
app = Flask(__name__)

# Load model machine learning
model = joblib.load('model_mobil.pkl')

# Route utama
@app.route('/')
def home():
    return render_template('index.html')

# Route prediksi
@app.route('/predict', methods=['POST'])
def predict():

    # Ambil data JSON dari request
    data = request.get_json()

    # Ambil fitur input
    horsepower = data['Horsepower']
    engine_size = data['Engine_size']
    fuel_efficiency = data['Fuel_efficiency']
    wheelbase = data['Wheelbase']

    # Ubah ke bentuk array
    features = np.array([
        [horsepower, engine_size, fuel_efficiency, wheelbase]
    ])

    # Prediksi harga
    prediction = model.predict(features)

    # Return hasil prediksi
    return jsonify({
        'predicted_price_in_thousands': round(prediction[0], 2)
    })

# Menjalankan server Flask
if __name__ == '__main__':
    app.run(debug=True)