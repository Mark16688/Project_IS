from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)


model = joblib.load("models/titanic_model.pkl")  # Log แสดงว่าระบบโหลดโมเดลสำเร็จ

# โหลดโมเดล Neural Network
nn_model = load_model("models/mnist_model.h5")

# ----------------- Routes สำหรับแสดงหน้าเว็บ -----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ml_research')
def ml_research():
    return render_template('ml_research.html')

@app.route('/ml_demo')
def ml_demo():
    return render_template('ml_demo.html')

@app.route('/nn_research')
def nn_research():
    return render_template('nn_research.html')

@app.route('/nn_demo')
def nn_demo():
    return render_template('nn_demo.html')

import pandas as pd

import pandas as pd

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ รับค่าจากแบบฟอร์ม
        pclass = int(request.form['Pclass'])
        sex = int(request.form['Sex'])
        age = float(request.form['Age'])
        sibsp = int(request.form['SibSp'])
        parch = int(request.form['Parch'])
        fare = float(request.form['Fare'])
        embarked = int(request.form['Embarked'])

        # ✅ ใช้ชื่อคอลัมน์เดียวกับตอน Train โมเดล
        feature_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        input_features = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], columns=feature_names)

        # ✅ ทำการพยากรณ์
        prediction = model.predict(input_features)[0]

        # ✅ คืนค่าผลลัพธ์ไปที่หน้าเว็บ
        result = "✅ Survived" if prediction == 1 else "❌ Did Not Survive"
        return render_template('ml_demo.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"
    
@app.route('/predict_nn', methods=['POST'])
def predict_nn():
    try:
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read())).convert('L')  # แปลงเป็นขาวดำ
        img = img.resize((28, 28))  # ปรับขนาด
        img_array = np.array(img) / 255.0  # Normalize
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape ให้ตรงกับโมเดล

        prediction = nn_model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        return jsonify({'prediction': int(predicted_digit)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)