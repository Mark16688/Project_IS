from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os


app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # บังคับให้ TensorFlow ใช้ CPU

model = joblib.load("Dataset/models_complete/titanic_model.pkl")  # Log แสดงว่าระบบโหลดโมเดลสำเร็จ

# โหลดโมเดล Neural Network
nn_model = load_model("Dataset/models_complete/mnist_model.h5")

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
    
    
@app.route("/predict_pickle", methods=["POST"]) ##อันใหม่
def predict_pickle():
    try:
        # ✅ โหลดโมเดลเฉพาะตอน Request เพื่อประหยัด Memory
        model = pickle.load(open("Dataset/models_complete/model.pkl", "rb"))

        # ✅ รับค่าจากฟอร์ม
        data = [float(x) for x in request.form.values()]

        # ✅ ทำการพยากรณ์
        prediction = model.predict([data])

        # ✅ คืนค่าผลลัพธ์ไปยังหน้าเว็บ
        return render_template("ml_demo.html", prediction=prediction[0])

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # ใช้พอร์ตจาก Railway
    app.run(host='0.0.0.0', port=port, debug=True)