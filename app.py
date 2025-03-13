from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from PIL import Image
import io
import os


app = Flask(__name__)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # บังคับให้ TensorFlow ใช้ CPU อย่างชัดเจน
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ปิด oneDNN เพื่อลด Warning

#model = joblib.load("models_complete/titanic_model.pkl")  # Log แสดงว่าระบบโหลดโมเดลสำเร็จ

# โหลดโมเดล Neural Network
#nn_model = load_model("models_complete/mnist_model.h5")

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


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ โหลดโมเดลเฉพาะตอนใช้งาน (ลด RAM)
        model = joblib.load("models_complete/titanic_model.pkl")

        # ✅ รับค่าจากแบบฟอร์ม
        pclass = int(request.form['Pclass'])
        sex = int(request.form['Sex'])
        age = float(request.form['Age'])
        sibsp = int(request.form['SibSp'])
        parch = int(request.form['Parch'])
        fare = float(request.form['Fare'])
        embarked = int(request.form['Embarked'])

        feature_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        input_features = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], columns=feature_names)

        prediction = model.predict(input_features)[0]

        # ✅ ลบโมเดลออกจากหน่วยความจำ
        del model

        result = "✅ Survived" if prediction == 1 else "❌ Did Not Survive"
        return render_template('ml_demo.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"
    
@app.route('/predict_nn', methods=['POST'])
def predict_nn():
    try:
        # ✅ โหลดโมเดลเฉพาะตอนใช้งาน (ลด RAM)
        nn_model = load_model("models_complete/mnist_model.h5")

        file = request.files['file']
        img = Image.open(io.BytesIO(file.read())).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = nn_model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        # ✅ ลบโมเดลออกจากหน่วยความจำ
        del nn_model
        tf.keras.backend.clear_session()  # ล้างกราฟ TensorFlow

        return jsonify({'prediction': int(predicted_digit)})

    except Exception as e:
        return jsonify({'error': str(e)})
    
    
# จำกัด RAM ไม่ให้ใช้เกินจำเป็น
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # เปิดให้ใช้แรมแบบ Dynamic
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]  # จำกัดให้ใช้ 1GB เท่านั้น
            )
    except RuntimeError as e:
        print(e)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # ใช้พอร์ต 5000 เป็นค่าเริ่มต้น
    app.run(host='0.0.0.0', port=port, debug=True)