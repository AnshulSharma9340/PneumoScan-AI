from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import os
from grad_cam import get_grad_cam

app = Flask(__name__)
model = load_model("model/final_model.h5")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    heatmap_path, prob = get_grad_cam(file_path)

    label = "PNEUMONIA Detected" if prob > 0.5 else "NORMAL"

    return render_template("result.html",
                           img_path=file_path,
                           heatmap=heatmap_path,
                           label=label,
                           probability=round(float(prob),4))

if __name__ == '__main__':
    app.run(debug=True)
