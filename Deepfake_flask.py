import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = tf.keras.models.load_model("deepfake_model-005.keras")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            img_array = preprocess_image(file_path)

            prediction = model.predict(img_array)[0][0] 
            result = "REAL" if prediction >= 0.5 else "FAKE"
            confidence = f"{prediction:.2f}"

            return render_template("index.html", file_path=file_path, result=result, confidence=confidence)

    return render_template("deppfake_html.html", file_path=None)

if __name__ == "__main__":
    app.run(debug=True)
