from flask import Flask, request, render_template
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

calorie = [52, 89, 43, 31, 25, 20, 41, 25, 40, 86, 15, 25, 149, 80, 69, 29, 61, 29, 14, 60, 40, 47, 31, 57, 81, 50, 83, 77, 16, 147, 23, 86, 86, 18, 28, 30]

cnn = tf.keras.models.load_model('trained_model1.h5')

def predict_image(image_path, test_set):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions)  # Return index of max element
    fruit_or_veggie = test_set.class_names[result_index]
    calorie_count = calorie[result_index]
    return fruit_or_veggie, calorie_count

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f:
            filename = f.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(image_path)
            fruit_or_veggie, calorie_count = predict_image(image_path, test_set)
            return render_template('result.html', filename=filename, fruit_or_veggie=fruit_or_veggie, calorie_count=calorie_count)
    return render_template('index.html')




if __name__ == '__main__':
    test_set = tf.keras.utils.image_dataset_from_directory(
        'test',
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(64, 64),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False
    )
    app.run(debug=True)
