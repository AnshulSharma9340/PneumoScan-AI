import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

def get_grad_cam(image_path, model_path='model/final_model.h5'):
    model = load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("Conv_1").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]

    cam = np.mean(grads * conv_outputs[0], axis=-1)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    final = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cam_path = "static/heatmap.jpg"
    cv2.imwrite(cam_path, final)

    return cam_path, predictions.numpy()[0][0]
