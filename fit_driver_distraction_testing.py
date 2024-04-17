import os
import sys
import numpy as np
import tensorflow as tf

base_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

if len(sys.argv) == 1:
    print("Please supply the driver distraction captured image")
    sys.exit(0)
else:
    file_name = sys.argv[1]

file_path = os.path.join(base_dir, file_name)
if not os.path.exists(file_path):
    print("Image supplied is not found")
    sys.exit(0)

model_name = "driver distraction.keras"
class_names = ['c0', 'c1', 'C2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
img_height = 180
img_width = 180
# load model
model_path = os.path.join(base_dir, model_name)
if not os.path.exists(model_path):
    print("Model is not found at defined place")
    print("Make sure keras model is placed in the script directory")
    sys.exit(0)
model = tf.keras.models.load_model(model_path)

# Show the model architecture
# model.summary()

# Resize the frame to the target size
img = tf.keras.utils.load_img(
    file_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
