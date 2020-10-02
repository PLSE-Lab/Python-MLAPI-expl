#!/usr/bin/env python
# coding: utf-8

# # TF Hub - Bounding boxes coordinates corrected

# ### This kernel is highly inspired by [**this code**](https://www.kaggle.com/xhlulu/intro-to-tf-hub-for-object-detection), please do not hesitate to upvote this kernel and the original one if it helps you.
# ### According to [**this discussion**](https://www.kaggle.com/c/open-images-2019-object-detection/discussion/98205), it seems that the order of coordinates for bounding boxes is different between kaggle and tensorflow. Putting the coordinates back in the correct order may give a much higher score using TF Hub as it is shown in the [**original kernel**](https://www.kaggle.com/xhlulu/intro-to-tf-hub-for-object-detection). In this kernel, we will implement this small correction.

# ## Imports

# In[ ]:


import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm


# ## Create prediction string into kaggle format, making correction for BB coordinates

# In[ ]:


def format_prediction_string(image_id, result):
    prediction_strings = []

    for i in range(len(result['detection_scores'])):
        class_name = result['detection_class_names'][i].decode("utf-8")
        # Coordinates of the bounding box in the correct order
        corrected_coordinates = [result['detection_boxes'][i][1], result['detection_boxes'][i][0], result['detection_boxes'][i][3], result['detection_boxes'][i][2]]
        boxes = corrected_coordinates
        score = result['detection_scores'][i]

        prediction_strings.append(
            f"{class_name} {score} " + " ".join(map(str, boxes))
        )

    prediction_string = " ".join(prediction_strings)

    return {
        "ImageID": image_id,
        "PredictionString": prediction_string
    }


# ## Inference

# In[ ]:


sample_submission_df = pd.read_csv('../input/sample_submission.csv')
image_ids = sample_submission_df['ImageId']
predictions = []

# Create session
with tf.Graph().as_default():
    # Create our inference graph
    image_string_placeholder = tf.placeholder(tf.string)
    decoded_image = tf.image.decode_jpeg(image_string_placeholder)
    decoded_image_float = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)
    # Expanding image from (height, width, 3) to (1, height, width, 3)
    image_tensor = tf.expand_dims(decoded_image_float, 0)
    # Load the model from tfhub.dev, and create a detector_output tensor
    model_url = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    detector = hub.Module(model_url)
    detector_output = detector(image_tensor, as_dict=True)
    # Initialize the Session
    init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]
    sess = tf.Session()
    sess.run(init_ops)

# Make prediction on test set
for image_id in tqdm(image_ids):
    # Load the image string
    image_path = f'../input/test/{image_id}.jpg'
    with tf.gfile.Open(image_path, "rb") as binfile:
        image_string = binfile.read()

    # Run our session
    result_out = sess.run(
        detector_output,
        feed_dict={image_string_placeholder: image_string}
    )
    predictions.append(format_prediction_string(image_id, result_out))

sess.close()

pred_df = pd.DataFrame(predictions)
pred_df.to_csv('submission.csv', index=False)

