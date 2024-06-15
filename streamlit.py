import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Hiding Streamlit's default menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('Potato Leaf Disease Prediction')


def main():
    file_uploaded = st.file_uploader('Choose an image...', type='jpg')
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        result, confidence = predict_class(image)
        st.write('Prediction : {}'.format(result))
        st.write('Confidence : {}%'.format(confidence))


def predict_class(image):
    with st.spinner('Loading Model...'):
        classifier_model = tf.keras.models.load_model('model_v1.h5', compile=False)

    # Preprocess the image to match the model's input shape
    test_image = image.resize((128, 128))  # Ensure this matches the original model's input size
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    class_names = ['Early_blight', 'Healthy', 'Late_blight']

    prediction = classifier_model.predict(test_image)
    confidence = round(100 * (np.max(prediction[0])), 2)
    final_pred = class_names[np.argmax(prediction)]
    return final_pred, confidence


footer = """
<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}
a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
}
</style>
<div class="footer">
    <p>Enjoy the prediction and thanks for using.</p>
    <p>Thanks and Regards,</p>
    <p>Sourav</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

if __name__ == '__main__':
    main()



# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
#
# # Function to load and preprocess the image
# def load_and_preprocess_image(image_file):
#     # Load the image
#     img = Image.open(image_file)
#     # Resize the image to match the model input size
#     img = img.resize((256, 256))
#     # Convert to numpy array
#     img_array = np.array(img)
#     # Expand dimensions to create a batch of size 1
#     img_array = np.expand_dims(img_array, axis=0)
#     # Rescale pixel values to [0, 1]
#     img_array = img_array / 255.0
#     return img_array
#
# # Load the model with custom objects
# @st.cache_data
# def load_model():
#     model = tf.keras.models.load_model('Potato_Plant_disease_detection_model.h5', compile=False)
#     model.compile(optimizer='adam',
#                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#                   metrics=['accuracy'])
#     return model
#
# # Define class names (adjust according to your dataset)
# class_names = ['Healthy', 'Early Blight', 'Late Blight']
#
# # Streamlit UI
# def main():
#     st.title('Potato Leaf Disease Classifier')
#
#     # File uploader widget
#     st.set_option('deprecation.showfileUploaderEncoding', False)
#     uploaded_file = st.file_uploader("Choose an image of a potato leaf...", type="jpg")
#
#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded potato leaf image', use_column_width=True)
#
#         # Preprocess the image
#         img_array = load_and_preprocess_image(uploaded_file)
#
#         # Load the model
#         model = load_model()
#
#         # Predict the class probabilities
#         predictions = model.predict(img_array)
#
#         # Get the predicted class label
#         predicted_class = np.argmax(predictions[0])
#         confidence = np.max(predictions[0]) * 100
#
#         st.write(f'Prediction: {class_names[predicted_class]}')
#         st.write(f'Confidence: {confidence:.2f}%')
#
# # Entry point of the Streamlit application
# if __name__ == '__main__':
#     main()
