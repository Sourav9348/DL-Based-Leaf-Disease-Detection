import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Potato Disease Classifier", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
    }
    .result-container {
        background-color: #ffc107;
        padding: 20px;
        border-radius: 5px;
    }
    .result-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: #6c757d;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
    }
    .footer p {
        margin: 0;
        padding: 0;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# Hide Streamlit's default menu and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


def predict_class(image):
    classifier_model = tf.keras.models.load_model('model_v1.h5', compile=False)
    test_image = image.resize((128, 128))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['Early_blight', 'Healthy', 'Late_blight']
    prediction = classifier_model.predict(test_image)
    return prediction[0]


def main():
    st.title("Potato Disease Classifier")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                prediction = predict_class(image)
                class_names = ['Early_blight', 'Healthy', 'Late_blight']
                result = class_names[np.argmax(prediction)]
                confidence = round(100 * np.max(prediction), 2)

                with col2:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown(f'<p class="result-header">Result: {result}</p>', unsafe_allow_html=True)

                    # Plot prediction probabilities
                    fig, ax = plt.subplots()
                    y_pos = np.arange(len(class_names))
                    ax.barh(y_pos, prediction * 100, align='center')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(class_names)
                    ax.invert_yaxis()  # Labels read top-to-bottom
                    ax.set_xlabel('Probability (%)')
                    ax.set_xlim(0, 100)  # Set x-axis limit to 100%

                    for i, v in enumerate(prediction * 100):
                        ax.text(v + 1, i, f'{v:.2f}%', color='black', va='center')

                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()

    # Add the custom footer
    st.markdown("""
    <div class="footer">
        <p>Enjoy the prediction and thanks for using.</p>
        <p>Thanks & regards,</p>
        <p>Sourav</p>
    </div>
    """, unsafe_allow_html=True)







# import streamlit as st
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# # Hiding Streamlit's default menu and footer
# hide_streamlit_style = """
# <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)
#
# st.title('Potato Leaf Disease Prediction')
#
#
# def main():
#     file_uploaded = st.file_uploader('Choose an image...', type='jpg')
#     if file_uploaded is not None:
#         image = Image.open(file_uploaded)
#         st.write("Uploaded Image.")
#         figure = plt.figure()
#         plt.imshow(image)
#         plt.axis('off')
#         st.pyplot(figure)
#         result, confidence = predict_class(image)
#         st.write('Prediction : {}'.format(result))
#         st.write('Confidence : {}%'.format(confidence))
#
#
# def predict_class(image):
#     with st.spinner('Loading Model...'):
#         classifier_model = tf.keras.models.load_model('model_v1.h5', compile=False)
#
#     # Preprocess the image to match the model's input shape
#     test_image = image.resize((128, 128))  # Ensure this matches the original model's input size
#     test_image = tf.keras.preprocessing.image.img_to_array(test_image)
#     test_image = np.expand_dims(test_image, axis=0)
#
#     class_names = ['Early_blight', 'Healthy', 'Late_blight']
#
#     prediction = classifier_model.predict(test_image)
#     confidence = round(100 * (np.max(prediction[0])), 2)
#     final_pred = class_names[np.argmax(prediction)]
#     return final_pred, confidence
#
#
# footer = """
# <style>
# a:link , a:visited{
#     color: white;
#     background-color: transparent;
#     text-decoration: None;
# }
# a:hover,  a:active {
#     color: red;
#     background-color: transparent;
#     text-decoration: None;
# }
# .footer {
#     position: fixed;
#     left: 0;
#     bottom: 0;
#     width: 100%;
#     background-color: transparent;
#     color: black;
#     text-align: center;
# }
# </style>
# <div class="footer">
#     <p>Enjoy the prediction and thanks for using.</p>
#     <p>Thanks and Regards,</p>
#     <p>Sourav</p>
# </div>
# """
# st.markdown(footer, unsafe_allow_html=True)
#
# if __name__ == '__main__':
#     main()

