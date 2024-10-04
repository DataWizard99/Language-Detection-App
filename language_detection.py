import streamlit as st
import pickle
import re

# Load your model and CountVectorizer from pickle files
def load_model_and_cv():
    model, cv = None, None
    try:
        with open('language_detection_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('cv.pkl', 'rb') as cv_file:
            cv = pickle.load(cv_file)
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
    
    return model, cv

# Function to predict language
def predict_language(text, model, cv):
    transformed_text = cv.transform([text])
    prediction = model.predict(transformed_text)
    return prediction[0]

# Apply CSS to customize font size and style
def apply_custom_css():
    st.markdown(
        """
        <style>
        .custom-title {
            font-size: 50px;
            font-weight: bold;
            color: #ff6347;  /* Tomato color */
            font-family: 'Courier New', Courier, monospace;
        }
        .custom-text {
            font-size: 20px;
            font-family: 'Arial', sans-serif;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

# Main streamlit app
def main():
    # Apply custom CSS
    apply_custom_css()

    # Display the custom title with emojis
    st.markdown('<h1 class="custom-title">🌍 Language Detection App 🧠</h1>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">This app detects the language of the input text. Enter some text below:</p>', unsafe_allow_html=True)

    # Load model and CountVectorizer
    model, cv = load_model_and_cv()

    if model is None or cv is None:
        st.error("Model or CountVectorizer could not be loaded. Please check the file paths.")
        return

    # Text input from user
    user_input = st.text_area("Enter text here", "")

    if st.button("Detect Language"):
        if user_input:
            prediction = predict_language(user_input, model, cv)
            st.success(f"Predicted Language: {prediction}")
        else:
            st.warning("Please enter some text to detect the language.")

if __name__ == "__main__":
    main()
