import streamlit as st
import pyperclip
import joblib

# --- Load model and vectorizer ---
def load_model_and_vectorizer():
    model = joblib.load('fake_review_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# --- Streamlit UI ---
st.set_page_config(page_title="Fake Review Detection", layout="centered")

# Inject CSS
st.markdown("""
    <style>
    /* Background of whole app */
    .stApp {
        background-color: #ffffff;
        padding: 2rem;
    }

    /* Main container */
    .main-container {
        max-width: 1000px;
        margin: 0 auto;
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    /* Title */
    h1 {
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        color: #333333;
    }

    /* Text area */
    textarea {
        background: #f8f9fa;
        border: 1px solid #ced4da;
        border-radius: 12px;
        padding: 1rem;
        font-size: 1rem;
        color: #333333;
    }

    /* Button Styling */
    div.stButton > button {
        width: 100%;
        border: none;
        padding: 0.8rem;
        font-size: 1rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    /* Paste Button */
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        margin-bottom: 1rem;
    }
    div.stButton > button:first-child:hover {
        background-color: #0056b3;
    }

    /* Predict Button */
    div.stButton > button:last-child {
        background-color: #7700ff;
        color: white;
    }
    div.stButton > button:last-child:hover {
        background-color: #5a018d;
    }

    /* Success and Error messages */
    .stAlert {
        border-radius: 12px;
        padding: 1rem;
    }

    /* Center text */
    .center-text {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Content ---

st.title("🛡️ Fake Review Detection App")
st.write("<div class='center-text'>Paste or enter a complaint/review text to predict if it is FAKE or GENUINE.</div>", unsafe_allow_html=True)

# Initialize session state
if 'user_review' not in st.session_state:
    st.session_state.user_review = ""

user_review = st.text_area("✍️ Enter Review Text:", value=st.session_state.user_review)

# Copy (Paste) Button
def paste_action():
    try:
        pasted_text = pyperclip.paste()
        st.session_state.user_review = pasted_text
        st.rerun()
    except Exception as e:
        st.error(f"Failed to paste text: {e}")

if st.button("📋 Paste Review"):
    paste_action()

# Predict Button
if st.button("🔍 Check Review"):
    if user_review.strip() == "":
        st.warning("⚠️ Please enter a review text first.")
    else:
        review_transformed = vectorizer.transform([user_review])
        prediction = model.predict(review_transformed)[0]

        if prediction == 1:
            st.error("❌ This review is likely FAKE!")
        else:
            st.success("✅ This review looks GENUINE!")

st.markdown('</div>', unsafe_allow_html=True)
