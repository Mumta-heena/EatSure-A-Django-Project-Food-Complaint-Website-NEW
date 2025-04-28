import streamlit as st
import pyperclip
import joblib

# --- Load the saved model and vectorizer ---
def load_model_and_vectorizer():
    # Load the model
    model = joblib.load('fake_review_model.pkl')
    
    # Load the TF-IDF vectorizer
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    return model, vectorizer
  

model, vectorizer = load_model_and_vectorizer()

# --- Streamlit UI ---
st.title("🛡️ Fake Review Detection App")

st.write("Paste the complaint, and the model will predict whether it is **fake** or **genuine**.")

# Initialize session state for user review if not already set
if 'user_review' not in st.session_state:
    st.session_state.user_review = ""

# Text area for the user to input a review
user_review = st.text_area("✍️ Enter the Review Text:", value=st.session_state.user_review)

# Paste button functionality
def paste_action():
    try:
        # Get text from the clipboard
        pasted_text = pyperclip.paste()
        
        # Update session state with the pasted text
        st.session_state.user_review = pasted_text
        st.rerun()  # Rerun the app to reflect the changes
        
    except Exception as e:
        st.error(f"Failed to paste text: {e}")

# Paste button
if st.button("Paste Review"):
    paste_action()

if st.button("Check Review"):
    if user_review.strip() == "":
        st.warning("Please enter a review text.")
    else:
        # Preprocess: transform the input using the loaded TF-IDF vectorizer
        review_transformed = vectorizer.transform([user_review])
        
        # Predict
        prediction = model.predict(review_transformed)[0]

        # Display result
        if prediction == 1:
            st.error("⚠️ This review is likely FAKE!")
        else:
            st.success("✅ This review looks GENUINE!")
