from flask import Flask, request, jsonify, render_template
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import joblib


app = Flask(__name__)


# Load the TF-IDF vectorizer for Intent Recognition from the file
tfidf_vectorizer_intent = joblib.load("models/tfidf_vectorizer_intent.pkl")

# Load the trained Intent Recognition model from the file
intent_recognition_model = joblib.load("models/intent_recognition_model.pkl")

# Load the TF-IDF vectorizer for Category Classification from the file
tfidf_vectorizer_category = joblib.load("models/tfidf_vectorizer_category.pkl")

# Load the trained Category Classification model from the file
category_classification_model = joblib.load("models/category_classification_model.pkl")


# Define the intent responses for each intent category
intent_responses = {
    "cancel_order": "Your order has been canceled successfully.",
    "change_order": "Your order has been updated.",
    "change_shipping_address": "Your shipping address has been changed.",
    "check_cancellation_fee": "The cancellation fee for your order is $10.",
    "check_invoices": "You can view and download your invoices from your account dashboard.",
    "check_payment_methods": "We accept various payment methods, including credit cards, debit cards, and PayPal.",
    "check_refund_policy": "Our refund policy allows for full refunds within 30 days of purchase.",
    "complaint": "I'm sorry to hear that you're facing an issue. Please provide more details, and we'll assist you.",
    "contact_customer_service": "You can reach our customer service team at [Phone Number] or [Email Address].",
    "contact_human_agent": "One of our human agents will be happy to assist you shortly.",
    "create_account": "You can create an online account on our website. Please visit the 'Create Account' page to get started.",
    "delete_account": "Your account has been deleted successfully.",
    "delivery_options": "You can choose from standard or express shipping options during checkout.",
    "delivery_period": "The estimated delivery time for your order is 2-3 business days.",
    "edit_account": "You can edit your account information on the 'Account Settings' page.",
    "get_invoice": "You can view and download your invoice for the latest order in your account dashboard.",
    "get_refund": "Your refund request has been processed, and the amount will be credited back to your original payment method.",
    "newsletter_subscription": "Thank you for subscribing to our newsletter!",
    "payment_issue": "Please provide more details about the payment issue, and we'll assist you.",
    "place_order": "Your order has been successfully placed.",
    "recover_password": "You can reset your password by clicking on the 'Forgot Password' link on the login page.",
    "registration_problems": "I'm sorry to hear that you're experiencing registration issues. Please provide more details, and we'll assist you.",
    "review": "Thank you for your review! Your feedback is valuable to us.",
    "set_up_shipping_address": "You can add or update your shipping address in your account settings.",
    "switch_account": "You can switch to a different account by logging out and then logging in with the desired account credentials.",
    "track_order": "You can track your order using the tracking number provided in the order confirmation email.",
    "track_refund": "Your refund request is being processed, and you will receive a confirmation email once it's completed."
}


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Apply stemming or lemmatization (if needed)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens


def predict_intent(user_input):
    processed_input = preprocess_text(user_input)
    input_tfidf = tfidf_vectorizer_intent.transform([' '.join(processed_input)])
    predicted_intent = intent_recognition_model.predict(input_tfidf)[0]
    return predicted_intent


def predict_category(user_input):
    processed_input = preprocess_text(user_input)
    input_tfidf = tfidf_vectorizer_category.transform([' '.join(processed_input)])
    predicted_category = category_classification_model.predict(input_tfidf)[0]
    return predicted_category


def virtual_assistant(user_input):
    processed_input = preprocess_text(user_input)
    predicted_intent = predict_intent(user_input)
    predicted_category = predict_category(user_input)
    response = intent_responses.get(predicted_intent, "I'm sorry, I don't understand.")
    return response


# Define a route handler for the root URL
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json['user_input']
    response = virtual_assistant(user_input)  # Pass the user_input to virtual_assistant function
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)