import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import smtplib

# Initialize label encoder and scaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

df = pd.read_csv("CreditCardFraud.csv")

# Load statistics data
statistics_data = df.describe()

# Preprocess data function
def preprocess_data(input_data):
    # Label encode categorical variables
    # Manual encoding for categorical variables
    input_data['Transaction Type'] = input_data['Transaction Type'].map({'Withdrawal': 0, 'Purchase': 1, 'Transfer': 2})
    input_data["Cardholder's Country"] = input_data["Cardholder's Country"].map({'US': 0, 'CA': 1, 'UK': 2, 'AU': 3})
    input_data["Merchant's Country"] = input_data["Merchant's Country"].map({'AU': 0, 'US': 1, 'CA': 2, 'UK': 3})
    input_data['Transaction Currency'] = input_data['Transaction Currency'].map({'GBP': 0, 'AUD': 1, 'USD': 2, 'CAD': 3})
    input_data['Device Type'] = input_data['Device Type'].map({'Desktop': 0, 'Mobile': 1, 'Tablet': 2})
    input_data['Gender'] = input_data['Gender'].map({'Female': 0, 'Male': 1})
    
    return input_data

X = df.drop(columns=['Fraud','MCC'])
y = df['Fraud']

X = preprocess_data(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest models
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
st.write("Random Forest Accuracy:", rf_accuracy)

# Main function to create the Streamlit app
def main():
    st.title('Credit Card Fraud Detection App')
    
    # Create input widgets for each parameter
    transaction_amount = st.slider('Transaction Amount', min_value=float(statistics_data['Transaction Amount']['min']), max_value=float(statistics_data['Transaction Amount']['max']), value=float(statistics_data['Transaction Amount']['mean']))
    transaction_type = st.selectbox('Transaction Type', ['Withdrawal', 'Purchase', 'Transfer'])
    cardholder_country = st.selectbox("Cardholder's Country", ['US', 'CA', 'UK', 'AU'])
    merchant_country = st.selectbox("Merchant's Country", ['AU', 'US', 'CA', 'UK'])
    transaction_currency = st.selectbox('Transaction Currency', ['GBP', 'AUD', 'USD', 'CAD'])
    transaction_time = st.slider('Transaction Time', min_value=float(statistics_data['Transaction Time']['min']), max_value=float(statistics_data['Transaction Time']['max']), value=float(statistics_data['Transaction Time']['mean']))
    device_type = st.selectbox('Device Type', ['Desktop', 'Mobile', 'Tablet'])
    distance = st.slider('Distance (km)', min_value=float(statistics_data['Distance (km)']['min']), max_value=float(statistics_data['Distance (km)']['max']), value=float(statistics_data['Distance (km)']['mean']))
    merchant_reputation = st.slider('Merchant Reputation Score', min_value=float(statistics_data['Merchant Reputation Score']['min']), max_value=float(statistics_data['Merchant Reputation Score']['max']), value=float(statistics_data['Merchant Reputation Score']['mean']))
    anomaly_score = st.slider('Transaction Anomaly Score', min_value=float(statistics_data['Transaction Anomaly Score']['min']), max_value=float(statistics_data['Transaction Anomaly Score']['max']), value=float(statistics_data['Transaction Anomaly Score']['mean']))
    day_of_week = st.slider('Day of Week', min_value=int(statistics_data['Day of Week']['min']), max_value=int(statistics_data['Day of Week']['max']), value=int(statistics_data['Day of Week']['mean']), step=1)
    transaction_frequency = st.slider('Transaction Frequency', min_value=float(statistics_data['Transaction Frequency']['min']), max_value=float(statistics_data['Transaction Frequency']['max']), value=float(statistics_data['Transaction Frequency']['mean']), step=1.0)
    credit_limit = st.slider('Credit Limit', min_value=float(statistics_data['Credit Limit']['min']), max_value=float(statistics_data['Credit Limit']['max']), value=float(statistics_data['Credit Limit']['mean']))
    account_age = st.slider('Account Age (months)', min_value=float(statistics_data['Account Age (months)']['min']), max_value=float(statistics_data['Account Age (months)']['max']), value=float(statistics_data['Account Age (months)']['mean']))
    age = st.slider('Age', min_value=float(statistics_data['Age']['min']), max_value=float(statistics_data['Age']['max']), value=float(statistics_data['Age']['mean']))
    gender = st.selectbox('Gender', ['Female', 'Male'])
    income = st.slider('Income', min_value=float(statistics_data['Income']['min']), max_value=float(statistics_data['Income']['max']), value=float(statistics_data['Income']['mean']))
    email = st.text_input('Email', '')
    
    # Create a button to trigger prediction
    if st.button('Predict'):
        # Create a dictionary with input data
        input_data = pd.DataFrame({
            'Transaction Amount': [transaction_amount],
            'Transaction Type': [transaction_type],
            "Cardholder's Country": [cardholder_country],
            "Merchant's Country": [merchant_country],
            'Transaction Currency': [transaction_currency],
            'Transaction Time': [transaction_time],
            'Device Type': [device_type],
            'Distance (km)': [distance],
            'Merchant Reputation Score': [merchant_reputation],
            'Transaction Anomaly Score': [anomaly_score],
            'Day of Week': [day_of_week],
            'Transaction Frequency': [transaction_frequency],
            'Credit Limit': [credit_limit],
            'Account Age (months)': [account_age],
            'Age': [age],
            'Gender': [gender],
            'Income': [income],
        })
        
        # Preprocess the input data
        input_data_preprocessed = preprocess_data(input_data)

        # Make predictions
        prediction = rf_classifier.predict(input_data_preprocessed)

        # Print the prediction
        st.write('Prediction:')
        if prediction[0] == 1:
            st.error('Fraudulent Credit Card')
            if email:
                # Send fraud alert email
                send_email(email, is_fraud=True)
                
                # Prompt user to block the credit card
                block_card = st.selectbox('This particular Credit card is fraudulent. Do you want to block this?', ('Yes', 'No'))
                if block_card == 'Yes':
                    # Send email regarding blocking of credit card
                    send_email(email, is_fraud=False)
        else:
            st.success('Non-Fraudulent Credit Card')

def send_email(receiver_email, is_fraud):
    # Email configuration
    sender_email = "asaninnovators48@gmail.com"  # Replace with your email
    password = "cjtx ioar dsls gnmw"  # Replace with your password
    
    if is_fraud:
        subject = "Fraud Alert!"
        body = "Our system has detected a potentially fraudulent credit card transaction."
    else:
        subject = "Credit Card Blocked"
        body = "Your credit card has been blocked due to fraudulent activity."
    
    # Create an SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(sender_email, password)
    
    # Constructing the message
    message = f"Subject: {subject}\n\n{body}"
    
    # Sending the email
    s.sendmail(sender_email, receiver_email, message)
    
    # Terminating the session
    s.quit()

# Run the Streamlit app
if __name__ == '__main__':
    main()
