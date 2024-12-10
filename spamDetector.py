import streamlit as st

import pickle

# Load the pre-trained model and vectorizer
model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

def main():
    # Application title and description
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or ham.")
    st.write("Project by: **Sudarshan Birajdar**")
    
    # Subheader for classification
    st.subheader("Classification")
    user_input = st.text_area("Enter an email to classify", height=150)

    # Classification logic
    if st.button("Classify"):
        if user_input:
            data = [user_input]
            vec = cv.transform(data).toarray()
            result = model.predict(vec)
            if result[0] == 0:
                st.success("This is Not A Spam Email")
            else:
                st.error("This is A Spam Email")
        else:
            st.warning("Please enter an email to classify.")

    # Add contact details at the bottom
    st.markdown("------ ")
    # st.markdown(" project by : Birajdar Sudarshan")
    st.markdown("### Contact Details")
    st.markdown("📧 Email: birajdarsudarshan6@example.com")  # Replace with your actual email
    st.markdown("🔗 [LinkedIn](https://linkedin.com/in/sudarshanbirajdar)")  # Replace with your LinkedIn profile
    st.markdown("🌐 [GitHub](https://github.com/bsudarshan7)")  # Replace with your GitHub link

# Run the main function
main()
