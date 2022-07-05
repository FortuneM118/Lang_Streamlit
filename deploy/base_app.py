"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os,re

# Data dependencies
import pandas as pd

import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')

# Vectorizer
news_vectorizer = open("resources/CountVectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

def remove_special_characters(text):
	text= re.sub(r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+','',text)
	text= re.sub(r'[-]',' ',text)
	text= re.sub(r'[_]', ' ', text)
	text= re.sub(r'[^\w\s]','',text)
	text= re.sub('[0-9]+', '', text)
	text= re.sub(r'[^\x00-\x7f]',r'', text)
	text = re.sub('RT', '', text)
	text = re.sub('rt', '', text)
	return text



STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Language Classifer")
	st.subheader("Text language classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw lang text data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Model Prediction")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			#tweet_text= remove_special_characters(tweet_text)
			tweet_text= tweet_text.lower()
			tweet_text= remove_stopwords(tweet_text)
			vect_text = tweet_cv.transform([tweet_text]).toarray()   
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/MultinomialNB_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			sentiment = ''
			if prediction == 'xho':
				sentiment = 'isiXhosa'
			if prediction == 'eng':
				sentiment = 'English'
			if prediction == 'nso':
				sentiment = 'Sepedi'
			if prediction == 'ven':
				sentiment = 'Tshivenda'
			if prediction == 'tsn':
				sentiment = 'Setswana'
			if prediction == 'nbl':
				sentiment = 'isiNdebele'
			if prediction == 'zul':
				sentiment = 'isiZulu'
			if prediction == 'ssw':
				sentiment = 'siSwati'
			if prediction == 'tso':
				sentiment = 'Xitsonga'
			if prediction == 'sot':
				sentiment = 'Sesotho'
			if prediction == 'afr':
				sentiment = 'Afrikaans'
			st.success("Text Categorized as: {} {}".format(prediction,sentiment))
			st.markdown("Clean Text: "+ tweet_text)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
