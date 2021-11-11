import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
st.write('''
#  Hotel Review Sentiment Analysis
''')

st.write("A Web app that detects whether an Review is Positive or Negative")

###Loading model and cv
cv = pickle.load(open('cv.pkl','rb'))
model = pickle.load(open('review.pkl','rb'))

review = st.text_input("Enter Your Review...")
new_review = re.sub('[^a-zA-Z]', ' ', review)
new_review = new_review.lower()
new_review = new_review.split()         
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()


Generate_pred = st.button("Predict Sentiment")
if Generate_pred:
    pred = model.predict(new_X_test)
    if review!="":
        if pred==1:
            st.write("Positive 😀")
        else:
            st.write("Negative 😑")