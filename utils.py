import pickle

cv = pickle.load(open("models/cv.pkl","rb"))
clf = pickle.load(open("models/clf.pkl","rb"))
 
def predict(email):
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return prediction

