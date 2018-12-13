from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import Form, SubmitField
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


app = Flask(__name__)


def tokenizer(text):
    return text.split()

def train_classify():
	data = pd.read_csv('BlackFriday.csv', encoding='utf-8')
	lb = LabelEncoder()
	data['Gender'] = lb.fit_transform(data['Gender'])
	data['Age'] = lb.fit_transform(data['Age'])
	data['City_Category'] = lb.fit_transform(data['City_Category'])
	data['Stay_In_Current_City_Years'] = lb.fit_transform(data['Stay_In_Current_City_Years'])
	data.drop(columns = ["User_ID","Product_ID","Occupation","Product_Category_1","Product_Category_2","Product_Category_3"],inplace=True)
	X = data.drop(['Purchase'],axis=1)
	y = data['Purchase']
    
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
	regr = RandomForestRegressor(max_depth=5,random_state=42,n_estimators=100)
	regr.fit(X_train,y_train)
    
	blackFriday_path = 'blackFriday.pkl'
	blackFriday = open(blackFriday_path, 'wb')
	pickle.dump(regr, blackFriday)
	blackFriday.close()

train = train_classify()

def unpickle():
	blackFriday_path = 'blackFriday.pkl'
	model_blackFriday = open(blackFriday_path, 'rb')
	clf_new = pickle.load(model_blackFriday)
	return clf_new

class blackFridayForm(Form):
	submit = SubmitField("Send")
@app.route("/")
def hello():
	form = blackFridayForm(request.form)
	return render_template('main.html', form=form)

@app.route('/analysis', methods=['POST'])
def result():
    form = blackFridayForm(request.form)
    if request.method == 'POST' and form.validate():
        
        gender = request.form['gender']
        ageRange = request.form['ageRange']
        maritalStatus = request.form['maritalStatus']      
        cityCategory = request.form['cityCategory']
        noOfYears = request.form['noOfYears']
        
        genderValue = getTransformedValueForGender(gender)
        ageRangeValue = getTransformedValueForAgeRange(ageRange)      
        cityCategoryValue = getTransformedValueForCityCategory(cityCategory)
        int_noOfYears = int(noOfYears)
        maritalStatusValue = getTransformedValueForMartialStatus(maritalStatus)
        
        requestedData = [genderValue, ageRangeValue, cityCategoryValue, int_noOfYears, maritalStatusValue]
        
        review_this = unpickle()
        result = review_this.predict([requestedData])
        
        return render_template('analysis.html',
                                gender=gender,ageRange=ageRange, maritalStatus=maritalStatus,cityCategory=cityCategory,noOfYears=noOfYears,                                 result=result)
    return render_template('main.html', form=form)

def getTransformedValueForGender(gender):
    if gender == 'Female':
       return 0  
    return 1

def getTransformedValueForAgeRange(ageRange):
    if ageRange == '0-17':
       return 0
    if ageRange == '18-25':
       return 1
    if ageRange == '26-35':
       return 2
    if ageRange == '36-45':
       return 3
    if ageRange == '46-50':
       return 4
    if ageRange == '51-55':
       return 5
    return 6

def getTransformedValueForMartialStatus(maritalStatus):
    if maritalStatus == 'Single':
       return 0  
    return 1

def getTransformedValueForCityCategory(cityCategory):
    if cityCategory == 'A':
       return 0 
    if cityCategory == 'B':
       return 1 
    return 2
    
if __name__ == "__main__":
    app.run(debug=True)