import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

st.markdown("# Are you LinkedIn?")
st.markdown("#### By Jamie Hazard, Georgetown University, Fall 2023")

#Read input file
s = pd.read_csv("social_media_usage.csv")

#Function clean_sm
def clean_sm(x):
    return np.where(x == 1, 1, 0)

#Create dataframe, update values
ss = pd.DataFrame({
    "sm_li":s["web1h"],
    "Income":np.where(s["income"] >9, np.nan, s["income"]),
    "Education":np.where(s["educ2"] >8, np.nan, s["educ2"]),
    "Parent":np.where(s["par"]== 1, 1, 0),
    "Marital":np.where(s["marital"] == 1, 1, 0),
    "Female":np.where(s["gender"] == 2, 1, 0),
    "Age":np.where(s["age"] >98, np.nan, s["age"])})

#Run function clean_sm
ss['sm_li'] = clean_sm(ss['sm_li'])

#Drop NA
ss = ss.dropna()

#Create target vector and feature set
y = ss["sm_li"]
x = ss[["Income", "Education", "Parent", "Marital", "Female", "Age"]]

#Split into training/test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=100)

#Instantiate a logistic regression model
lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train,y_train)

#Evaluate the model
y_pred = lr.predict(x_test)

"This application calculates the likelihood that a person is a user of LinkedIn based on an input set of social media usage and a logistic regression model."

"Complete the following information, which will be used to predict whether or not you are a LinkedIn user:"

#Income
Income = st.selectbox(label="Household Income",
options=("Less than $10,000",
        "$10,000 to $19,999",
        "$20,000 to $29,999",
        "$30,000 to $39,999",
        "$40,000 to $49,999",
        "$50,000 to $74,999",
        "$75,000 to $99,999",
        "$100,000 to $149,999",
        "$150,000 or more",
        "Don't know",
        "Refused"))

if Income == "Less than $10,000":
        Income = 1
elif Income == "$10,000 to $19,999":
        Income = 2     
elif Income == "$20,000 to $29,999":
        Income = 3
elif Income == "$30,000 to $39,999":
        Income = 4
elif Income == "$40,000 to $49,999":
        Income = 5
elif Income == "$50,000 to $74,999":
        Income = 6
elif Income == "$75,000 to $99,999":
        Income = 7
elif Income == "$100,000 to $149,999":
        Income = 8
elif Income == "$150,000 or more":
        Income = 9
elif Income == "Don't know":
        Income = None
elif Income == "Refused":
        Income = None
else:
        Income = None

#Education
Education = st.selectbox(label="Highest Level of School/Degree Completed",
options=("Less than high school (Grades 1-8 or no formal schooling)",
        "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
        "High school graduate (Grade 12 with diploma or GED certificate)",
        "Some college, no degree (includes some community college)",
        "Two-year associate degree from a college or university",
        "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
        "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
        "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)",
        "Don't know",
        "Refused"))

if Education == "Less than high school (Grades 1-8 or no formal schooling)":
        Education = 1
elif Education == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
        Education = 2     
elif Education == "High school graduate (Grade 12 with diploma or GED certificate)":
        Education = 3
elif Education == "Some college, no degree (includes some community college)":
        Education = 4
elif Education == "Two-year associate degree from a college or university":
        Education = 5
elif Education == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
        Education = 6
elif Education == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
        Education = 7
elif Education == "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)":
        Education = 8
elif Education == "Don't know":
        Education = None
elif Education == "Refused":
        Education = None
else:
        Education = None

#Parent
Parent = st.selectbox(label="Are you a parent of a child under 18 living in your home?",
options=("Yes",
        "No",
        "Don't know",
        "Refused"))

if Parent == "Yes":
        Parent = 1
else:
        Parent = 0

#Marital
Marital = st.selectbox(label="Marital Status",
options=("Married",
        "Living with a partner",
        "Divorced",
        "Separated",
        "Widowed",
        "Never been married",
        "Don't know",
        "Refused"))

if Marital == "Married":
        Marital = 1
else:
        Marital = 0

#Female
Female = st.selectbox(label="Gender",
options=("Male",
        "Female",
        "Other",
        "Don't know",
        "Refused"))

if Female == "Female":
        Female = 1
else:
        Female = 0

Age = st.slider(label="Age",
                min_value = 1,
                max_value=98,
                value = 30)

#Display input results (temporary)
#st.write(f"Income:  {Income}")
#st.write(f"Education:  {Education}")
#st.write(f"Parent:  {Parent}")
#st.write(f"Marital:  {Marital}")
#st.write(f"Female:  {Female}")
#st.write(f"Age:  {Age}")

#Use inputs to update dataframe
pred_df = pd.DataFrame({
    "Income": [Income],
    "Education": [Education],
    "Parent": [Parent],
    "Marital": [Marital],
    "Female": [Female],
    "Age": [Age]})

#Calculate probability
probability = lr.predict_proba(pred_df)[:, 1]
probability_rev = 1-probability
probability_pct = round(probability[0]*100,2)
st.write(f"**The probability of being a LinkedIn user is:  {probability_pct}%**")

