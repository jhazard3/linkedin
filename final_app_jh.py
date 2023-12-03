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

st.markdown("# Predict LinkedIn Users")

"Complete the following information:"
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

st.write(f"Income:  {Income}")

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

st.write(f"Education:  {Education}")

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

st.write(f"Parent:  {Parent}")

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

st.write(f"Marital:  {Marital}")

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

st.write(f"Female:  {Female}")

#Age
age_values = list(range(0, 99))
Age = st.selectbox("Select Age:", age_values)

st.write(f"Age:  {Age}")

