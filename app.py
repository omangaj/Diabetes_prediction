import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import streamlit as st
from PIL import Image

dataset = pd.read_csv("diabetes.csv")

dataset.shape

dataset.isnull().sum()
dataset.duplicated().sum()
dataset = dataset.drop_duplicates()

dataset.describe()
dataset.info()

x = dataset.iloc[:, :-1]
y = dataset["Outcome"]

sc = StandardScaler()
sc.fit(x)
sc_x = sc.fit_transform(x)
x = pd.DataFrame(sc_x, columns=x.columns)



# lo = LogisticRegression()

# for i in range(1,100):
#     x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = i)
#     lo.fit(x_train,y_train)
#     train_score = lo.score(x_train,y_train)*100
#     test_score = lo.score(x_test,y_test)*100
#     x_pred = lo.predict(x_test)
#     acc = accuracy_score(y_test,x_pred)*100

#     if train_score > test_score > train_score-2:
#         print(i," ",train_score," ",test_score," ",acc)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

lo = LogisticRegression(class_weight={1: 2})
lo.fit(x_train, y_train)

train_score = lo.score(x_train, y_train) * 100
test_score = lo.score(x_test, y_test) * 100

y_test_pred = lo.predict(x_test)
y_train_pred = lo.predict(x_train)

acc_test = accuracy_score(y_test,y_test_pred) * 100
acc_train = accuracy_score(y_train,y_train_pred) * 100



def app():
    img = Image.open(r"doctor.jpg")
    img = img.resize((200,200))
    st.image(img,caption="Diabetes_image",width=200)

    st.title("Diabetes Disease Prediction")

    st.sidebar.title("input features")
    Pregnancies = st.sidebar.slider("Pregnancies",0,17,3)
    Glucose = st.sidebar.slider("Glucose",0,199,117,)
    BloodPressure = st.sidebar.slider("Blood Pressure",0,122,72)
    SkinThickness = st.sidebar.slider("Skin Thickness",0,99,23)
    Insulin = st.sidebar.slider("Insulin",0,846,30)
    BMI = st.sidebar.slider("BMI",0.0,67.1,32.0)
    DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function",0.078,2.42,0.3725,0.001)
    Age = st.sidebar.slider("Age",21,81,29)

    data = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
    np_array = np.asarray(data)
    reshape_data = np_array.reshape(1, -1)
    val = lo.predict(reshape_data)
    if val[0] == 1:
        st.warning(f"{val[0]} : Yes, You have diabetes.")
    else:
        st.success(f"{val[0]} : No, You not have diabetes.")


if __name__ == "__main__":
    app()