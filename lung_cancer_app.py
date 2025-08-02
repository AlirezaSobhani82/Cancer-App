import streamlit as st
import numpy as np
import pandas as pd
import joblib

# بارگذاری مدل، اسکیلر و نام ستون‌ها
model = joblib.load("D:\\10project Aiolearn\\Cancer\\logistic_model.pkl")
scaler = joblib.load("D:\\10project Aiolearn\\Cancer\\scaler.pkl")
columns = joblib.load("D:\\10project Aiolearn\\Cancer\\feature_names.pkl")

st.set_page_config(page_title="تشخیص سرطان ریه", page_icon="🫁")
st.title("🫁 اپلیکیشن تشخیص احتمال سرطان ریه")
st.markdown("لطفاً اطلاعات زیر را وارد کنید:")

def to_binary(value): return 1 if value == "بله" or value == "مرد" else 0

# فرم ورودی بر اساس ترتیب ستون‌ها
gender = st.radio("جنسیت", ["مرد", "زن"])
age = st.slider("سن", 18, 100, 30)
smoking = st.radio("آیا سیگار می‌کشید؟", ["بله", "خیر"])
yellow_fingers = st.radio("آیا انگشتان زرد دارید؟", ["بله", "خیر"])
anxiety = st.radio("آیا اضطراب دارید؟", ["بله", "خیر"])
peer_pressure = st.radio("آیا تحت فشار دوستان هستید؟", ["بله", "خیر"])
chronic_disease = st.radio("آیا بیماری مزمن دارید؟", ["بله", "خیر"])
fatigue = st.radio("آیا احساس خستگی دارید؟", ["بله", "خیر"])
allergy = st.radio("آیا آلرژی دارید؟", ["بله", "خیر"])
wheezing = st.radio("آیا خس‌خس سینه دارید؟", ["بله", "خیر"])
alcohol_consuming = st.radio("آیا الکل مصرف می‌کنید؟", ["بله", "خیر"])
coughing = st.radio("آیا سرفه می‌کنید؟", ["بله", "خیر"])
shortness_of_breath = st.radio("آیا تنگی نفس دارید؟", ["بله", "خیر"])
swallowing_difficulty = st.radio("آیا در بلع مشکل دارید؟", ["بله", "خیر"])
chest_pain = st.radio("آیا درد قفسه سینه دارید؟", ["بله", "خیر"])

# ساخت آرایه ورودی با ترتیب دقیق
input_data = np.array([[
    to_binary(gender),
    age,
    to_binary(smoking),
    to_binary(yellow_fingers),
    to_binary(anxiety),
    to_binary(peer_pressure),
    to_binary(chronic_disease),
    to_binary(fatigue),
    to_binary(allergy),
    to_binary(wheezing),
    to_binary(alcohol_consuming),
    to_binary(coughing),
    to_binary(shortness_of_breath),
    to_binary(swallowing_difficulty),
    to_binary(chest_pain)
]])

# تبدیل به DataFrame با نام ستون‌ها
input_df = pd.DataFrame(input_data, columns=columns)
input_scaled = scaler.transform(input_df)

# دکمه پیش‌بینی
if st.button("تشخیص بده"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ احتمال سرطان ریه وجود دارد. لطفاً با پزشک مشورت کنید.")
    else:
        st.success("✅ احتمال سرطان ریه وجود ندارد.")

#streamlit run "D:\\10project Aiolearn\\Cancer\\lung_cancer_app.py"