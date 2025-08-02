import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# بارگذاری داده‌ها
data = pd.read_csv("D:\\10project Aiolearn\\Cancer\\survey lung cancer.csv")

# تبدیل داده‌های متنی به عددی
data.GENDER = [1 if i == "M" else 0 for i in data.GENDER]
data.LUNG_CANCER = [1 if i == "YES" else 0 for i in data.LUNG_CANCER]

# تقسیم ویژگی‌ها و هدف
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# ذخیره نام ستون‌ها برای استفاده در اپلیکیشن
feature_names = X.columns.tolist()
joblib.dump(feature_names, "D:\\10project Aiolearn\\Cancer\\feature_names.pkl")

# تقسیم داده‌ها
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1)

# نرمال‌سازی
scaler = StandardScaler()
Xtrain = scaler.fit_transform(xtrain)

# آموزش مدل
model = LogisticRegression(max_iter=10000)
model.fit(Xtrain, ytrain)

# ذخیره مدل و اسکیلر
joblib.dump(model, "D:\\10project Aiolearn\\Cancer\\logistic_model.pkl")
joblib.dump(scaler, "D:\\10project Aiolearn\\Cancer\\scaler.pkl")

#python train_model.py