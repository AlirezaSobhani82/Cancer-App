# 🫁 Lung Cancer Prediction App | اپلیکیشن پیش‌بینی سرطان ریه

این پروژه با استفاده از یادگیری ماشین و Streamlit طراحی شده تا احتمال ابتلا به سرطان ریه را بر اساس علائم اولیه پیش‌بینی کند.

## 📌 ویژگی‌ها
- پیش‌بینی احتمال سرطان ریه با مدل Random Forest
- آموزش مدل با داده‌های متوازن (Oversampling)
- رابط کاربری ساده و فارسی با Streamlit
- ذخیره‌سازی مدل و اسکیلر با joblib

## 🚀 نحوه اجرا

### 1. نصب وابستگی‌ها

pip install streamlit scikit-learn pandas joblib

### 2. آموزش مدل
python model/train_model.py

### 3. اجرای اپلیکیشن
streamlit run app/lung_cancer_app.py

```bash
