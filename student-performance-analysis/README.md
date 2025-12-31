# 📊 Student Performance Analysis Project

## 🔥 Project Overview

This project analyzes student academic performance using **Python, Data Analysis, and Machine Learning**. The goal is to understand how demographic and academic factors affect student scores and to build a predictive model for student performance.

This is an **end-to-end project** covering:

* Data cleaning & preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Machine Learning prediction
* Visual report generation

---

## 🧠 Problem Statement

Educational institutions want to understand:

* What factors influence student performance?
* Does test preparation improve scores?
* Can we predict a student’s overall performance?

This project answers these questions using real-world data.

---

## 🗂️ Dataset

**Source:** Students Performance Dataset

### Columns:

* `gender`
* `race/ethnicity`
* `parental level of education`
* `lunch`
* `test preparation course`
* `math score`
* `reading score`
* `writing score`

---

## 🛠️ Tools & Technologies

* **Python**
* **Pandas** – Data manipulation
* **Matplotlib & Seaborn** – Data visualization
* **Scikit-learn** – Machine Learning
* **Random Forest Regressor** – Prediction model

---

## 📁 Project Structure

```
student-performance-analysis/
│
├── data/
│   ├── StudentsPerformance.csv
│   └── processed_students_data.csv
│
├── visuals/
│   ├── avg_score_distribution.png
│   ├── gender_vs_avg_score.png
│   ├── test_prep_vs_score.png
│   └── actual_vs_predicted.png
│
├── analysis.py
└── README.md
```

---

## 🔄 Data Processing Steps

1. Loaded raw dataset
2. Cleaned column names
3. Created a new feature `average_score`
4. Saved processed dataset for reuse

---

## 📊 Exploratory Data Analysis (EDA)

Generated visual insights:

* Distribution of average student scores
* Gender-wise performance comparison
* Impact of test preparation course

All visualizations are saved automatically in the `visuals/` folder.

---

## 🤖 Machine Learning Model

* **Model Used:** Random Forest Regressor
* **Target Variable:** Average Score
* **Features Used:**

  * Gender
  * Parental Education
  * Lunch Type
  * Test Preparation Course

### Evaluation Metrics:

* Mean Absolute Error (MAE)
* R² Score

The model predicts student performance with improved accuracy over basic linear models.

---

## 📈 Results & Insights

* Students who completed test preparation scored higher on average
* Lunch type shows correlation with academic performance
* Ensemble models outperform linear regression for this dataset

---

## ▶️ How to Run the Project

```bash
pip install pandas matplotlib seaborn scikit-learn
python analysis.py
```

---

## 🚀 Future Improvements

* Add Power BI / Tableau dashboard
* Deploy using Streamlit
* Hyperparameter tuning
* Add more ML models for comparison

---

## 👨‍💻 Author

**Santhosh T S**
Aspiring Data Analyst | SDE | UI/UX Designer

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to fork or contribute.

---

**Learning by building. Improving by iteration.**
