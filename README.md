# Bank Customer Profiling and Churn Risk Analysis System

This is a Streamlit-based data analysis dashboard for bank customer profiling and churn risk analysis.

## Features

- Upload bank customer CSV data
- View customer profile and basic statistics
- Segment customers into different groups
- Analyze churn risk by age, region, product number, and active status
- Train a Random Forest model to predict churn risk
- Export high-risk customer list as CSV
- Generate business suggestions for customer retention and targeted marketing

## Tech Stack

- Python
- Streamlit
- Pandas
- Matplotlib
- Scikit-learn

## Data Source

This project uses publicly available and anonymized bank customer churn data from Kaggle for learning and portfolio demonstration purposes.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py