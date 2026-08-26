# Instacart Customer Analytics & Recommendation System

A customer analytics and recommendation project built using Python and the Instacart Online Grocery Basket dataset, combining **RFM segmentation, customer value estimation, K-Means clustering, ALS collaborative filtering, association rule mining, recommendation evaluation, and an interactive Streamlit dashboard**.

**Project completed:** December 2025  
<br>
**Portfolio repository published:** August 2026

---

## Project Overview

The objective of this project is to analyze customer purchasing behavior and develop a framework for customer segmentation and personalized product recommendations.

The project covers the complete analytical workflow — from data preparation and customer profiling to recommendation modeling and interactive visualization.

This repository presents my individual technical implementation. The work originally began as part of an academic capstone project, for which I led the dataset research, notebook development, and technical implementation.

---

## Business Objectives

The project aims to answer questions such as:

- Which customers are the most engaged and valuable?
- Which customers may be at risk of becoming inactive?
- Can customers be grouped based on purchasing behavior?
- Do customers show distinct product-department preferences?
- Which products can be recommended to individual customers?
- Which products are commonly purchased together?

---

## Dataset

This project uses the [Instacart Online Grocery Basket Analysis dataset](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset/data) available on Kaggle.

The raw dataset is not stored in this repository due to its size — see [`data/README.md`](data/README.md) for download and setup instructions.

---

## Data Preparation

The original Instacart transaction data is relatively large. To make the analysis practical on local hardware, a reproducible sample of **200,000 prior orders** is used, with:

```python
random_state=42
```

---

## RFM Customer Segmentation

Customer behavior is analyzed using three RFM dimensions:

- **Recency** — how recently a customer placed an order
- **Frequency** — number of unique orders placed
- **Monetary proxy** — number of products purchased

Because the Instacart dataset does not contain actual product prices, the Monetary component represents historical item count rather than financial spending.

Customers are grouped into business-oriented segments including Champions, Loyal Customers, Recent Customers, At Risk, Hibernating, and Others.

---

## Customer Lifetime Value Proxy

The Instacart dataset does not provide product prices, so this project uses a simplified customer-value proxy:

```
CLV Proxy = Historical Item Count × Assumed Average Item Price
```

An assumed average item price of **$3** is used.

The resulting CLV values should be interpreted as analytical customer-value estimates, not actual financial lifetime value.

---

## Customer Clustering

Two K-Means clustering approaches are explored.

**Behavioral Clustering** — Recency, Frequency, and Monetary proxy features are standardized and used to identify **4 behavioral customer clusters**, distinguishing customers with different purchasing patterns and engagement levels.

**Department Preference Clustering** — customer purchases are analyzed across product departments to identify **5 department-preference clusters**, providing an additional view of shopping preferences beyond traditional RFM segmentation.

---

## Recommendation System

The project explores multiple recommendation approaches.

**ALS Collaborative Filtering** — an implicit-feedback recommendation model is built using Alternating Least Squares (ALS). Customer-product interaction counts construct a sparse user-item matrix used to generate personalized product recommendations.

**Association Rule Mining** — Apriori and association-rule mining identify products that frequently occur together in customer baskets. To keep the analysis computationally manageable, this is performed on a subset of orders and the most frequently purchased products.

**Hybrid Recommendation Logic** — the analytical notebook also explores combining ALS collaborative-filtering recommendations with association-rule recommendations to produce more explainable product suggestions.

---

## Recommendation Evaluation

A time-based train/test approach is used, where each customer's latest order is treated as the test interaction. The ALS recommendation model produced the following offline results:

| Metric | Score |
|---|---:|
| Precision@5 | 0.0264 |
| Recall@5 | 0.0147 |
| Precision@10 | 0.0230 |
| Recall@10 | 0.0250 |

These results provide a baseline for evaluating recommendation performance on a large and sparse grocery-product catalog.

---

## Streamlit Dashboard

An interactive Streamlit application was developed to demonstrate the customer analytics and recommendation workflow. The dashboard allows users to:

- Filter customers by RFM segment
- Select an individual customer
- View Recency, Frequency, and Monetary metrics
- View the estimated CLV proxy
- Generate ALS product recommendations
- Compare average CLV proxy across customer segments

The Streamlit application demonstrates the ALS collaborative-filtering model. The hybrid ALS + association-rule workflow is explored separately in the analytical notebook.

---

## Technologies Used

Python, Pandas, NumPy, SciPy, Scikit-learn, Implicit ALS, MLxtend, Matplotlib, Seaborn, Streamlit, Jupyter Notebook

---

## Repository Structure

```text
instacart-customer-analytics-recommender/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── instacart_customer_analytics.ipynb
│
├── app/
│   ├── app.py
│   ├── rfm_user_details.csv
│   └── clv_segment_summary.csv
│
└── data/
    └── README.md
```

Raw data, generated datasets, and large model artifacts are excluded from version control.

---

## How to Run

Install the required Python packages:

```
pip install -r requirements.txt
```

Download the Instacart dataset and follow the setup instructions in [`data/README.md`](data/README.md).

The main analytical workflow is available in [`notebooks/instacart_customer_analytics.ipynb`](notebooks/instacart_customer_analytics.ipynb).

The Streamlit application can be launched using:

```
python -m streamlit run app/app.py
```

The ALS model artifact used by the Streamlit application is excluded from version control because of its size.

---

## Limitations

- The analysis uses a reproducible sample of 200,000 prior orders because of local memory constraints.
- Actual product prices are unavailable, so Monetary and CLV are proxy measures.
- RFM segments represent a static analysis period.
- Grocery recommendation data is highly sparse, which makes exact product prediction challenging.
- The current Streamlit application demonstrates ALS recommendations rather than the complete hybrid recommendation workflow.

---

## Future Improvements

Potential extensions include:

- Training the recommender on a larger portion of the transaction dataset
- Improving recommendations for cold-start customers
- Incorporating temporal purchasing behavior
- Adding product and category features to the recommendation model
- Improving hybrid recommendation ranking
- Incorporating actual product prices for financial CLV modeling
- Deploying the Streamlit application publicly
