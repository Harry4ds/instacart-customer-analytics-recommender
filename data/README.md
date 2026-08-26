# Dataset

This project uses the **Instacart Online Grocery Basket Analysis** dataset available on Kaggle.

## Source

Kaggle dataset: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset/data

## Required Files

After downloading the dataset, place the following files inside `data/raw/`:

- `orders.csv`
- `order_products__prior.csv`
- `products.csv`
- `departments.csv`

The notebook does not require `aisles.csv` or `order_products__train.csv` for the final workflow.

## Expected Folder Structure

```text
data/
├── README.md
├── raw/
│   ├── orders.csv
│   ├── order_products__prior.csv
│   ├── products.csv
│   └── departments.csv
│
└── processed/
```

The `processed/` directory is created automatically by the notebook when required.

## Data Sampling

To make the analysis practical on local hardware, the project uses a reproducible sample of **200,000 prior orders** with `random_state=42`.

## Repository Data Policy

Raw and generated datasets are intentionally excluded from this GitHub repository.

The `.gitignore` file excludes:

- `data/raw/`
- `data/processed/`

This keeps the repository lightweight while allowing the analysis to be reproduced by downloading the original dataset from Kaggle.
