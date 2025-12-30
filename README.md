# AI/ML Internship Tasks

This repository contains AI/ML internship tasks demonstrating end-to-end machine learning workflows, including classical ML pipelines and multimodal deep learning.

---

## Task 1: Fine-tuning BERT-Uncased Model

### BERT News Topic Classifier

This project fine-tunes a BERT (`bert-base-uncased`) model to classify news headlines from the AG News dataset into 4 categories:

- World  
- Sports  
- Business  
- Sci/Tech  

A Gradio interface is provided for real-time predictions.

### Objective
Fine-tune a transformer-based model for news topic classification and deploy a lightweight interactive UI for testing headlines.

### Dataset
**AG News** dataset (Hugging Face Datasets):

- Training set: 120,000 news headlines  
- Test set: 7,600 news headlines  
- Categories: World, Sports, Business, Sci/Tech  

For fast training in this project, a subset of 8,000 training examples and 2,000 test examples is used.

### Methodology
1. Tokenization using `bert-base-uncased` tokenizer, max length 128  
2. Model: `BertForSequenceClassification` (num_labels=4)  
3. Fine-tuning:  
   - Learning rate: 2e-5  
   - Batch size: 16  
   - Epochs: 2  
   - Optimizer: AdamW with weight decay 0.01  
4. Evaluation: Accuracy and weighted F1-score  
5. Deployment: Gradio interface for live predictions

### Results
| Headline | Predicted Topic |
|----------|----------------|
| Apple launches new AI powered chip | Sci/Tech |
| Lionel Messi scores a hat-trick | Sports |
| UN holds emergency meeting | World |
| Stock markets fall sharply | Business |

---

## Task 2: End-to-End ML Pipeline (Telco Churn Prediction)

### Objective
Build a reusable and production-ready machine learning pipeline to predict customer churn using Scikit-learn.

### Dataset
**Telco Customer Churn dataset** (CSV)

- Features include demographics, account info, and services  
- Target: Churn (Yes / No)

### Implementation
- Preprocessing:
  - Scaling numeric features (StandardScaler)  
  - Encoding categorical features (OneHotEncoder)  
  - Combined via ColumnTransformer  
- Models:
  - Logistic Regression  
  - Random Forest  
- Hyperparameter tuning with GridSearchCV  
- Exported pipeline using joblib for reuse in production

### Skills Gained
- ML pipeline construction  
- Hyperparameter tuning with GridSearchCV  
- Model export and production readiness

### Files
- `task2_colab.ipynb` – Colab notebook for Task 2  
- `telco_churn_pipeline.pkl` – Exported pipeline  
- `requirements.txt` – Dependencies

---

## Task 3: Multimodal ML (Housing Price Prediction)

### Objective
Predict housing prices using both structured tabular data and house images.

### Dataset
Housing Sales dataset + images (public or custom dataset)

- Tabular features: area, bedrooms, bathrooms, location_score  
- Image features: house images

### Implementation
- CNN used to extract features from images  
- Tabular data normalized/scaled  
- Image features combined with tabular features  
- Regression model trained using both modalities  
- Evaluated using MAE and RMSE

### Skills Gained
- Multimodal machine learning  
- Convolutional Neural Networks (CNNs)  
- Feature fusion (tabular + image)  
- Regression modeling and evaluation

### Files
- `task3_multimodal_colab.ipynb` – Colab notebook for Task 3  
- `data/` – Contains CSV and images  
- `requirements.txt` – Dependencies

---

## How to Run

### Task 1
1. Open the notebook `AG_News_Classifier.ipynb` in Google Colab  
2. Run all cells to train the BERT model and launch Gradio interface  
3. Test using sample news headlines  

### Task 2
1. Open `task2_colab.ipynb` in Google Colab  
2. Upload the Telco Churn CSV file  
3. Run all cells to train models and export pipeline (`.pkl`)  

### Task 3
1. Open `task3_multimodal_colab.ipynb` in Google Colab  
2. Upload dataset CSV and image folder (if using custom images)  
3. Run all cells to train multimodal regression model  

---

## Evaluation

- **Task 1:** Accuracy, weighted F1-score, sample predictions  
- **Task 2:** Accuracy, classification report, best hyperparameters  
- **Task 3:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)  

---

## Install Dependencies
```bash
pip install -r requirements.txt
