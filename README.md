# Document Classification & Invoice Extraction API

## Project Overview

### This project is a Document Classification & Invoice Extraction API using NLP and deep learning. It classifies documents into categories like invoices, budgets, and emails using a DistilBERT model. If a document is classified as an invoice, the API extracts relevant fields such as invoice number, date, amount, and vendor.

## Installation & Setup

### Prerequisites

- Python 3.8+
- FastAPI
- PyTorch
- Tesseract OCR
- Transformers (Hugging Face)
- Uvicorn

## Models and Algorithms and techniques used

### 1. Document Classification

### Used DistilBERT, a transformer-based deep learning model, for text classification.

#### DistilBERT (Distilled BERT) is a lighter and faster version of BERT (Bidirectional Encoder Representations from Transformers). It is designed to retain most of BERT’s accuracy while being smaller and more efficient.

#### Key Features:
- Smaller Size: 40% fewer parameters than BERT.
- Faster Inference: 60% faster while maintaining 97% of BERT’s performance.
- Uses Knowledge Distillation: Trained by compressing knowledge from BERT.
- Retains BERT’s Bi-directional Attention: Ensures high-quality text understanding.

#### Why Use DistilBERT?
- Faster classification with reduced computational cost.
- Lower memory consumption, making it ideal for real-time applications like our API.
- Pretrained models available in Hugging Face, making it easy to fine-tune for specific tasks.

### 2. Invoice Extraction

  Regular Expressions (Regex): Extracts structured information such as invoice numbers, dates, amounts, and vendor names from text.

  Pattern Matching: Patterns are designed to identify invoice-specific terms like Invoice #, dates in various formats, and currency values.
  
  Text Preprocessing: Removes unwanted characters and normalizes text before extraction.

## Steps

### 1. Clone the repository
```bash
git clone https://github.com/your-username/document-classification-api.git cd document-classification-api
```
### 2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
### 3. Install dependencies:
```bash
pip install -r requirements.txt
```
### 4. Run the FastAPI server:
```bash
cd document-classification-api
```

## Folder Structure

![image](https://github.com/user-attachments/assets/ba314372-afd0-4a4e-bb66-7cc6d2ece80e)



## Dataset Exploration and Preprocessing

### 1. Install Kaggle
```bash
pip install kaggle
```
### 2.Get Kaggle API Key and store it in your project

![Capture](https://github.com/user-attachments/assets/2d1c9b93-dc16-4a6e-bbf1-625f8298e34a)

### 3. Download the dataset and move it into your project and explore it.

![2-explore](https://github.com/user-attachments/assets/8e8b2dea-8e4b-47ce-95a3-8874f273ea46)

### 4. Extract the data from images after labeling it as invoice ,email ,Bill etc and clean the data and prepare it for training (remove unwanted characters and symboles) using preprocess.py  and extract_text.py

 
![3-extract_text](https://github.com/user-attachments/assets/745a12f1-0247-40ae-afc3-ce20c071c110)
#### Convert Image to Text using OCR

![4-autolabel](https://github.com/user-attachments/assets/b9b0f46d-64a8-496a-8398-e3a6801c4ae6)
#### Autolabeling as Invoice , Bill , Email etc.

![5-preprocess](https://github.com/user-attachments/assets/7e1756e6-cfd1-4c43-9761-d5e12703aaa2)
 #### Preprocessing to normalise the data
 
![6-csvfile](https://github.com/user-attachments/assets/629566fc-5962-4472-8668-da476d837a49)
 #### Extracting the data to a .CSV file and  viewing it.




## Model Training (train_model.py) and Evaluation on various metrics like Accuracy, Precision ,F1 score 

- Used DistilBERTForSequenceClassification from Hugging Face.

- Tokenized text using DistilBertTokenizer.

- Trained on labeled data with PyTorch.

- Saved trained model for later inference.

![7-training(1)](https://github.com/user-attachments/assets/57d78dc7-3faa-4104-80f5-0cc0e5415af9)

![7-training(2)](https://github.com/user-attachments/assets/5235438c-3f83-4e0b-8eab-f0f491de1bb8)



## Evaluation

From the scores displayed in the above images, the following can be inferred:

### Accuracy (90.67%): The model correctly classified approximately 90.67% of the test samples. This is a strong performance, indicating that the model is well-trained.

### Precision (91.16%): Out of all the positive predictions made by the model, 91.16% were actually correct. This suggests a low false positive rate, meaning the model is reliable when it predicts a positive class.

### Recall (90.67%): The model correctly identified 90.67% of the actual positive cases. This means it is effectively capturing most of the relevant instances.

### F1 Score (90.72%): The F1 Score is a balance between precision and recall. A score of 90.72% indicates a well-balanced model that neither favors precision nor recall too much.

### The train loss (0.4387) and eval loss (0.2541) suggest that the model performed better on the validation set than the training set, possibly due to regularization techniques or a relatively small dataset.




## Invoice Data extraction (extract_invoice_data.py) and saving it to a .CSV file 

![8-3](https://github.com/user-attachments/assets/dae353c7-a7c4-4e0f-8e6e-40219ca10128)

![image](https://github.com/user-attachments/assets/2ccd65a2-ac10-4812-b6da-7a06262b9d11)


## FastAPI Integration

The trained model is integrated into a REST API built with FastAPI. The API allows users to upload a document, classify it, and extract relevant details if it's an invoice

![11](https://github.com/user-attachments/assets/c87c560a-6c81-4f98-8e12-f398e56da930)


### API Endpoints

#### 1. Home Endpoint

URL: /

Method: GET

Response:

{ "message": "Document Classification & Invoice Extraction API is Running!" }



#### 2. Document Classification & Extraction

URL: /classify_and_extract

Method: POST

Request: Upload a text file


### Test with curl
```bash
curl -X 'POST' 'http://127.0.0.1:8000/classify_and_extract' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'file=@invoice.pdf'
```


#### replace invoice.pdf with a text file of yours to test



![12](https://github.com/user-attachments/assets/165cbb5b-8385-476b-9430-59bae6b34dd6)


## Future Improvements

- Improve classification accuracy with a larger dataset.

- Deploy the API using Docker & AWS Lambda.

- Extend the extraction model to support more document types.







