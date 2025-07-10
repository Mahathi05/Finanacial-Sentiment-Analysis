# Finanacial-Sentiment-Analysis
This project is a demonstration of how Artificial Intelligence (AI) can be used to understand the sentiment (the "mood") of financial news. The goal is to automatically classify a sentence from a financial report or headline as Positive, Negative, or Neutral.

This ability is valuable for investors, traders, and financial analysts who need to quickly gauge market reactions to news without manually reading thousands of articles.

## Table of Contents
- Project Overview<br />
- The Dataset<br />
- How It Works: The Process<br />
- Tools and Technology<br />
- Results: How Well Did It Perform?

## Project Overview
I built a machine learning model that reads a sentence of financial text and predicts its sentiment. The project follows these key steps:

**1.Prepare the Data:** Cleaned and organized a dataset of financial sentences.<br />
**2.Choose a Model:** Selected a powerful, pre-trained AI model called "DistilBERT" as a starting point.<br />
**3.Train the Model (Fine-Tuning):** Taught the AI model to specialize in financial language and recognize positive, negative, and neutral tones.<br />
**4.Test the Model:** Measured how accurately the model could predict sentiment on data it had never seen before.<br />
**5.Make Predictions:** Used the final, trained model to analyze new sentences.<br />

The entire project is documented in the Jupyter Notebook: fin_sentiment_analysis.ipynb

## The Dataset
The project uses a dataset containing approximately 5,000 sentences taken from financial news and reports. Each sentence in the dataset was already labeled by a human as positive, negative, or neutral. This labeled data is crucial for teaching our AI model.<br />
- **Source:** [Financial Sentiment dataset found on Kaggle]([https://pages.github.com/](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis/data))<br />
- **Size:** ~5,000 sentences<br />
- **Labels:** positive, negative, neutral<br />

## How It Works: The Process

Building an AI model is like teaching a student a new skill. Here’s the step-by-step learning process we followed:

### 1. Data Preparation
The first step is getting the data ready for the AI. I used pandas to load the sentences and scikit-learn to convert the text labels ('positive', 'negative') into numbers (2, 0) that the model can understand. The data was then split into a training set (for teaching the model) and a testing set (for grading its performance).

### 2. Tokenization: Translating Words to Numbers
Computers don't understand words, they understand numbers. Tokenization is the process of breaking down sentences into smaller pieces (tokens) and converting them into unique numeric IDs. I used a "Tokenizer" from the Hugging Face library that was specifically designed for our AI model. This step also adds special tokens and creates an "attention mask" to help the model focus on the real words and ignore padding.

### 3. Model Training (Fine-Tuning)
Instead of building a model from scratch, which takes a huge amount of data and time, I used a technique called **transfer learning**.<br />
- **Base Model**: I started with **DistilBERT**, a powerful general-purpose language model pre-trained by Google on the entire internet. It already has a great understanding of language.
- **Fine-Tuning:** I then "fine-tuned" this model by training it further on our specific financial dataset. This teaches the model the nuances and special vocabulary of finance. The training process was managed using the **Hugging Face Trainer**, which efficiently handles the learning loop on a GPU.

### 4. Evaluation
After training, I tested the model on the testing set—data it had not seen before. This tells us how well the model generalizes to new information. We measured its performance using two key metrics:

- **Accuracy:** The percentage of predictions that were correct.
- **F1-Score:** A more advanced score that balances a model's ability to be precise with its ability to find all relevant examples, which is very useful for datasets where the categories aren't perfectly balanced.

## Tools and Technology

This project was built using industry-standard tools for data science and machine learning in Python.

**Language:** Python<br />
**Core Libraries:**
   - **Hugging Face** transformers: For accessing the DistilBERT model and the Trainer API.
   - **PyTorch:** The deep learning framework used to build and train the model.
   - **Pandas:** For loading and managing the data.
   - **Scikit-learn:** For data preprocessing and splitting.
**Environment:** Google Colab (using a GPU for accelerated training).

## Results: How Well Did It Perform?
The fine-tuned model performed well, demonstrating a strong ability to classify financial sentiment.
- **Accuracy:** 80%
- **Weighted F1-Score:** 80%
