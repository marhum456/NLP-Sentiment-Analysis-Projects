# NLP-Sentiment-Analysis-Projects

## 📋 Project Overview
This repository contains two comprehensive sentiment analysis projects on Twitter data, completed as academic coursework. The projects demonstrate different approaches to sentiment analysis:

1.  **Project 1:** Lexicon-based sentiment analysis using **TextBlob** for polarity and subjectivity scoring
2.  **Project 2:** Machine Learning classification using **SVM, Naive Bayes, and Logistic Regression** with multiple feature extraction techniques

The primary goal is to compare a simple, rule-based approach (TextBlob) with sophisticated statistical Machine Learning models for sentiment classification tasks.

## 🔬 Projects in Detail

### 🔍 Project 1: Sentiment Analysis with TextBlob

**Objective:** Classify tweets as Positive, Negative, or Neutral using TextBlob's sentiment polarity scoring.

**Methodology:**
- **Comprehensive Text Pre-processing:**
  - Lowercasing, stopword removal, lemmatization
  - Punctuation and special character removal
  - URL and numeric value cleaning
  - Tokenization and stemming
- **Sentiment Scoring:** Calculated polarity (-1 to 1) and subjectivity (0 to 1) using TextBlob
- **Visualization:** Generated word clouds and distribution charts for each sentiment category

**Results:** Analysis of 5,642 tweets revealed:
- **44.1% Positive** sentiment
- **48.4% Neutral** sentiment  
- **7.4% Negative** sentiment

### 🤖 Project 2: ML Model Classification (Global Warming Belief)

**Objective:** Classify tweets based on belief in global warming into `Yes`, `No`, and `Neutral` categories using multiple ML algorithms.

**Technical Approach:**
- **Models Implemented:**
  - Support Vector Machine (SVM)
  - Naive Bayes (Multinomial)
  - Logistic Regression
- **Feature Extraction:**
  - TF-IDF Vectorizer
  - Count Vectorizer (Bag-of-Words)
- **Evaluation Metrics:** Precision, Recall, F1-Score, Accuracy

**Performance Summary:**
| Model | Best Accuracy | Best F1-Score (Yes) | Best Vectorizer |
|-------|---------------|---------------------|-----------------|
| Logistic Regression | **69%** | **0.77** | TF-IDF |
| SVM | 67% | 0.76 | TF-IDF |
| Naive Bayes | 66% | 0.75 | Bag-of-Words |

**Key Finding:** Logistic Regression with TF-IDF features achieved the highest overall accuracy (69%) and best performance on the "Yes" class (F1-Score: 0.77).

## 🚀 How to Run the Code

### Prerequisites
- Python 3.7+
- pip package manager

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Twitter-Sentiment-Analysis-ML-vs-TextBlob.git
   cd Twitter-Sentiment-Analysis-ML-vs-TextBlob
