# NLP-Sentiment-Analysis-Projects

## 📋 Project Overview
This repository contains two comprehensive sentiment analysis projects on Twitter data, completed as academic coursework. The projects demonstrate different approaches to sentiment analysis:

1.  **Project 1:** Lexicon-based sentiment analysis using **TextBlob** for polarity and subjectivity scoring
2.  **Project 2:** Machine Learning classification using **SVM, Naive Bayes, and Logistic Regression** with multiple feature extraction techniques

The primary goal is to compare a simple, rule-based approach (TextBlob) with sophisticated statistical Machine Learning models for sentiment classification tasks.

## Projects in Detail

### 🔍 Project 1: Sentiment Analysis with TextBlob
- **Objective:** To classify tweets as Positive, Negative, or Neutral based on sentiment polarity.
- **Key Steps:**
  - Comprehensive text pre-processing (lowercasing, stopword removal, lemmatization, etc.).
  - Polarity and Subjectivity calculation using TextBlob.
  - Visualization using Word Clouds and Bar Charts.
- **Results:** The analysis of 5,642 tweets showed:
  - **44.1% Positive**
  - **48.4% Neutral**
  - **7.4% Negative**

### 🤖 Project 2: ML Model Classification (Global Warming Belief Dataset)
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

## How to Run the Code

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Twitter-Sentiment-Analysis-ML-vs-TextBlob.git
    cd Twitter-Sentiment-Analysis-ML-vs-TextBlob
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Jupyter Notebooks:**
    Launch Jupyter Lab and open the notebooks in the `notebooks/` directory.
    ```bash
    jupyter lab
    ```

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, TextBlob, Matplotlib, Seaborn, WordCloud

## Key Insights & Conclusion
- **TextBlob** is excellent for a quick, interpretable sentiment analysis without needing labeled data.
- **Machine Learning models** (especially Logistic Regression) provide more power and accuracy for specific classification tasks but require a labeled dataset.
- The choice between the two methods depends on the project's goal: quick insights (TextBlob) vs. high-accuracy classification (ML Models).

## Author
**Your Name**
- LinkedIn: https://www.linkedin.com/in/muhammad-arhum01/
- Email: muhammadarhum277@gmail.com
