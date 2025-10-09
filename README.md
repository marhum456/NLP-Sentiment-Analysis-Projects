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

### Classification Models Performance Summary

### Overview
Comparison of three machine learning classifiers (SVM, Naive Bayes, Logistic Regression) using two text vectorization techniques (TF-IDF and Bag of Words) for global warming tweet classification.

### Performance Metrics Table

| Model | Vectorization | Accuracy | Precision (Neutral) | Precision (No) | Precision (Yes) | Recall (Neutral) | Recall (No) | Recall (Yes) | F1 (Neutral) | F1 (No) | F1 (Yes) | Macro Avg F1 |
|-------|---------------|----------|---------------------|----------------|-----------------|------------------|-------------|--------------|--------------|---------|----------|-------------|
| SVM | TF-IDF | 0.670 | 0.620 | 0.640 | 0.700 | 0.520 | 0.470 | 0.830 | 0.570 | 0.540 | 0.760 | 0.623 |
| SVM | BoW | 0.670 | 0.570 | 0.660 | 0.720 | 0.580 | 0.470 | 0.780 | 0.580 | 0.550 | 0.750 | 0.627 |
| Naive Bayes | TF-IDF | 0.650 | 0.650 | 0.640 | 0.660 | 0.430 | 0.370 | 0.870 | 0.520 | 0.470 | 0.750 | 0.580 |
| Naive Bayes | BoW | 0.660 | 0.610 | 0.530 | 0.740 | 0.570 | 0.660 | 0.710 | 0.590 | 0.590 | 0.730 | 0.637 |
| Logistic Regression | TF-IDF | 0.680 | 0.650 | 0.670 | 0.700 | 0.530 | 0.470 | 0.840 | 0.580 | 0.550 | 0.770 | 0.633 |
| Logistic Regression | BoW | 0.680 | 0.600 | 0.640 | 0.730 | 0.580 | 0.490 | 0.800 | 0.590 | 0.560 | 0.760 | 0.637 |

### Key Findings

### 🏆 Best Performers by Metric

| Metric | Best Model | Score |
|--------|------------|-------|
| **Overall Accuracy** | Logistic Regression (TF-IDF & BoW) | 0.680 |
| **Best Precision (Yes)** | Naive Bayes (BoW) | 0.740 |
| **Best Recall (Yes)** | Naive Bayes (TF-IDF) | 0.870 |
| **Best F1-Score (Yes)** | Logistic Regression (TF-IDF) | 0.770 |
| **Best Macro Avg F1** | SVM (BoW), Naive Bayes (BoW), Logistic Regression (BoW) | 0.637 |

### 📊 Performance Insights

1. **Logistic Regression** consistently achieves the highest accuracy (0.68) with both vectorization methods
2. **Naive Bayes with TF-IDF** has the highest recall for "Yes" class (0.87) but suffers from class imbalance
3. **SVM with TF-IDF** shows strong performance for the "Yes" class with good balance
4. **Bag of Words** generally provides more balanced performance across classes
5. **TF-IDF** tends to perform better for the majority class ("Yes")

### 🎯 Recommendation

**For Overall Performance**: Logistic Regression with either TF-IDF or Bag of Words
**For Balanced Classes**: Naive Bayes with Bag of Words  
**For "Yes" Class Detection**: Naive Bayes with TF-IDF (if recall is priority) or Logistic Regression with TF-IDF (if F1-score is priority)

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
