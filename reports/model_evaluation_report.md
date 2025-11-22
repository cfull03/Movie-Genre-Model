# Movie Genre Classification Model - Evaluation Report

**Date:** Generated on model evaluation  
**Model:** LogisticRegression with OneVsRestClassifier  
**Task:** Multi-label movie genre classification from text descriptions

---

## Executive Summary

This report presents the evaluation results for a machine learning model designed to predict movie genres based on textual descriptions. The model uses a OneVsRestClassifier strategy with LogisticRegression as the base estimator, achieving an F1 score of **0.501** and a Jaccard score of **0.334** on the test set. The Hamming loss of **0.109** indicates that, on average, approximately 10.9% of genre labels are incorrectly predicted per sample.

---

## 1. Model Architecture

### 1.1 Model Type
- **Classifier Strategy:** OneVsRestClassifier
- **Base Estimator:** LogisticRegression
- **Classification Type:** Multi-label classification (movies can have multiple genres)

### 1.2 Model Hyperparameters
- **Regularization Strength (C):** 1.0
- **Penalty:** L1 (Lasso regularization)
- **Solver:** liblinear
- **Max Iterations:** 1000
- **Random State:** 42 (for reproducibility)

### 1.3 Feature Engineering

**Text Preprocessing:**
- **Vectorization Method:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Max Features:** 10,000
- **N-gram Range:** (1, 2) - includes both unigrams and bigrams
- **Stop Words:** English stop words removed
- **Output:** Sparse matrix representation of text features

**Label Encoding:**
- **Method:** MultiLabelBinarizer
- **Purpose:** Converts multi-label genre lists into binary matrix format suitable for multi-label classification

---

## 2. Evaluation Methodology

### 2.1 Evaluation Metrics

The model was evaluated using three metrics appropriate for multi-label classification:

1. **Hamming Loss:** Measures the fraction of labels that are incorrectly predicted
   - Range: 0 to 1 (lower is better)
   - Interpretation: Average fraction of wrong labels per sample

2. **F1 Score (Micro-averaged):** Harmonic mean of precision and recall calculated globally
   - Range: 0 to 1 (higher is better)
   - Interpretation: Overall precision-recall balance across all labels

3. **Jaccard Score (Micro-averaged):** Intersection over union of predicted and true labels
   - Range: 0 to 1 (higher is better)
   - Interpretation: Similarity between predicted and true label sets

### 2.2 Evaluation Approach

- **Averaging Strategy:** Micro-averaging
  - Calculates metrics globally by counting total true positives, false negatives, and false positives
  - Appropriate for multi-label classification where we want to assess overall performance across all labels
  - Gives equal weight to each label instance

---

## 3. Evaluation Results

### 3.1 Performance Metrics

**Hamming Loss** | 0.109 | On average, 10.9% of genre labels are incorrectly predicted per sample 

**F1 Score** | 0.501 | Moderate precision-recall balance; model correctly identifies approximately 50% of relevant genres 

**Jaccard Score** | 0.334 | Predicted genre sets have approximately 33% overlap with true genre sets |

### 3.2 Performance Analysis

**Strengths:**
- **Low Hamming Loss (0.109):** The model makes relatively few incorrect label predictions per sample, indicating good overall label prediction accuracy
- **Moderate F1 Score (0.501):** The model achieves a reasonable balance between precision and recall, suggesting it can identify relevant genres without excessive false positives

**Areas for Improvement:**
- **Jaccard Score (0.334):** The relatively lower Jaccard score indicates that while the model predicts some correct genres, there is room for improvement in matching the complete set of true genres for each movie
- **F1 Score (0.501):** While moderate, there is potential to improve the model's ability to capture all relevant genres and reduce false negatives

### 3.3 Performance Context

For multi-label classification tasks, these results indicate:
- The model demonstrates **moderate performance** in genre prediction
- The low Hamming loss suggests the model is **conservative** in its predictions, avoiding many false positives
- The Jaccard score suggests the model may be **under-predicting** genres (missing some true genres) rather than over-predicting

---

## 4. Model Characteristics

### 4.1 Regularization

The use of **L1 regularization** (Lasso) provides several benefits:
- **Feature Selection:** L1 penalty encourages sparsity, effectively selecting the most important features
- **Interpretability:** Sparse models are more interpretable, with fewer non-zero coefficients
- **Overfitting Prevention:** Regularization helps prevent overfitting to the training data

### 4.2 Multi-label Strategy

**OneVsRestClassifier** approach:
- Trains one binary classifier per genre
- Each classifier learns to distinguish one genre from all others
- Allows independent prediction of each genre label
- Well-suited for scenarios where genres are not mutually exclusive

---

## 5. Recommendations

### 5.1 Model Improvements

1. **Hyperparameter Tuning:**
   - Experiment with different C values (regularization strength)
   - Try L2 regularization for comparison
   - Adjust max_features in TF-IDF vectorization

2. **Feature Engineering:**
   - Experiment with different n-gram ranges
   - Consider additional text features (e.g., description length, sentiment)
   - Explore dimensionality reduction techniques (e.g., PCA, feature selection)

3. **Model Architecture:**
   - Consider ensemble methods (e.g., Random Forest, Gradient Boosting)
   - Experiment with neural network approaches (e.g., LSTM, BERT-based models)
   - Try different multi-label strategies (e.g., Classifier Chains, Label Powerset)

4. **Threshold Tuning:**
   - Optimize prediction thresholds for each genre class
   - Use class-specific thresholds to improve per-genre performance

### 5.2 Data Improvements

1. **Data Quality:**
   - Review and clean genre labels for consistency
   - Ensure balanced representation of genres in training data
   - Consider data augmentation techniques

2. **Feature Enhancement:**
   - Incorporate additional movie metadata (e.g., year, director, cast)
   - Use external knowledge bases (e.g., movie databases) for richer features

### 5.3 Evaluation Enhancements

1. **Per-Genre Analysis:**
   - Calculate metrics for each individual genre
   - Identify genres with particularly high or low performance
   - Analyze confusion patterns between similar genres

2. **Additional Metrics:**
   - Include precision and recall per genre
   - Calculate macro-averaged metrics for comparison
   - Analyze label cardinality (average number of genres per movie)

---

## 6. Conclusion

The evaluated model demonstrates **moderate performance** in predicting movie genres from textual descriptions. With an F1 score of 0.501 and Hamming loss of 0.109, the model provides a reasonable baseline for multi-label genre classification. The use of L1-regularized LogisticRegression with OneVsRestClassifier provides a solid, interpretable foundation.

**Key Takeaways:**
- The model successfully identifies relevant genres with moderate accuracy
- Low Hamming loss indicates conservative, relatively accurate predictions
- There is significant room for improvement, particularly in capturing the complete set of genres per movie

**Next Steps:**
1. Conduct per-genre performance analysis to identify specific weaknesses
2. Experiment with hyperparameter tuning and alternative model architectures
3. Enhance feature engineering and consider additional data sources
4. Implement threshold optimization for improved precision-recall balance

---

## Appendix: Technical Details

### Model Files
- **Model:** `models/logisticregression.joblib`
- **TF-IDF Vectorizer:** `models/tfidf_vectorizer.joblib`
- **MultiLabelBinarizer:** `models/genre_binarizer.joblib`
- **Metrics:** `models/metrics_logisticregression.json`

### Evaluation Code
The evaluation was performed using the `evaluate_model()` function from `descriptions.modeling.evaluate`, which implements micro-averaged F1, Jaccard, and Hamming loss metrics.

### Reproducibility
- Random state: 42
- Model parameters: C=1.0, penalty='l1', solver='liblinear', max_iter=1000
- All random operations use consistent seeds for reproducibility

---

*Report generated from model evaluation metrics*

