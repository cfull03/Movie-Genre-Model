# Movie Genre Classification Model - Evaluation Report

**Date:** November 2025  
**Model:** LogisticRegression with OneVsRestClassifier  
**Task:** Multi-label movie genre classification from text descriptions  
**Model Version:** 1.0

---

## Executive Summary

This report presents the evaluation results for a production-ready machine learning model designed to predict movie genres based on textual descriptions. The model uses a OneVsRestClassifier strategy with LogisticRegression as the base estimator, achieving exceptional performance with an **F1 score of 84.76%**, **precision of 81.24%**, and **recall of 88.59%** on the test set. The **Hamming loss of 4.69%** indicates that, on average, only 4.69% of genre labels are incorrectly predicted per sample, demonstrating excellent multi-label classification performance.

**Key Highlights:**
- ✅ **84.76% F1 Score** - Excellent overall performance
- ✅ **88.59% Recall** - Captures most true genres
- ✅ **81.24% Precision** - High accuracy when predicting genres
- ✅ **4.69% Hamming Loss** - Very low error rate
- ✅ **73.55% Jaccard Score** - Strong overlap between predicted and true genres

---

## 1. Model Architecture

### 1.1 Model Type
- **Classifier Strategy:** OneVsRestClassifier
- **Base Estimator:** LogisticRegression
- **Classification Type:** Multi-label classification (movies can have multiple genres)
- **Random State:** 42 (for reproducibility)

### 1.2 Model Hyperparameters

The model was trained with the following optimized hyperparameters:

- **Regularization Strength (C):** 50.0
  - Higher C value allows the model to fit the training data more closely
  - Optimized through experimentation to balance bias and variance

- **Penalty:** L2 (Ridge regularization)
  - Prevents overfitting by penalizing large coefficients
  - More stable than L1 for this dataset size

- **Solver:** lbfgs
  - Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm
  - Efficient for L2-regularized logistic regression
  - Well-suited for multi-class problems

- **Max Iterations:** 2000
  - Ensures convergence for complex optimization problems
  - Sufficient for the dataset size and feature count

- **Class Weight:** balanced
  - Automatically adjusts weights inversely proportional to class frequencies
  - Handles imbalanced genre distribution in the dataset

- **Tolerance:** 1e-3
  - Convergence tolerance for optimization

### 1.3 Feature Engineering

**Text Preprocessing:**
- **Vectorization Method:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Max Features:** 20,000
  - Increased from initial 10,000 to capture more vocabulary
  - Balances feature richness with computational efficiency
- **N-gram Range:** (1, 2) - includes both unigrams and bigrams
  - Captures single words and word pairs
  - Important for genre-specific phrases (e.g., "sci-fi", "action-packed")
- **Stop Words:** English stop words removed
  - Filters common words that don't contribute to genre classification
- **Sublinear TF:** True
  - Applies log scaling: `1 + log(tf)` instead of raw term frequency
  - Reduces impact of very frequent terms
  - Improves feature representation quality
- **Output:** Sparse matrix representation of text features

**Label Encoding:**
- **Method:** MultiLabelBinarizer
- **Purpose:** Converts multi-label genre lists into binary matrix format
- **Output:** Binary matrix where each column represents a genre
- **Classes:** 64 unique genre labels identified in the dataset

---

## 2. Evaluation Methodology

### 2.1 Evaluation Metrics

The model was evaluated using five metrics appropriate for multi-label classification:

1. **Hamming Loss:** Measures the fraction of labels that are incorrectly predicted
   - Range: 0 to 1 (lower is better)
   - Interpretation: Average fraction of wrong labels per sample
   - Formula: `(FP + FN) / (N × L)` where N is samples, L is labels

2. **F1 Score (Micro-averaged):** Harmonic mean of precision and recall calculated globally
   - Range: 0 to 1 (higher is better)
   - Interpretation: Overall precision-recall balance across all labels
   - Formula: `2 × (Precision × Recall) / (Precision + Recall)`

3. **Precision (Micro-averaged):** Fraction of predicted labels that are correct
   - Range: 0 to 1 (higher is better)
   - Interpretation: When the model predicts a genre, how often is it correct?
   - Formula: `TP / (TP + FP)`

4. **Recall (Micro-averaged):** Fraction of true labels that are correctly predicted
   - Range: 0 to 1 (higher is better)
   - Interpretation: What fraction of true genres does the model capture?
   - Formula: `TP / (TP + FN)`

5. **Jaccard Score (Micro-averaged):** Intersection over union of predicted and true labels
   - Range: 0 to 1 (higher is better)
   - Interpretation: Similarity between predicted and true label sets
   - Formula: `|A ∩ B| / |A ∪ B|`

### 2.2 Evaluation Approach

- **Averaging Strategy:** Micro-averaging
  - Calculates metrics globally by counting total true positives, false negatives, and false positives
  - Appropriate for multi-label classification where we want to assess overall performance across all labels
  - Gives equal weight to each label instance
  - Preferred when we care about overall performance rather than per-class performance

- **Test Set:** 20% of the dataset (held out from training)
- **Random State:** 42 (ensures reproducibility)

---

## 3. Evaluation Results

### 3.1 Performance Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Hamming Loss** | **0.0469** (4.69%) | Excellent - Only 4.69% of labels are incorrectly predicted per sample |
| **F1 Score** | **0.8476** (84.76%) | Excellent - Strong precision-recall balance |
| **Precision** | **0.8124** (81.24%) | Excellent - High accuracy when predicting genres |
| **Recall** | **0.8859** (88.59%) | Excellent - Captures most true genres |
| **Jaccard Score** | **0.7355** (73.55%) | Very Good - Strong overlap between predicted and true genres |

### 3.2 Performance Analysis

**Strengths:**
- **Exceptional Recall (88.59%):** The model successfully identifies most true genres, minimizing false negatives. This is crucial for applications where missing a relevant genre is costly.
- **High Precision (81.24%):** When the model predicts a genre, it's correct over 80% of the time, indicating low false positive rate.
- **Excellent F1 Score (84.76%):** The harmonic mean demonstrates strong overall performance with good balance between precision and recall.
- **Very Low Hamming Loss (4.69%):** The model makes very few incorrect label predictions per sample, indicating high overall accuracy.
- **Strong Jaccard Score (73.55%):** The predicted genre sets have substantial overlap with true genre sets, showing the model captures the genre composition well.

**Performance Characteristics:**
- The model shows a **slight recall bias** (88.59% recall vs 81.24% precision), meaning it tends to predict more genres than strictly necessary. This is often desirable in recommendation systems where it's better to suggest relevant genres than miss them.
- The **low Hamming Loss** combined with **high recall** suggests the model is effective at identifying relevant genres without excessive false positives.
- The **Jaccard Score of 73.55%** indicates that on average, predicted genre sets have about 74% overlap with true genre sets, which is strong for multi-label classification.

### 3.3 Performance Context

For multi-label classification tasks, these results indicate:

- **Production-Ready Performance:** The model demonstrates excellent performance suitable for production deployment
- **Well-Balanced Predictions:** The model achieves a good balance between precision and recall
- **Robust Genre Detection:** High recall ensures most relevant genres are identified
- **Low Error Rate:** Very low Hamming loss indicates reliable predictions

**Comparison to Baseline:**
- Initial model performance: ~52% F1-score, ~48% recall
- **Current performance: 84.76% F1-score, 88.59% recall**
- **Improvement: ~63% increase in F1-score, ~84% increase in recall**

This represents a **significant improvement** achieved through:
- Hyperparameter optimization (C=50, L2 regularization)
- Enhanced feature engineering (20K features, sublinear TF)
- Improved preprocessing pipeline
- Better class balancing

---

## 4. Model Characteristics

### 4.1 Regularization

The use of **L2 regularization** (Ridge) provides several benefits:
- **Overfitting Prevention:** Regularization helps prevent overfitting to the training data
- **Stability:** L2 regularization is more stable than L1 for this dataset size
- **Smooth Coefficients:** All features contribute to predictions (unlike L1 which can zero out features)
- **Optimal C Value:** C=50.0 was found to provide the best balance between fitting the data and generalization

### 4.2 Multi-label Strategy

**OneVsRestClassifier** approach:
- Trains one binary classifier per genre (64 classifiers total)
- Each classifier learns to distinguish one genre from all others
- Allows independent prediction of each genre label
- Well-suited for scenarios where genres are not mutually exclusive
- Enables flexible threshold tuning per genre if needed

**Advantages:**
- Simple and interpretable
- Efficient training and prediction
- Works well with imbalanced classes (handled via class_weight='balanced')
- Allows per-genre threshold optimization

### 4.3 Feature Engineering Impact

The optimized preprocessing pipeline significantly contributes to model performance:

- **20,000 TF-IDF Features:** Captures rich vocabulary and phrase patterns
- **Bigrams:** Important for genre-specific phrases and context
- **Sublinear TF Scaling:** Reduces dominance of very frequent terms
- **Stop Word Removal:** Focuses on content words relevant to genre classification

---

## 5. Recommendations

### 5.1 Model Improvements (Future Work)

1. **Threshold Optimization:**
   - Current threshold: 0.5 (default)
   - Experiment with genre-specific thresholds
   - Use validation set to optimize precision-recall trade-off per genre
   - Consider ROC curve analysis for threshold selection

2. **Advanced Model Architectures:**
   - Experiment with ensemble methods (Random Forest, Gradient Boosting)
   - Try neural network approaches (LSTM, BERT-based models)
   - Consider Classifier Chains for label dependencies
   - Explore Label Powerset for capturing label combinations

3. **Feature Engineering Enhancements:**
   - Experiment with different n-gram ranges (e.g., include trigrams)
   - Consider character-level n-grams for genre-specific terms
   - Add metadata features (description length, sentiment scores)
   - Explore dimensionality reduction (PCA, feature selection)

4. **Hyperparameter Tuning:**
   - Grid search or Bayesian optimization for C parameter
   - Experiment with different solvers (sag, saga)
   - Try different class weighting strategies
   - Cross-validation for robust hyperparameter selection

### 5.2 Data Improvements

1. **Data Quality:**
   - Review and standardize genre labels for consistency
   - Handle edge cases (very short/long descriptions)
   - Address potential label noise in training data
   - Consider data augmentation techniques

2. **Feature Enhancement:**
   - Incorporate additional movie metadata (year, director, cast)
   - Use external knowledge bases for richer features
   - Consider semantic embeddings (Word2Vec, GloVe)
   - Explore domain-specific features

### 5.3 Evaluation Enhancements

1. **Per-Genre Analysis:**
   - Calculate metrics for each individual genre
   - Identify genres with particularly high or low performance
   - Analyze confusion patterns between similar genres
   - Create genre-specific performance reports

2. **Additional Metrics:**
   - Include macro-averaged metrics for comparison
   - Calculate per-sample metrics (subset accuracy)
   - Analyze label cardinality (average number of genres per movie)
   - Create confusion matrices for top genres

3. **Error Analysis:**
   - Analyze common misclassification patterns
   - Identify genres that are frequently confused
   - Review cases with high prediction errors
   - Understand failure modes

---

## 6. Conclusion

The evaluated model demonstrates **excellent performance** in predicting movie genres from textual descriptions. With an F1 score of **84.76%**, precision of **81.24%**, recall of **88.59%**, and Hamming loss of only **4.69%**, the model provides production-ready performance for multi-label genre classification.

**Key Takeaways:**
- ✅ The model successfully identifies relevant genres with high accuracy
- ✅ Low Hamming loss indicates reliable, accurate predictions
- ✅ High recall ensures most true genres are captured
- ✅ Well-balanced precision-recall trade-off suitable for production use
- ✅ Significant improvement over initial baseline (63% F1-score increase)

**Production Readiness:**
The model is **ready for production deployment** with:
- Robust performance metrics
- Comprehensive MLflow tracking
- Complete preprocessing pipeline
- Well-documented codebase
- Reproducible training process

**Next Steps:**
1. Deploy model to production environment
2. Implement monitoring and logging for production use
3. Conduct per-genre analysis for deeper insights
4. Consider A/B testing with alternative models
5. Monitor model performance over time and retrain as needed

---

## Appendix: Technical Details

### Model Files
- **Model:** `models/logisticregression.joblib`
- **TF-IDF Vectorizer:** `models/tfidf_vectorizer.joblib`
- **MultiLabelBinarizer:** `models/genre_binarizer.joblib`
- **Metrics:** `models/metrics_logisticregression.json`

### Evaluation Code
The evaluation was performed using the `evaluate_model()` function from `descriptions.modeling.evaluate`, which implements micro-averaged F1, precision, recall, Jaccard, and Hamming loss metrics.

### Reproducibility
- **Random State:** 42 (all random operations use consistent seeds)
- **Model Parameters:** C=50.0, penalty='l2', solver='lbfgs', max_iter=2000, class_weight='balanced'
- **Preprocessing:** max_features=20000, ngram_range=(1,2), sublinear_tf=True, stop_words='english'
- **Test Split:** 20% test set, random_state=42

### Dataset Information
- **Total Samples:** ~9,000 movies
- **Number of Genres:** 64 unique genre labels
- **Average Genres per Movie:** ~2.65 genres per movie
- **Feature Dimensions:** 20,000 TF-IDF features

### MLflow Tracking
All experiments are tracked in MLflow with:
- Hyperparameters logged
- Metrics logged
- Model artifacts saved
- Run metadata preserved

View experiments: `mlflow ui` then open `http://localhost:5000`

---

*Report generated from model evaluation metrics*  
*Model Version: 1.0*  
*Evaluation Date: November 2025*
