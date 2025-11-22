# Capstone Chapter 4: Results and Discussion

## 4.1 Introduction
This chapter presents the comprehensive results obtained from the development and implementation phases of the CropTap Hybrid Crop Recommendation System. It outlines the outcomes of the machine learning model's performance, the rule-based system's accuracy, and the integrated hybrid system's overall effectiveness. This section also delves into a thorough discussion of these results, interpreting their significance, addressing unexpected findings, and relating them back to the project's objectives and the existing body of knowledge in agricultural technology.

## 4.2 Presentation of Results

### 4.2.1 Data Characteristics and Preprocessing Outcomes
*   **Initial Data Quality:** Summarize findings from Phase 2 (Data Exploration), highlighting key distributions, missing data handling, and any significant outliers that influenced subsequent steps. Reference `notebooks/01_data_exploration.ipynb` and relevant plots (e.g., yield distribution, climate variable histograms).
*   **Feature Engineering Impact:** Present a summary of the engineered features (the 9 core features) and their distributions. Include a **correlation heatmap of the engineered features** (referencing `notebooks/feature_correlation_heatmap.png`) to illustrate relationships and identify any potential multicollinearity, discussing its implications for the chosen model.
*   **Target Variable Transformation:** Discuss the `suitability_score` creation, its distribution, and how this normalization contributed to model stability and interpretability.

### 4.2.2 Machine Learning Model Performance
*   **Model Selection:** Reiterate the chosen model (Random Forest Regressor) and the rationale based on hyperparameter tuning (GridSearchCV) and comparative validation results.
*   **Evaluation Metrics:** Present the detailed performance metrics from Phase 5 (Model Evaluation) on the unseen validation set:
    *   **R-squared (R²):** Report the value and interpret its meaning in terms of the proportion of variance explained.
    *   **Mean Absolute Error (MAE):** Report the value and explain the average magnitude of prediction errors.
    *   **Root Mean Squared Error (RMSE):** Report the value and discuss its implications for prediction accuracy, giving higher weight to larger errors.
*   **Interpretability Analysis:**
    *   **Feature Importance:** Include and discuss the `feature_importance.png` plot, highlighting which of the 9 engineered features had the most significant influence on the model's predictions. Analyze why certain features were more important than others (e.g., NPK match versus regional success).
    *   **Prediction Analysis:** Present the `predicted_vs_actual.png` scatter plot. Discuss the model's ability to generalize by comparing predicted vs. actual suitability scores, noting any patterns of over- or under-prediction and discussing potential biases.

### 4.2.3 Rule-Based System Performance
*   **Rule Set Description:** Briefly re-state the criteria and logic used in the rule-based system (e.g., strict NPK ranges, pH compatibility, climate windows).
*   **Scoring Mechanism:** Explain how a crop's suitability is scored based on the farmer's input against these rules.
*   **Performance on Test Cases:** If specific test cases were run against the rule-based system independently, present their outcomes and any observed strengths or limitations (e.g., very precise but potentially rigid).

### 4.2.4 Hybrid System Performance
*   **Integration and Weighting:** Explain how the 40% rule-based and 60% ML model scores are combined to form the final `hybrid_score`.
*   **Overall Recommendation Quality:**
    *   Present sample recommendations, showcasing the ranked list of crops, their individual rule-based, ML, and hybrid scores, and other contextual information (e.g., expected yield, risks).
    *   Discuss how the hybrid approach balances the precision of rules with the adaptability of ML, leading to more robust recommendations.
    *   Provide qualitative assessment of the recommendations' relevance and usefulness based on simulated scenarios or expert feedback (if available).

## 4.3 Discussion of Results

### 4.3.1 Interpretation of Findings
*   **Fulfillment of Objectives:** Discuss how the results align with or deviate from the project's initial objectives. Did the system achieve its goal of providing optimal crop recommendations?
*   **Strengths of the Hybrid Approach:** Elaborate on how the combination of rule-based and ML models mitigates the weaknesses of each individual approach. For instance, the rule-based system provides a grounded biological plausibility, while the ML model captures subtle, non-linear patterns.
*   **Insights from Feature Importance:** Discuss practical agricultural implications derived from feature importance. For example, if `npk_match` is highly important, it reinforces the need for accurate soil testing and nutrient management.
*   **Model Generalizability and Limitations:**
    *   Address the extent to which the model can generalize to new, unseen farmer inputs or regions.
    *   Discuss limitations, such as the reliance on historical data, potential biases in input data (e.g., limited soil test data for some regions), or the simplified nature of some features (e.g., average climate).
    *   Acknowledge the potential impact of uncaptured variables (e.g., specific pest outbreaks, unexpected climate events) not present in the training data.

### 4.3.2 Comparison with Related Work
*   **Contextualization:** Briefly compare the performance and approach of the CropTap system with existing crop recommendation systems or similar agricultural ML applications discussed in the literature review (Chapter 2).
*   **Unique Contributions:** Highlight the novel aspects or unique contributions of the CropTap system, particularly its specific hybrid architecture tailored for Filipino farming conditions.

### 4.3.3 Practical Implications and Recommendations
*   **For Farmers:** Discuss the practical benefits of the system for Filipino farmers, such as improved decision-making, potential for increased yields, and reduced risk through tailored recommendations.
*   **For Agricultural Policymakers/Researchers:** Suggest how the insights gained (e.g., feature importance, regional performance variations) could inform agricultural policy, resource allocation, or further research.
*   **Ethical Considerations:** Briefly touch upon any ethical considerations related to data privacy (if applicable), fairness of recommendations, or potential for digital divide.

## 4.4 Conclusion
Summarize the key findings and their implications, providing a concise overview of the success and limitations of the developed system. Reiterate the value proposition of the CropTap Hybrid Crop Recommendation System based on the presented results and discussion.