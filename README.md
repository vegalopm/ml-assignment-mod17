# Practical Application III: Comparing Classifiers

Link to Jupyter Notebook: https://github.com/vegalopm/ml-assignment-mod17/blob/main/prompt_III.ipynb

## Goal

The goal is to compare the performance of the classifiers (K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines) with a dataset related to marketing bank products over the telephone.

## Aproach

In order to compare the performance of classifiers KNN, Logistic Regression, Decision Tress and SVM, I will do hyperparameter tunning with grid search and changing the performance metric. To facilitate this analysis and make the code simpler, I defined two methods: `grid_search` and `plot_results`.

### grid_search

The method `grid_search` receives as parameters a dictionary with the model's name and a tuple with the default instances of the corresponding classifiers and its parameter grid that will be used to run `GridSearchCV` for hyperparameter tuning and grid search. 

The method also has `scoring` as parameter, which allows changing the performance metric. This `scoring` parameter is passed to `GridSearchCV` to find the best estimator with the corresponding performance metric.

This method returns the train time, train accuracy, test accuracy, best parameters and the best estimator.

### plot_values

Using the information about the best parameters, best threshold and best estimator, this methods output this information an it generates the following plots for each classifier:
- Confusion Matrix
- Precision-Recall
- ROC-AUC Curves

This helps as visual representation of the performance of each classifier.

### Hyperparameters to tune

These are the list of hyperparameters to tune per classifier and its justification (where needed):

#### All classifiers
* `class_weight`: (where possible) with options set to `balanced` or `{0: 1, 1: 15}`. This is because the data is heavily imbalanced (as mentioned above).
  * `balanced` automatically adjusts loss function weights inversely proportional to class frequencies. However, with this calculation, it gives a ratio of 1:8, which might not be enough.
  * `{0: 1, 1: 15}` is provided to penalize missing the 'Yes' even more than `balanced`.

#### KNN
* `n_neighbors`: A very small value overfits to noise and a very large value underfits.
* `weights`: Possible values being:
   * `uniform`: all neighbors weigh equally.
   * `distance`: closer neighbors have a stronger vote.

#### Logistic Regression
* `C`: Lower values accept more classification errors, which prevents overfitting. High valies penalize misclassifications, but risks overfitting.
* `penalty`: To test both `l1` and `l2` regularization.

#### SVM
* `C`: Same as in Logistic Regression

**NOTE:** In SVM `C` and `class_weight` are the only parameters I'm tunning because I noticed, after multiple rounds, that `kernel='rbf'` + `gamma='auto'` were always giving better results than other configurations and training SVM takes a long time with this dataset.

#### Decision Tree
* `max_depth`
* `min_samples_split`: To prevent the creation of small group of outliers.

**NOTE:** In the parameter grids in the Jupyter Notebook you might see not that many values, because with a long list of parameters, training was taking very long. I tried to keep the best params I noticed after multiple runs.


## Findings

### Default scorer

For this analysis, I did not change the performance metric and relied on the default scorer of each classifier. Given that there is an imbalance in the dataset with the number of "Yes" and "No" in customers that accepted the long-term invesment offer, I tried passing different parameters of `class_weight` to SVM and Decision Tree so this imbalance is taking in consideration during traning.

#### Default Scorer: Results

| Model              | Train Time               | Train Accuracy | Test Accuracy |
|--------------------|--------------------------|----------------|---------------|
| KNN                | 12.300 seconds           | 90.21%         | 89.91%        |
| LogisticRegression | 29.270 seconds           | 83.19%         | 82.62%        |
| SVM                | 7 minutes, 12.27 seconds | 85.26%         | 84.56%        |
| DecisionTree       | 6.176 seconds            | 87.47%         | 87.17%        |

- **KNN**: Out of the tree, it provided the best test accuracy, but by looking at the confusion matrix, it heavily misclassfied "Yes". The True Positive rate for "Yes" is 21.38%.
- **Logistic Regression**: Based on the Train accuracy and Test Accuracy results, it is not overfitted, but its score is very low (less than the baseline).  It did a better jon than KNN in classifying "Yes", with a True positive rate of 61.99% for "Yes". However, false positives for "Yes" also increased, but that is expected.
- **SVM**: Based on the Train accuracy and Test Accuracy results, it is not overfitted, but its score is very low (less than the baseline). It did a better jon than KNN in classifying "Yes", with a True positive rate of 60.36% for "Yes". However, false positives for "Yes" also increased (less than Logistic Regression), but that is expected.
- **Decision Tree**: Based on the Train accuracy and Test Accuracy results, it is not overfitted, but its score is very low (less than the baseline). It did a better job than KNN on classifying true "Yes", but performed slightly worse than SVM. The True Positive rate for "Yes" is 46.65%.

**Best True Positive score**: Logistic Regression (61.99%).

### F1 scorer

Here F1 will be used as performance metric for training. In theory, it should perform better since F1 is recommended when data classes are heavily skewed.

#### F1 Scorer: Results

| Model              | Train Time               | Train Accuracy | Test Accuracy |
|--------------------|--------------------------|----------------|---------------|
|                KNN |            5.118 seconds |         50.60% |        36.26% |
| LogisticRegression |           28.378 seconds |         45.69% |        44.50% |
|                SVM | 3 minutes, 54.07 seconds |         49.16% |        46.78% |
|       DecisionTree |            7.904 seconds |         51.05% |        47.87% |

The train accuracy and test accuracy are calculated based on F1. The true positive rates remained the same for Logistic Regression and SVM, but they improved for KNN and Decision Tree:
- KNN: True positive rate of 27.65%.
- Logistic Regression: True positive rate of 61.99%
- SVM: True positive rate of 60.36%
- Decision Tree: True positive rate of 56.37%.

**Best True Positive score**: Logistic Regression (61.99%).

### Precision scorer

Precision is used when the cost of false positives if very high. Because of this, I'm not expecting this will help for addressin my goal, but it would be good to check the results for academic purposes.

#### Precision scorer: Results

| Model              | Train Time               | Train Accuracy | Test Accuracy |
|--------------------|--------------------------|----------------|---------------|
|                KNN |            5.212 seconds |         69.21% |        62.54% |
| LogisticRegression |           29.634 seconds |         35.93% |        34.70% |
|                SVM | 7 minutes, 27.59 seconds |         40.21% |        38.18% |
|       DecisionTree |            4.149 seconds |         44.77% |        43.42% |

The results were very similar to F1:
- KNN: True positive rate of 21.81%
- Logistic Regression: True positive rate of 61.99%
- SVM: True positive rate of 60.36%
- Decision Tree: True positive rate of 46.65%

**Best True Positive score**: Logistic Regression (61.99%).

### Recall scorer

Recall is used for scoring when false negatives are costly or dangerous. What I'm testing with recall here is to see if it helps with the imbalance of true positives for "Yes" by penalizing more false negatives.

#### Recall scorer: Results

| Model              | Train Time               | Train Accuracy | Test Accuracy |
|--------------------|--------------------------|----------------|---------------|
|                KNN |            9.825 seconds |         96.31% |        28.51% |
| LogisticRegression |           28.661 seconds |         78.86% |        77.97% |
|                SVM | 5 minutes, 40.60 seconds |         86.91% |        82.07% |
|       DecisionTree |            6.814 seconds |         78.49% |        75.05% |

There were mixed results here. KNN decreased in quality. The other classifiers increased the number of false positives significantly, but it was expected. Although, the false positive rate for SVM is very high.

* KNN: Has a high train accuracy (96.31%) but a very low test accuracy (31.75%). This makes me think the model is overfit.
* Logistic Regression: True positive rate of 77.97% and a False positive rate of 38.35%.
* SVM: True positive rate of 82.07% and a False positive rate of 76.49%.
* Decision Tree: True positive rate of 75.05% and a False positive rate of 35.58%.

**Best True Positive score**: Hard to tell. The best score is SVM (82.07%), but the false positive rate is very high (76.49%). I would say the best is between Logistic Regression (77.97%) and Decision Tree (75.05%) were very close, with also very close false positive rates (38.35% and 35.58% respectively).

### PR-AUC (Area Under the Precision-Recall Curve) scorer

ROC AUC as scoring was also explored, but did not had good results because of the extreme class imbalance (because the positive class is very rare). I removed it from this report because I had already showed two scorers with terrible performance. But, while researching why ROC AUC had a terrible performance, it drived me to find PR-AUC, which is recommended for these cases. 

PR-AUC measures the trade-off between Precision and Recall across various probability thresholds and it is used when there is a heavily imbalanced dataset where the positive class is rare and finding true positives matters most.

To use it in sklearn, the scoring parameter should be set to `average_precision`.

#### PR-AUC (Area Under the Precision-Recall Curve) scorer: Results

| Model              | Train Time              | Train Accuracy | Test Accuracy |
|--------------------|-------------------------|----------------|---------------|
|                KNN |           7.566 seconds |         46.22% |        43.13% |
| LogisticRegression |          30.633 seconds |         45.69% |        42.71% |
|                SVM | 4 minutes, 1.99 seconds |         43.52% |        41.39% |
|       DecisionTree |          12.341 seconds |         45.56% |        42.23% |

KNN performed really bad here. However, Logistic Regression, SVM and Decision Tree had really good results:
- KNN: True positive rate 20.41%
- Logistic Regression: True positive rate 62.09% and false positive rate of 14.95%
- SVM: True positive rate 60.91% and false positive rate of 14.62%
- Decision Tree: True positive rate 75.16% and false positive rate of 35.59%

**Best True Positive score**: Decision Tree had a great true positive rate (75.16%), but its false positive (35.59%) is greater than SVM and Deicision Tree. Logistic Regression had a really good performance too with a true positive rate of 62.09% and a false positive rate of 14.95%.

#### Extra: Checking parameters for PR-AUC

Because PR-AUC performed the best out of all the scorers, I extracted the most relevant features for Logistic Regression, SVM and Decision Tree:
* In all of the classifiers, macroenomic indicators such as Employment Variation Rate (emp.var.rate), Euribor 3-month rate (euribor3m) and Number of Employees (nr.employed) are shown as high importance to drive down or up the acceptance rate. Employment Variation Rate and Number of Employees affect negatively the acceptance rate, while Euribor 3-month rate increases the chances of the acceptance rate.
* Month is also shown as an important indicator. With March, August and December increasing the odds of acceptance, while May, Novemeber and June reducing the odds of acceptance.

## Next Steps and Recommendations
* Recommendation: For designing a marketing campaing where the goal is to maximize the number the customers that sign-up for a long-term investment, it is important that the model maximizes identifying the customers that have more chances to sign-up (true positive). If as consequence of this maximization process, the model identifies customers as having high chance to sign-up, but in reality they would not sign-up (false positive), it's not that damaging compared to **not** reaching out to customers that would've sign-up (false negative). For that reason PR-AUC is the recommended scorer.
* Recommendation: In these kind of marketing campaigns, it's very common that the number of customers that does not accept the offer is greater than the number of customers that accept the offer. This is another reason why PR-AUC is the recommended scorer, since it measures the trade-off between Precision and Recall across various probability thresholds.
* Recommendation: Choose a Decision Tree classifier with a PR-AUC scorer for best capturing true positives (75.16%) if a 35.59% of false positive rate is not a concern. If the high false positive rate is a concern, pick a Logistic Regression with a PR-AUC scorer, which has a lower false positive rate (14.95%), if a lower true positive rate of 62.09% is acceptable.
* Next steps: Calculate how cost effective it is to choose between the performance of the Decision Tree classifier vs Logistic Regression. Does capturing more true positives outweights capturing extra false positives?
* Next steps: The next marketing campaings should be planned in the months of March, August or December for better outcome and avoid May, Novemeber and June.
* Next steps: The campaing will be more successful when the Euribor 3-month rate is higher. If Number of Employees or Employment Variation Rate are high, it decreases the odds of a successful campaing.