# Automating-Machine-Learning
Automating Machine Learning  Testing using Github and DeepChecks

Model Check	comment
Train Test Performance	checks how performance differs between train and test, so it helps catch overfitting.
Confusion Matrix Report	very useful to see which classes your model confuses.
ROC Report	relevant for binary classification, especially if ranking quality matters.
Single Dataset Performance	good for a quick view of core classification metrics on one dataset.
Weak Segments Performance	important in pricing because the model may work overall but fail on specific customer or product segments.
Performance Bias	useful if you want to check whether performance is uneven across groups.
Performance Bias	very relevant if your classifier outputs probabilities and you use them in pricing decisions.
Prediction Drift	useful in production to detect if prediction behavior changes over time.
Model Inference Time	relevant if pricing decisions must be made quickly
Boosting Overfit	only if you are using boosting models like XGBoost, LightGBM, CatBoost.
	
Data Check	comment
Label Drift	useful even if there are no brand-new classes, because class proportions can still shift a lot.
Feature Drift	checks whether input distributions changed, which can hurt classification performance even when labels stay the same.
Multivariate Drift	stronger version of drift checking, useful when relationships between features change.
Train Test Samples Mix	checks whether train and test got mixed or contaminated.
Date Train Test Leakage Overlap / Duplicates	important if your pricing/classification data is time-based  / catch duplicate leakage across splits.
New Category if you use categorical variables	important if your pricing/classification data is time-based.
<img width="1245" height="457" alt="image" src="https://github.com/user-attachments/assets/68d474c5-7d0a-482e-be5e-454fc8d02bda" />
