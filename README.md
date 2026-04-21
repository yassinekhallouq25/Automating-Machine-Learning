Title

Implement Log-Loss Check for Pricing Models

Description

Add a custom model check to compute log-loss (cross-entropy loss) on both the training and test sets for pricing projects.

This check should help evaluate the quality of predicted probabilities and provide a comparison point for ElasticBoost against selected baseline models.

Objective

Measure how well the model’s predicted probabilities align with the true labels by computing log-loss on train and test data.

Why this matters

Log-loss is a useful metric for probabilistic classification because it penalizes confident wrong predictions more heavily than simple accuracy-based metrics. In pricing projects, this helps assess whether ElasticBoost produces reliable probability estimates.

Scope
Compute log-loss on the training set
Compute log-loss on the test set
Compare ElasticBoost results against one or more baseline models
Integrate the metric as a custom model check in the Deepchecks-based validation workflow
Baseline comparison

We should define a small set of baseline models to make the check meaningful. Possible baselines include:

Dummy classifier
Prior or previously used model
Any simple benchmark model already available in the project

The main goal is to compare ElasticBoost log-loss performance against these baselines and document the outcome.

Dependencies
Access to predicted probabilities (y_pred) for train and test sets
Access to true labels (y_true) for train and test sets
Integration point for adding the check under model custom checks
Definition of Done
Log-loss is computed for both training and test sets
The check is implemented as a custom model check
At least one baseline model is included for comparison
ElasticBoost performance is compared against the selected baseline(s)
The results and interpretation are documented
A decision is documented on whether an acceptance threshold is needed
Open questions
Which baseline models should be used by default?
Do we want to define an acceptable threshold for log-loss, or keep this check comparative only?
Should the check fail automatically under certain conditions, or only report results?
Risks
Baseline selection may vary across projects
Predicted probabilities may require standardization depending on model output format
Threshold definition may not generalize well across datasets







Title

Implement PR Curve, ROC Curve, and AUC Check for Pricing Models

Description

Add a custom evaluation check for pricing models that computes and visualizes Precision-Recall (PR) and ROC performance from predicted probabilities.

This issue includes two parts:

Precision-Recall Curve
Since Deepchecks does not natively provide a Precision-Recall curve check, this part must be implemented manually. The check should compute precision and recall across probability thresholds and generate the corresponding curve.
ROC Curve & AUC
Verify whether Deepchecks’ built-in checks are compatible with ElasticBoost. If not, implement ROC curve computation and AUC calculation manually as part of the custom check.

The implementation should be reusable across pricing projects and integrated into the existing model custom checks framework.

Objective

Evaluate classification performance from predicted probabilities using threshold-based diagnostics and provide both:

visual outputs through PR and ROC curves
quantitative summaries through Average Precision / PR AUC and ROC AUC
Why this matters

For pricing models, predicted probabilities often provide more insight than hard class predictions. PR and ROC analyses help assess how well the model separates classes across thresholds.

This is particularly useful when:

class distribution is imbalanced
threshold selection matters for downstream decisions
accuracy alone is not sufficient to evaluate model quality
Scope
Compute a Precision-Recall curve from predicted probabilities
Compute a ROC curve
Compute ROC AUC
Add a quantitative PR metric such as Average Precision (AP) or PR AUC
Check whether any part of this can rely on Deepchecks directly for ElasticBoost
If not, implement the required logic manually
Package the implementation in reusable code for future pricing projects
Integrate it as a custom model check
Quantitative evaluation

The PR curve should not be only visual; it should also be evaluated quantitatively.

Recommended metrics:

Average Precision (AP): preferred summary metric for Precision-Recall performance
PR AUC: acceptable if computed consistently
ROC AUC: standard summary metric for ROC performance
Recommendation

Use:

Average Precision (AP) for the PR curve
ROC AUC for the ROC curve

This provides a clear numeric summary for each evaluation.

Expected implementation details
Precision-Recall Curve
Manually compute precision and recall over probability thresholds
Plot the PR curve
Compute and document a summary metric:
preferably Average Precision
or PR AUC if that becomes the selected convention
ROC Curve
Verify whether Deepchecks supports this for ElasticBoost
If compatible, use the built-in implementation where appropriate
Otherwise:
manually compute false positive rate and true positive rate
plot the ROC curve
compute ROC AUC
Dependencies
Access to predicted probabilities (y_pred)
Access to true labels (y_true)
Ability to integrate the implementation under model custom checks
Plotting support for generating PR and ROC visualizations
Definition of Done
Precision-Recall curve is implemented manually
PR curve is plotted using predicted probabilities across thresholds
A quantitative PR metric is computed and documented
preferably Average Precision
ROC curve is computed
ROC AUC is computed and documented
Deepchecks compatibility with ElasticBoost is verified and documented
If native support is insufficient, manual implementation is completed
The code is reusable as a function or module for future pricing projects
The check is integrated into the custom model checks framework
Open questions
Should we standardize on Average Precision or PR AUC for PR evaluation?
Should this check only report metrics, or also define warning/failure thresholds?
Do we want the same plots and metrics on both train and test sets, or test only?
Risks
Deepchecks compatibility limitations with ElasticBoost may require manual implementation
PR interpretation can be sensitive to class imbalance
Threshold-based plots may be less useful if probabilities are poorly calibrated





Title

Extend Calibration Check with Threshold Support for Pricing Models

Description

Use the existing Deepchecks CalibrationScore check for model calibration analysis and adapt its implementation so it can support threshold-based evaluation for pricing projects.

The calibration visualization already exists in Deepchecks, so this mission is not about creating a new calibration plot from scratch. Instead, the goal is to update or extend the source code to make the check more actionable by adding threshold logic based on a quantitative calibration metric.

Two candidate metrics should be considered for this thresholding logic:

Expected Calibration Error (ECE)
Brier Score

The check should remain compatible with ElasticBoost and integrate into the existing model custom checks workflow.

Objective

Enhance the existing calibration check so that it not only visualizes calibration quality, but also evaluates it quantitatively against a defined threshold.

Why this matters

Calibration is important when model outputs are used as probabilities. A well-calibrated model produces predicted probabilities that match observed outcome frequencies.

For example, if a model predicts an event with probability 0.8, that event should occur about 80% of the time among similar cases. In pricing projects, this is important because poorly calibrated probabilities can lead to incorrect downstream decisions even when ranking performance appears acceptable.

A threshold-based calibration check makes the result easier to interpret and use in validation workflows.

Scope
Reuse the existing Deepchecks CalibrationScore plot
Review and modify the source code if needed to support threshold-based validation
Evaluate whether the check is fully compatible with ElasticBoost
Add logic to assess calibration quality using a numeric metric
Support thresholding on either:
ECE
Brier Score
Integrate the updated check into the model custom checks framework
Candidate metrics
Expected Calibration Error (ECE)

ECE measures the average difference between:

predicted probabilities
observed frequencies

The predictions are grouped into bins, and for each bin we compare:

the average predicted probability
the actual fraction of positive outcomes

ECE is then computed as a weighted average of these differences across bins.

Interpretation:

lower ECE means better calibration
an ECE of 0 indicates perfect calibration
Brier Score

Brier Score measures the mean squared error between:

predicted probabilities
actual binary outcomes

For binary classification, it is computed as the average of:

(
p
^
	​

i
	​

−y
i
	​

)
2

where:

p
^
	​

i
	​

 is the predicted probability
y
i
	​

 is the true label

Interpretation:

lower Brier Score means better probabilistic predictions
it captures both calibration and refinement, not calibration alone
Recommendation
Use the existing CalibrationScore plot from Deepchecks for visualization
Add threshold support on top of a quantitative metric
Prefer ECE if the goal is specifically to measure calibration quality
Consider Brier Score if a broader probability-quality metric is also useful

A final decision should be made on which metric is more appropriate for pricing validation.

Expected implementation details
Confirm whether CalibrationScore works as-is with ElasticBoost
Inspect Deepchecks source code to determine how to extend the current check
Add a configurable threshold parameter to the calibration check
Use the selected metric to evaluate whether calibration is acceptable
Keep the existing calibration curve visualization
Ensure the implementation is reusable across pricing projects
Dependencies
Access to predicted probabilities (y_pred)
Access to true labels (y_true)
Existing Deepchecks calibration check source code
Integration point under model custom checks
Decision on whether thresholding should use ECE or Brier Score
Definition of Done
Existing Deepchecks calibration plot is reused successfully
Source code is reviewed and updated to support threshold-based validation
The check works with ElasticBoost, or compatibility limitations are documented
A calibration metric is selected for thresholding
Threshold logic is implemented
Calibration score is computed and interpreted
Calibration curve remains available in the check output
The updated version is integrated into the custom model checks framework
Open questions
Should thresholding rely on ECE, Brier Score, or support both?
What threshold values should be considered acceptable?
Should the check produce a warning, a failure, or only an informational result when the threshold is exceeded?
Should the threshold be global or configurable by project?
Risks
Deepchecks internals may require non-trivial source code changes
ElasticBoost compatibility may limit direct reuse of the existing implementation
ECE depends on binning strategy, so results may vary depending on configuration
Brier Score is not a pure calibration metric, which may affect interpretation




















