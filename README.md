1) Checks importants sur les données brutes

Dans la partie data integrity, les checks disponibles incluent notamment IsSingleValue, SpecialCharacters, MixedNulls, MixedDataTypes, StringMismatch, DataDuplicates, StringLengthOutOfBounds, ConflictingLabels, OutlierSampleDetection, FeatureLabelCorrelation, IdentifierLabelCorrelation, FeatureFeatureCorrelation, ainsi que ClassImbalance, ColumnsInfo et PercentOfNulls. ClassImbalance vérifie la distribution de la cible, PercentOfNulls mesure le pourcentage de valeurs nulles par colonne, SpecialCharacters cherche les valeurs composées uniquement de caractères spéciaux, et OutlierSampleDetection détecte des outliers via l’algorithme LoOP.

Pour ton cas, les plus importants sur le raw data sont :

ClassImbalance
ConflictingLabels
DataDuplicates
FeatureLabelCorrelation
IdentifierLabelCorrelation
PercentOfNulls
MixedNulls
MixedDataTypes
IsSingleValue
OutlierSampleDetection
FeatureFeatureCorrelation
2) Checks importants sur train/test

La suite train_test_validation() est faite pour vérifier la qualité du split train/test, y compris distribution, fuite et intégrité. La liste des checks comprend DatasetsSizeComparison, NewLabelTrainTest, CategoryMismatchTrainTest, StringMismatchComparison, DateTrainTestLeakageDuplicates, DateTrainTestLeakageOverlap, IndexTrainTestLeakage, TrainTestSamplesMix, FeatureLabelCorrelationChange, FeatureDrift, LabelDrift et MultivariateDrift.

Ceux que je considère comme prioritaires sont :

TrainTestSamplesMix : détecte des échantillons du test qui apparaissent aussi dans le train.
IndexTrainTestLeakage : vérifie si des index de test sont présents dans le train.
FeatureDrift : mesure la dérive feature par feature entre train et test.
LabelDrift : mesure la dérive de la cible entre train et test.
MultivariateDrift : mesure une dérive globale sur tout le dataset avec un modèle entraîné à distinguer train et test.
FeatureLabelCorrelationChange : regarde si la relation feature → target change entre train et test.
CategoryMismatchTrainTest : utile si tu as des variables catégorielles avec de nouvelles modalités en test.
NewLabelTrainTest : utile si des labels apparaissent uniquement en test.
DatasetsSizeComparison : contrôle le ratio de taille train/test.
StringMismatchComparison : utile si tes catégories texte ont des variantes type "Paris" vs "paris " entre train et test.

Pour un dataset de pricing/classification, mon ordre de priorité serait :
TrainTestSamplesMix → IndexTrainTestLeakage → FeatureDrift → LabelDrift → MultivariateDrift → FeatureLabelCorrelationChange → checks catégories/chaînes.

3) Checks de performance les plus pertinents

La suite model_evaluation() sert à évaluer la performance sur différents métriques et segments, à analyser les erreurs, le surapprentissage, la comparaison à une baseline, le drift de prédiction, etc. Dans la liste de checks du modèle, la doc cite notamment RocReport, ConfusionMatrixReport, WeakSegmentPerformance, PredictionDrift, SimpleModelComparison, CalibrationScore, UnusedFeatures, BoostingOverfit et ModelInferenceTime. Le quickstart et la galerie mentionnent aussi TrainTestPerformance, SingleDatasetPerformance, SegmentPerformance, ModelInfo, PerformanceBias et MultiModelPerformanceReport.

Pour une classification, les checks les plus pertinents sont :

TrainTestPerformance : pour voir la dégradation entre train et test.
ConfusionMatrixReport : indispensable pour voir le type d’erreurs.
RocReport : Deepchecks calcule la ROC par classe en one-vs-all et marque aussi le seuil optimal via l’indice de Youden.
CalibrationScore : trace la calibration curve avec le Brier score pour chaque classe ; très important si tu utilises des probabilités dans la décision métier.
SimpleModelComparison : compare ton modèle à une baseline simple.
WeakSegmentPerformance : trouve les segments où le modèle est faible.
SegmentPerformance : utile pour analyser les perfs par segments métier.
PredictionDrift : regarde la dérive des prédictions entre deux jeux.
UnusedFeatures : repère les features peu exploitées ; la doc souligne qu’un trop grand nombre de features peut nuire à la perf et au temps d’entraînement.
ModelInferenceTime : utile si la latence compte.
4) Checks “en plus” que je trouve très pertinents

Au-delà du minimum, j’ajouterais souvent :

PerformanceBias si tu veux vérifier des écarts de performance entre sous-groupes ; la doc le présente comme utile pour les analyses de fairness et pour détecter des disparités de performance.
MultiModelPerformanceReport si tu compares plusieurs modèles ; ce check donne un résumé des scores de plusieurs modèles sur des datasets de test.
SingleDatasetPerformance si tu veux inspecter la perf sur un seul batch spécifique.
BoostingOverfit si tu utilises un modèle de boosting.
5) Ma shortlist pratique

Si je devais garder l’essentiel, je ferais :

Raw
ClassImbalance, ConflictingLabels, DataDuplicates, FeatureLabelCorrelation, IdentifierLabelCorrelation, PercentOfNulls, MixedNulls, MixedDataTypes, IsSingleValue, OutlierSampleDetection.

Train/Test
TrainTestSamplesMix, IndexTrainTestLeakage, FeatureDrift, LabelDrift, MultivariateDrift, FeatureLabelCorrelationChange, CategoryMismatchTrainTest.

Performance
TrainTestPerformance, ConfusionMatrixReport, RocReport, CalibrationScore, SimpleModelComparison, WeakSegmentPerformance, SegmentPerformance, PredictionDrift
