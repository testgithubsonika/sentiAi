from attack_classifier import XGBoostAttackClassifier

clf = XGBoostAttackClassifier.load("models/xgb_attack_classifier.joblib")

print("Loaded successfully")