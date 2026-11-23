# Baselines on the held-out test set

- Train: 20,400 complaints. Test: 3,600 (most recent).
- **Product accuracy (TF-IDF + LR):** 0.829 (majority class: 0.056)
- **Monetary relief base rate in test:** 0.071
- **Monetary relief (TF-IDF + LR, balanced):** {'precision': 0.237, 'recall': 0.611, 'f1': 0.342}

```
                                                         precision    recall  f1-score   support

                            Checking or savings account       0.80      0.74      0.77       621
                                            Credit card       0.86      0.81      0.83       641
                                        Debt collection       0.88      0.87      0.88       636
     Money transfer, virtual currency, or money service       0.70      0.81      0.75       415
                                               Mortgage       0.95      0.93      0.94       375
Payday loan, title loan, personal loan, or advance loan       0.70      0.79      0.74       365
                                           Student loan       0.95      0.89      0.92       200
                                  Vehicle loan or lease       0.86      0.86      0.86       347

                                               accuracy                           0.83      3600
                                              macro avg       0.84      0.84      0.84      3600
                                           weighted avg       0.83      0.83      0.83      3600

```
