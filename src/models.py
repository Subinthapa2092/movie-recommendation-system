import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error
)


def run_baseline(X_train, X_test, y_train, y_test):
    majority = int(y_train.mode()[0])
    baseline_preds = np.full(len(y_test), majority)

    print('=== Baseline (Majority Class) ===')
    print(f'Test RMSE : {mean_squared_error(y_test, baseline_preds)**0.5:.4f}')
    print(f'Test MAE  : {mean_absolute_error(y_test, baseline_preds):.4f}')
    print(f'Accuracy  : {accuracy_score(y_test, baseline_preds):.2%}')
    print(f'F1 Score  : {f1_score(y_test, baseline_preds, zero_division=0):.4f}')
    return baseline_preds


def run_logistic(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds       = lr.predict(X_test)
    lr_train_preds = lr.predict(X_train)

    print('=== Logistic Regression ===')
    print(f'Train RMSE : {mean_squared_error(y_train, lr_train_preds)**0.5:.4f}')
    print(f'Test RMSE  : {mean_squared_error(y_test,  lr_preds)**0.5:.4f}')
    print(f'Test MAE   : {mean_absolute_error(y_test,  lr_preds):.4f}')
    print(f'Accuracy   : {accuracy_score(y_test, lr_preds):.2%}')
    print(f'F1 Score   : {f1_score(y_test, lr_preds):.4f}')
    print()
    print(classification_report(y_test, lr_preds, target_names=['Not Liked', 'Liked']))
    return lr, lr_preds, lr_train_preds


def run_decision_tree(X_train, X_test, y_train, y_test, feature_cols):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_preds       = dt.predict(X_test)
    dt_train_preds = dt.predict(X_train)

    print('=== Decision Tree ===')
    print(f'Train RMSE : {mean_squared_error(y_train, dt_train_preds)**0.5:.4f}')
    print(f'Test RMSE  : {mean_squared_error(y_test,  dt_preds)**0.5:.4f}')
    print(f'Test MAE   : {mean_absolute_error(y_test,  dt_preds):.4f}')
    print(f'Accuracy   : {accuracy_score(y_test, dt_preds):.2%}')
    print(f'F1 Score   : {f1_score(y_test, dt_preds):.4f}')
    print()
    print(classification_report(y_test, dt_preds, target_names=['Not Liked', 'Liked']))

    feat_imp = pd.Series(dt.feature_importances_, index=feature_cols) \
                 .sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 4))
    feat_imp.plot(kind='bar', color='seagreen', edgecolor='black')
    plt.title('Decision Tree — Top 10 Feature Importances')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.close()

    return dt, dt_preds, dt_train_preds


def run_naive_bayes(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_preds       = nb.predict(X_test)
    nb_train_preds = nb.predict(X_train)

    print('=== Naive Bayes ===')
    print(f'Train RMSE : {mean_squared_error(y_train, nb_train_preds)**0.5:.4f}')
    print(f'Test RMSE  : {mean_squared_error(y_test,  nb_preds)**0.5:.4f}')
    print(f'Test MAE   : {mean_absolute_error(y_test,  nb_preds):.4f}')
    print(f'Accuracy   : {accuracy_score(y_test, nb_preds):.2%}')
    print(f'F1 Score   : {f1_score(y_test, nb_preds):.4f}')
    print()
    print(classification_report(y_test, nb_preds, target_names=['Not Liked', 'Liked']))

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, nb_preds), annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Not Liked', 'Liked'], yticklabels=['Not Liked', 'Liked'])
    plt.title('Naive Bayes — Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
    plt.close()

    return nb, nb_preds, nb_train_preds


def compare_models(y_train, y_test,
                   baseline_preds,
                   lr_preds, lr_train_preds,
                   dt_preds, dt_train_preds,
                   nb_preds, nb_train_preds):

    results = pd.DataFrame({
        'Model'      : ['Baseline', 'Logistic Regression', 'Decision Tree', 'Naive Bayes'],
        'Train RMSE' : ['-',
            f'{mean_squared_error(y_train, lr_train_preds)**0.5:.4f}',
            f'{mean_squared_error(y_train, dt_train_preds)**0.5:.4f}',
            f'{mean_squared_error(y_train, nb_train_preds)**0.5:.4f}'],
        'Test RMSE'  : [
            f'{mean_squared_error(y_test, baseline_preds)**0.5:.4f}',
            f'{mean_squared_error(y_test, lr_preds)**0.5:.4f}',
            f'{mean_squared_error(y_test, dt_preds)**0.5:.4f}',
            f'{mean_squared_error(y_test, nb_preds)**0.5:.4f}'],
        'Test MAE'   : [
            f'{mean_absolute_error(y_test, baseline_preds):.4f}',
            f'{mean_absolute_error(y_test, lr_preds):.4f}',
            f'{mean_absolute_error(y_test, dt_preds):.4f}',
            f'{mean_absolute_error(y_test, nb_preds):.4f}'],
        'Accuracy'   : [
            f'{accuracy_score(y_test, baseline_preds):.2%}',
            f'{accuracy_score(y_test, lr_preds):.2%}',
            f'{accuracy_score(y_test, dt_preds):.2%}',
            f'{accuracy_score(y_test, nb_preds):.2%}'],
        'F1 Score'   : [
            f'{f1_score(y_test, baseline_preds, zero_division=0):.4f}',
            f'{f1_score(y_test, lr_preds):.4f}',
            f'{f1_score(y_test, dt_preds):.4f}',
            f'{f1_score(y_test, nb_preds):.4f}'],
    })

    print('=== Final Model Comparison ===')
    print(results.to_string(index=False))

    accs = [accuracy_score(y_test, p) for p in [baseline_preds, lr_preds, dt_preds, nb_preds]]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(results['Model'], accs,
                   color=['gray', 'steelblue', 'seagreen', 'coral'], edgecolor='black')
    plt.axhline(y=accs[0], color='red', linestyle='--', label='Baseline')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for bar, val in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.2%}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()
    plt.close()

    return results
