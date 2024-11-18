'''

given

Logistic Regression, SVM, Random Forest, Gradient Boosting
'''
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
data['Age'].fillna(data['Age'].mean(),inplace=True)
data['Cabin'].fillna('N',inplace=True)
data['Embarked'].fillna('N',inplace=True)

data.loc[data["Sex"] == "male", "Sex_encode"] = 0
data.loc[data["Sex"] == "female", "Sex_encode"] = 1

from sklearn.preprocessing import LabelEncoder

# Null 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 피처 제거
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

# 레이블 인코딩 수행.
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞에서 설정한 데이터 전처리 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


Y = data['Survived']
X = data.drop('Survived', axis=1)
X = transform_features(X)
X['Fare'] = np.log(X['Fare'] + 1)
X['Age'] = np.log(X['Age'] + 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, Y, \
                                                  test_size=0.2, random_state=11)

from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")  # 경고 제거

# 각 모델과 검색 공간 정의
param_spaces = {
    'LogisticRegression': {
        'C': (1e-6, 1e+6, 'log-uniform'),  # 정규화 강도
        'penalty': ['l2'],  # L2 규제
        'solver': ['lbfgs'],  # solver 선택
    },
    'SVC': {
        'C': (1e-3, 1e+3, 'log-uniform'),  # 정규화 강도
        'kernel': ['linear', 'rbf'],  # 커널 선택
        'gamma': (1e-4, 1e+1, 'log-uniform'),  # 감마 파라미터
    },
    'RandomForestClassifier': {
        'n_estimators': (10, 500),  # 트리 개수
        'max_depth': (3, 20),  # 최대 깊이
        'min_samples_split': (2, 10),  # 최소 샘플 분리
    },
    'GradientBoostingClassifier': {
        'learning_rate': (0.01, 1.0, 'log-uniform'),  # 학습률
        'n_estimators': (10, 500),  # 트리 개수
        'max_depth': (3, 20),  # 최대 깊이
    }
}

models = {
    'LogisticRegression': LogisticRegression(),
    'SVC': SVC(probability=True),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}

# 베이지안 최적화를 실행하고 결과 저장
best_models = {}
results = {}

for model_name, model in models.items():
    print(f"Optimizing {model_name}...")
    search = BayesSearchCV(
        estimator=model,
        search_spaces=param_spaces[model_name],
        n_iter=30,  # 탐색 반복 횟수
        cv=3,  # 3-Fold 교차 검증
        scoring='roc_auc',  # ROC-AUC 점수 기준
        n_jobs=-1,  # 병렬 처리
        random_state=42
    )
    search.fit(X_train, y_train)
    best_models[model_name] = search.best_estimator_
    results[model_name] = {
        'Best Params': search.best_params_,
        'Best Score': search.best_score_
    }
    print(f"Best Params for {model_name}: {search.best_params_}")
    print(f"Best ROC-AUC Score for {model_name}: {search.best_score_}")

# 최적화 결과 출력
for model_name, result in results.items():
    print(f"\nModel: {model_name}")
    print(f"Best Params: {result['Best Params']}")
    print(f"Best ROC-AUC: {result['Best Score']:.4f}")

# 최적 모델 성능 평가
print("\nEvaluating best models on test set...\n")
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else "N/A"

    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}" if roc_auc != "N/A" else "ROC-AUC: Not available")
    print("-" * 40)
