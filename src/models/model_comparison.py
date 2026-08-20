"""
model_comparison.py
2주차 산출물 — XGBoost / LightGBM / RandomForest 성능 비교

기존 data_processed/의 X_train, X_test, y_train, y_test를 그대로 사용.
결과는 표로 출력되고 CSV로도 저장되어 README/대시보드에 바로 붙일 수 있다.

실행:
    python src/model_comparison.py
"""

import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_DIR = "data_processed"
OUTPUT_PATH = "data_processed/model_comparison_result.csv"


def load_data():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def get_models():
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42,
        ),
    }


def evaluate(model, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")

    return {
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "train_time_sec": round(train_time, 2),
    }, confusion_matrix(y_test, pred)


def run_comparison():
    X_train, X_test, y_train, y_test = load_data()
    models = get_models()

    rows = []
    for name, model in models.items():
        print(f"[학습 중] {name} ...")
        metrics, cm = evaluate(model, X_train, X_test, y_train, y_test)
        metrics["model"] = name
        rows.append(metrics)
        print(f"  → accuracy={metrics['accuracy']}, f1={metrics['f1_weighted']}, "
              f"time={metrics['train_time_sec']}s")
        print(f"  confusion matrix:\n{cm}\n")

    result_df = pd.DataFrame(rows)[["model", "accuracy", "f1_weighted", "train_time_sec"]]
    result_df = result_df.sort_values("f1_weighted", ascending=False)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=== 최종 비교 결과 ===")
    print(result_df.to_string(index=False))
    print(f"\n결과 저장 위치: {OUTPUT_PATH}")
    return result_df


if __name__ == "__main__":
    run_comparison()
