"""
model_trainer_advanced.py
학습 고도화 파이프라인 — 아래 5가지를 모두 적용한 버전.

    1) Stratified K-Fold 교차검증  (Optuna 탐색 내부에서 사용)
    2) Optuna 하이퍼파라미터 탐색  (XGBoost / LightGBM / RandomForest 각각)
    3) Early Stopping             (XGBoost / LightGBM — 부스팅 계열에만 적용)
    4) 클래스 불균형 처리          (compute_sample_weight, 모든 모델에 공통 적용)
    5) 앙상블(Voting)              (3개 모델 예측을 결합)

기존 model_trainer.py는 그대로 두고 이 파일을 별도 실행해서 비교하세요.
각 기법이 왜 필요한지, 어떻게 적용하는지는 docs/TRAINING_METHODS_GUIDE.md 참고.

사용법:
    python src/model_trainer_advanced.py --region seoul --trials 30
    python src/model_trainer_advanced.py --region yangsan --trials 15   # 데이터 적은 지역은 trial 수를 줄여도 무방

지역별 데이터 폴더 구조 가정:
    data_processed/{region}/X_train.csv
    data_processed/{region}/X_test.csv
    data_processed/{region}/y_train.csv
    data_processed/{region}/y_test.csv
    (없다면 load_region_data()의 경로만 실제 구조에 맞게 수정하면 됩니다)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)  # 탐색 로그가 너무 많이 찍히는 것 방지

N_SPLITS = 5
RANDOM_STATE = 42
EARLY_STOP_ROUNDS = 30


# ────────────────────────────────────────────────
# 1. 데이터 로드
# ────────────────────────────────────────────────
def load_region_data(region: str):
    base = f"data_processed/{region}"
    X_train = pd.read_csv(f"{base}/X_train.csv")
    X_test = pd.read_csv(f"{base}/X_test.csv")
    y_train = pd.read_csv(f"{base}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{base}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


# ────────────────────────────────────────────────
# 2. 클래스 불균형 처리 — 모든 모델에 공통으로 쓸 가중치 계산
# ────────────────────────────────────────────────
def get_sample_weights(y):
    """클래스 등장 빈도의 역수에 비례한 가중치를 만든다.
    예: 사망사고가 경상사고보다 10배 적으면, 사망사고 샘플 1건이
    경상사고 샘플 10건과 동일한 비중으로 학습에 반영된다.
    """
    return compute_sample_weight(class_weight="balanced", y=y)


# ────────────────────────────────────────────────
# 3. Optuna 목적함수 — 모델별 (Stratified K-Fold + Early Stopping 내장)
# ────────────────────────────────────────────────
def make_xgb_objective(X, y, sample_weight):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, valid_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
            w_tr = sample_weight[train_idx]

            model = XGBClassifier(
                **params,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                early_stopping_rounds=EARLY_STOP_ROUNDS,
            )
            model.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_val, y_val)], verbose=False,
            )
            pred = model.predict(X_val)
            scores.append(f1_score(y_val, pred, average="weighted"))

        return float(np.mean(scores))

    return objective


def make_lgbm_objective(X, y, sample_weight):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        }

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, valid_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
            w_tr = sample_weight[train_idx]

            model = LGBMClassifier(**params, random_state=RANDOM_STATE, verbosity=-1)
            model.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False)],
            )
            pred = model.predict(X_val)
            scores.append(f1_score(y_val, pred, average="weighted"))

        return float(np.mean(scores))

    return objective


def make_rf_objective(X, y, sample_weight):
    """RandomForest는 순차적으로 트리를 쌓는 부스팅 계열이 아니라
    Early Stopping 개념 자체가 없다. 대신 n_estimators를 Optuna 탐색
    범위에 포함시켜 최적값을 직접 찾는다.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, valid_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
            w_tr = sample_weight[train_idx]

            model = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            pred = model.predict(X_val)
            scores.append(f1_score(y_val, pred, average="weighted"))

        return float(np.mean(scores))

    return objective


# ────────────────────────────────────────────────
# 4. 지역 하나에 대한 전체 파이프라인
# ────────────────────────────────────────────────
def run_pipeline(region: str, n_trials: int = 30):
    print(f"\n{'='*50}\n[{region}] 학습 고도화 파이프라인 시작 (trials={n_trials})\n{'='*50}")

    X_train, X_test, y_train, y_test = load_region_data(region)
    sample_weight_full = get_sample_weights(y_train)

    # ---- Optuna 탐색 (모델별) ----
    best_params = {}
    objective_makers = {
        "xgboost": make_xgb_objective,
        "lightgbm": make_lgbm_objective,
        "random_forest": make_rf_objective,
    }

    for name, make_objective in objective_makers.items():
        print(f"\n[{name}] Optuna 탐색 중 ({n_trials}회, {N_SPLITS}-fold CV)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            make_objective(X_train, y_train, sample_weight_full),
            n_trials=n_trials,
        )
        best_params[name] = study.best_params
        print(f"  최적 파라미터: {study.best_params}")
        print(f"  CV F1(weighted) 평균: {study.best_value:.4f}")

    # ---- 최종 모델 학습 (holdout으로 Early Stopping 기준점 확보) ----
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )
    w_tr = get_sample_weights(y_tr)

    xgb_final = XGBClassifier(
        **best_params["xgboost"], eval_metric="mlogloss",
        random_state=RANDOM_STATE, early_stopping_rounds=EARLY_STOP_ROUNDS,
    )
    xgb_final.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)

    lgbm_final = LGBMClassifier(**best_params["lightgbm"], random_state=RANDOM_STATE, verbosity=-1)
    lgbm_final.fit(
        X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False)],
    )

    rf_final = RandomForestClassifier(**best_params["random_forest"], random_state=RANDOM_STATE, n_jobs=-1)
    rf_final.fit(X_tr, y_tr, sample_weight=w_tr)

    tuned_models = {"xgboost": xgb_final, "lightgbm": lgbm_final, "random_forest": rf_final}

    # ---- 개별 성능 확인 ----
    print(f"\n[{region}] 개별 모델 테스트 성능")
    individual_scores = {}
    for name, model in tuned_models.items():
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="weighted")
        individual_scores[name] = f1
        print(f"  {name}: accuracy={acc:.4f}, f1={f1:.4f}")

    # ---- 앙상블 (Soft Voting) ----
    ensemble = VotingClassifier(
        estimators=[(k, v) for k, v in tuned_models.items()],
        voting="soft",
    )
    ensemble.fit(X_tr, y_tr, sample_weight=w_tr)  # VotingClassifier는 내부적으로 재학습됨

    pred_ens = ensemble.predict(X_test)
    acc_ens = accuracy_score(y_test, pred_ens)
    f1_ens = f1_score(y_test, pred_ens, average="weighted")
    print(f"\n[{region}] 앙상블(Voting) 테스트 성능: accuracy={acc_ens:.4f}, f1={f1_ens:.4f}")
    print(classification_report(y_test, pred_ens))

    # ---- 결과 저장 ----
    out_dir = f"data_processed/{region}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    xgb_final.save_model(f"{out_dir}/xgboost_tuned_model.json")

    result_summary = {
        "region": region,
        "individual_scores": individual_scores,
        "ensemble_score": f1_ens,
        "best_model": max(
            {**individual_scores, "ensemble": f1_ens}.items(), key=lambda x: x[1]
        )[0],
    }
    with open(f"{out_dir}/training_result_summary.json", "w", encoding="utf-8") as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료:")
    print(f"  {out_dir}/best_params.json")
    print(f"  {out_dir}/xgboost_tuned_model.json")
    print(f"  {out_dir}/training_result_summary.json")
    print(f"  최고 성능 모델: {result_summary['best_model']}")

    return result_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region", required=True,
        help="지역명 (예: seoul, busan, daegu, incheon, daejeon, yangsan)",
    )
    parser.add_argument("--trials", type=int, default=30, help="Optuna 탐색 횟수 (기본 30)")
    args = parser.parse_args()

    run_pipeline(args.region, n_trials=args.trials)
