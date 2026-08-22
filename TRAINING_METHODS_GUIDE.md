# 학습 방법 고도화 가이드

`model_trainer_advanced.py`에 적용된 5가지 기법을 원리·코드 위치·적용 방법·주의사항 순으로 설명합니다.

---

## 1. Stratified K-Fold 교차검증

### 원리
데이터를 5등분(fold)해서, 매번 4개는 학습에 1개는 검증에 쓰는 걸 5번 반복합니다. 그리고 5번의 성능을 평균 냅니다.

기존 방식(단일 train/test 분할)의 문제는, 우연히 테스트셋에 쉬운 케이스만 몰리면 성능이 실제보다 좋게 나오고, 어려운 케이스만 몰리면 나쁘게 나온다는 점이에요. K-Fold는 이 "운"의 영향을 5번 평균으로 상쇄시킵니다.

`Stratified`가 붙은 이유: 사고 결과(예: 사망/중상/경상) 비율이 원본 데이터와 각 fold에서 동일하게 유지되도록 나눕니다. 그냥 무작위로 나누면 특정 fold에 사망사고가 하나도 안 들어가는 경우가 생길 수 있는데, 이걸 방지합니다.

### 코드 위치
```python
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
for train_idx, valid_idx in skf.split(X, y):
    ...
```
`make_xgb_objective`, `make_lgbm_objective`, `make_rf_objective` 세 함수 안에 공통으로 들어 있습니다.

### 적용 방법
- `N_SPLITS = 5`가 파일 상단에 있습니다. 데이터가 매우 적은 지역(경남처럼 3천 건대)은 `3`으로 낮춰도 됩니다 — fold 수가 많을수록 각 fold의 검증 데이터가 줄어들어 오히려 불안정해질 수 있어요.
- 그 외에는 따로 손댈 게 없습니다. Optuna 탐색 안에서 자동으로 사용됩니다.

### 결과 해석
로그에 찍히는 `CV F1(weighted) 평균`이 바로 5-fold 평균 점수입니다. 이 값과 최종 테스트 성능(`개별 모델 테스트 성능`)이 크게 차이 나면 과적합을 의심해야 합니다.

---

## 2. Optuna 하이퍼파라미터 탐색

### 원리
XGBoost 같은 모델은 `max_depth`, `learning_rate` 등 사람이 직접 정해야 하는 값(하이퍼파라미터)이 많습니다. 이걸 손으로 하나씩 바꿔가며 시도하는 대신, Optuna가 "이전 시도의 결과를 참고해서 다음엔 어떤 값을 시도하면 좋을지" 스스로 판단하며 탐색합니다 (베이지안 최적화). Grid Search처럼 모든 조합을 다 시도하지 않아도 되기 때문에, 같은 시간 대비 더 좋은 조합을 찾습니다.

### 코드 위치
```python
study = optuna.create_study(direction="maximize")
study.optimize(make_objective(X_train, y_train, sample_weight_full), n_trials=n_trials)
```
`run_pipeline()` 함수 중간에 XGBoost/LightGBM/RandomForest 각각에 대해 실행됩니다.

### 적용 방법
- 실행 시 `--trials` 값으로 탐색 횟수를 조절합니다.
  ```bash
  python src/model_trainer_advanced.py --region seoul --trials 50      # 데이터 많은 지역: 넉넉하게
  python src/model_trainer_advanced.py --region yangsan --trials 15    # 데이터 적은 지역: 적게 해도 충분
  ```
- 탐색 범위(`trial.suggest_int(...)`, `trial.suggest_float(...)`)는 각 objective 함수 안에 있습니다. 예를 들어 `max_depth`를 3~10이 아니라 더 깊게 보고 싶으면 `trial.suggest_int("max_depth", 3, 15)`처럼 범위만 바꾸면 됩니다.
- 지역마다 최적 파라미터가 다르게 나올 수 있습니다 (서울처럼 데이터가 많으면 더 복잡한 모델이, 경남처럼 적으면 단순한 모델이 유리한 경향이 있음). `best_params.json`에 지역별로 저장되니 나중에 비교해보세요.

### 결과 해석
`data_processed/{region}/best_params.json`을 열어보면 지역별로 어떤 조합이 최적이었는지 확인할 수 있습니다. 이 자체가 "지역마다 위험 패턴이 다르다"는 근거 자료로도 쓸 수 있어요 (README나 발표자료에 활용 가능).

---

## 3. Early Stopping

### 원리
XGBoost·LightGBM은 트리를 순차적으로 계속 추가하면서 성능을 개선하는 방식(부스팅)입니다. 그런데 트리를 너무 많이 추가하면 학습 데이터에만 지나치게 맞춰지는 과적합이 일어납니다. Early Stopping은 별도로 떼어둔 검증 데이터(`eval_set`)에서 성능이 `EARLY_STOP_ROUNDS`(기본 30)번 연속으로 개선되지 않으면, 그 시점에서 학습을 멈추고 가장 좋았던 시점의 모델을 사용합니다.

즉, `n_estimators=800`으로 설정해도 실제로는 200번째에서 멈출 수도 있습니다 — 사람이 정확한 숫자를 몰라도 데이터가 알아서 적정선을 찾아줍니다.

**주의**: RandomForest는 순차적으로 트리를 쌓지 않고 독립적으로 여러 트리를 만든 뒤 투표하는 방식이라 Early Stopping 개념 자체가 없습니다. 그래서 RandomForest는 Optuna가 `n_estimators`를 직접 탐색하도록 되어 있습니다.

### 코드 위치
```python
model = XGBClassifier(..., early_stopping_rounds=EARLY_STOP_ROUNDS)
model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
```
XGBoost와 LightGBM 학습 부분 모두에 들어 있습니다 (Optuna 탐색 중에도, 최종 학습 시에도 동일하게 적용).

### 적용 방법
- `EARLY_STOP_ROUNDS = 30`을 조절할 수 있습니다. 값을 늘리면(예: 50) 더 오래 참고 지켜보다가 멈추고, 줄이면(예: 15) 더 성급하게 멈춥니다. 데이터가 적은 지역은 노이즈에 민감하니 값을 좀 더 크게(40~50) 주는 게 안전합니다.
- xgboost 버전에 따라 `early_stopping_rounds`를 생성자가 아니라 `.fit()`에 넣어야 하는 구버전도 있습니다. `pip show xgboost`로 버전을 확인하고, 2.0 미만이면 `model.fit(..., early_stopping_rounds=30)` 형태로 옮겨야 합니다.

### 결과 해석
`verbose=False`로 꺼놨지만, `verbose=True`로 바꾸면 매 라운드 검증 성능이 출력되면서 실제로 몇 라운드에서 멈췄는지 볼 수 있습니다. 디버깅할 때 잠깐 켜보는 걸 추천합니다.

---

## 4. 클래스 불균형 처리

### 원리
사고 데이터는 보통 "경상사고"가 압도적으로 많고 "사망사고"는 매우 적습니다. 아무 처리 없이 학습하면 모델은 "무조건 경상사고라고 예측"해도 정확도가 높게 나오는 함정에 빠지기 쉽습니다 — 정작 가장 중요한 사망사고 예측은 못 하게 되는 거죠.

`compute_sample_weight(class_weight="balanced", y=y)`는 각 샘플에 가중치를 부여합니다. 적게 등장하는 클래스(사망사고)의 샘플일수록 더 큰 가중치를 받아서, 모델이 학습할 때 "이 샘플을 틀리면 손해가 크다"고 인식하게 만듭니다.

### 코드 위치
```python
def get_sample_weights(y):
    return compute_sample_weight(class_weight="balanced", y=y)
```
계산된 가중치는 모든 모델의 `.fit(..., sample_weight=...)` 인자로 전달됩니다.

### 적용 방법
- 지금 코드는 자동으로 전체 클래스에 균등한 중요도를 부여하는 `"balanced"` 모드입니다. 특정 클래스(예: 사망사고)에 더 강하게 가중치를 주고 싶다면, `sklearn.utils.class_weight.compute_sample_weight` 대신 직접 딕셔너리를 만들어 넘길 수 있습니다:
  ```python
  custom_weights = {"사망": 5.0, "중상": 2.0, "경상": 1.0}
  sample_weight = y.map(custom_weights).values
  ```
- 서울·부산처럼 데이터가 많은 지역은 `imbalanced-learn` 라이브러리의 `SMOTE`로 소수 클래스를 아예 새로운 샘플로 늘리는 방법도 있습니다 (지금 코드에는 포함 안 함 — 필요하면 추가해드릴게요). 데이터가 적은 지역(경남)은 SMOTE보다 지금의 가중치 방식이 더 안전합니다. 인위적으로 생성된 소수 데이터가 과도하게 많아질 위험이 있기 때문입니다.

### 결과 해석
`classification_report`에서 각 클래스별 recall(재현율)을 확인하세요. 특히 "사망" 클래스의 recall이 처리 전후로 얼마나 개선됐는지가 핵심 지표입니다. 전체 accuracy만 보면 이 개선이 잘 안 보일 수 있어요.

---

## 5. 앙상블 (Voting)

### 원리
XGBoost, LightGBM, RandomForest는 각각 조금씩 다른 방식으로 학습하기 때문에, 서로 다른 실수를 합니다. 세 모델의 예측 확률을 평균 내면(soft voting), 한 모델이 특정 케이스에서 크게 틀려도 나머지 두 모델이 보정해주는 효과가 있어 전반적으로 더 안정적인 성능을 냅니다.

### 코드 위치
```python
ensemble = VotingClassifier(
    estimators=[(k, v) for k, v in tuned_models.items()],
    voting="soft",
)
ensemble.fit(X_tr, y_tr, sample_weight=w_tr)
```

### 적용 방법
- `voting="soft"`는 각 모델의 예측 확률을 평균 냅니다. `voting="hard"`로 바꾸면 각 모델의 최종 예측(다수결)만 봅니다 — 일반적으로 soft가 더 좋은 성능을 냅니다.
- 세 모델에 동일한 가중치를 주고 있는데, 특정 모델 성능이 확실히 더 좋다면 가중치를 다르게 줄 수 있습니다:
  ```python
  ensemble = VotingClassifier(
      estimators=[...], voting="soft",
      weights=[2, 1, 1],  # xgboost에 2배 가중치
  )
  ```
- `VotingClassifier`는 `.fit()`을 호출하면 내부적으로 모델을 다시 학습합니다 (이미 학습된 모델을 그대로 재사용하지 않음). 데이터가 매우 커서 재학습이 부담스러우면, `estimators_`와 `classes_`를 직접 세팅해서 재학습을 건너뛰는 방법도 있는데, 코드가 복잡해지는 것에 비해 얻는 게 적어서 기본 방식을 유지했습니다.

### 결과 해석
로그의 마지막 부분에 개별 모델 3개와 앙상블의 f1 점수가 모두 출력됩니다. 앙상블이 항상 이기는 건 아닙니다 — 만약 개별 모델 중 하나가 앙상블보다 낫다면, 그 지역은 앙상블 없이 단일 모델을 쓰는 게 나을 수도 있습니다. `training_result_summary.json`의 `best_model` 값이 최종적으로 어떤 걸 골라야 하는지 알려줍니다.

---

## 전체 실행 순서 (지역 1개 기준)

```bash
# 1. 필요한 패키지 설치 (requirements_additions.txt에 optuna 추가되어 있음)
pip install -r requirements_additions.txt

# 2. 지역 하나 실행 (예: 경남/양산 — 데이터 적으니 trial 수 적게)
python src/model_trainer_advanced.py --region yangsan --trials 15

# 3. 데이터 많은 지역은 trial 늘려서 실행
python src/model_trainer_advanced.py --region seoul --trials 50
python src/model_trainer_advanced.py --region busan --trials 40
python src/model_trainer_advanced.py --region daegu --trials 40
python src/model_trainer_advanced.py --region incheon --trials 40
python src/model_trainer_advanced.py --region daejeon --trials 30
```

각 지역이 끝날 때마다 `data_processed/{region}/` 밑에 세 파일이 생깁니다:
- `best_params.json` — 모델별 최적 하이퍼파라미터
- `xgboost_tuned_model.json` — 튜닝된 XGBoost 모델 (SHAP 등에 재사용 가능)
- `training_result_summary.json` — 개별/앙상블 성능 비교 및 최종 추천 모델

