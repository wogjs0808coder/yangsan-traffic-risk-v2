"""
shap_explainer.py
3주차 산출물 — SHAP 기반 설명가능성 (지역별, 다중분류 대응)

data_processed/{region}/xgboost_tuned_model.json (model_trainer_advanced.py 산출물)과
X_test.csv, classes.csv를 이용해 SHAP 분석을 수행한다.

이 프로젝트는 다중분류(사고유형, 지역마다 11~14개 클래스)이기 때문에 SHAP 값도
클래스별로 따로 나온다. 아래 두 가지를 만든다:
    1) 전체 요약: 모든 클래스 평균 |SHAP| 기준 변수 중요도 막대그래프
    2) 특정 클래스(기본값: 가장 많이 등장하는 유형) beeswarm plot

주의: 이 스크립트는 로컬에서 직접 검증하지 못했습니다 (개발 환경에 xgboost/shap
설치가 안 되어 있었음). 실행 중 에러가 나면 에러 메시지를 그대로 알려주세요.

사용법:
    python src/explain/shap_explainer.py --region yangsan
    python src/explain/shap_explainer.py --region seoul --target-class 7
"""

import argparse
import os

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# 한글 폰트 설정 — 기본 폰트(DejaVu Sans)에는 한글 글리프가 없어서 라벨이 깨진다.
# Windows에는 "맑은 고딕"이 기본 내장되어 있어 별도 설치 없이 바로 쓸 수 있다.
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호가 깨지는 것도 같이 방지


def load_model_and_data(region: str):
    data_dir = f"data_processed/{region}"
    model = XGBClassifier()
    model.load_model(f"{data_dir}/xgboost_tuned_model.json")

    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    classes = pd.read_csv(f"{data_dir}/classes.csv")["class_name"].tolist()
    return model, X_test, classes


def compute_shap_values(model, X_test, sample_size: int = 1000):
    """SHAP 계산은 샘플 수가 많으면 오래 걸리므로, 너무 크면 일부만 샘플링한다."""
    if len(X_test) > sample_size:
        X_sample = X_test.sample(sample_size, random_state=42)
    else:
        X_sample = X_test

    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)  # Explanation 객체
    return shap_values, X_sample


def plot_overall_importance(shap_values, X_sample, out_dir):
    """모든 클래스에 대한 평균 |SHAP| 기준 변수 중요도 (클래스 상관없이 전반적으로 중요한 변수)."""
    values = shap_values.values
    if values.ndim == 3:
        # shape: (n_samples, n_features, n_classes) -> 샘플+클래스 평균
        mean_abs = np.abs(values).mean(axis=(0, 2))
    else:
        # 일부 SHAP/모델 조합에서는 2D로 나올 수 있음 (샘플만 평균)
        mean_abs = np.abs(values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    plt.figure(figsize=(8, 6))
    top15 = importance_df.head(15)
    plt.barh(top15["feature"][::-1], top15["mean_abs_shap"][::-1])
    plt.xlabel("평균 |SHAP| (모든 사고유형 기준)")
    plt.title("전체 변수 중요도")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/overall_importance.png", dpi=150)
    plt.close()

    importance_df.to_csv(f"{out_dir}/overall_importance.csv", index=False, encoding="utf-8-sig")
    print(f"저장됨: {out_dir}/overall_importance.png, overall_importance.csv")
    return importance_df


def plot_class_summary(shap_values, classes, class_index, out_dir):
    """특정 사고유형 하나에 대한 SHAP summary plot (beeswarm)."""
    class_name = classes[class_index]

    plt.figure()
    if shap_values.values.ndim == 3:
        class_shap = shap_values[:, :, class_index]
    else:
        class_shap = shap_values  # 이미 단일 클래스 형태인 경우

    shap.plots.beeswarm(class_shap, show=False, max_display=15)
    plt.title(f"사고유형: {class_name}")
    plt.tight_layout()

    safe_name = str(class_name).replace("/", "_").replace(" ", "_")
    out_path = f"{out_dir}/class_{class_index}_{safe_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"저장됨: {out_path}")


def explain_single_prediction(model, input_row: pd.DataFrame, classes: list) -> dict:
    """app.py 대시보드용: 사용자가 입력한 조건 1건에 대해 예측된 사고유형과
    그 예측에 가장 크게 기여한 변수 top-N을 반환한다.

    app.py에서:
        from src.explain.shap_explainer import explain_single_prediction
        result = explain_single_prediction(model, user_input_df, classes)
    형태로 바로 붙일 수 있다.
    """
    pred_class_idx = int(model.predict(input_row)[0])
    pred_class_name = classes[pred_class_idx]

    explainer = shap.Explainer(model)
    shap_values = explainer(input_row)

    if shap_values.values.ndim == 3:
        class_shap = shap_values.values[0, :, pred_class_idx]
    else:
        class_shap = shap_values.values[0]

    contributions = dict(zip(input_row.columns, class_shap))
    sorted_contrib = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

    return {"predicted_class": pred_class_name, "contributions": sorted_contrib}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument(
        "--target-class", type=int, default=None,
        help="beeswarm plot을 그릴 클래스 인덱스. 생략하면 가장 많이 등장하는 클래스를 자동 선택.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=1000,
        help="SHAP 계산에 사용할 샘플 수 (전체 X_test가 크면 느려지므로 샘플링, 기본 1000)",
    )
    args = parser.parse_args()

    out_dir = f"data_processed/{args.region}/shap_outputs"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{args.region}] 모델·데이터 로딩...")
    model, X_test, classes = load_model_and_data(args.region)

    print(f"[{args.region}] SHAP 값 계산 중 (샘플 {min(len(X_test), args.sample_size)}건)...")
    shap_values, X_sample = compute_shap_values(model, X_test, args.sample_size)

    print(f"[{args.region}] 전체 변수 중요도 계산 중...")
    plot_overall_importance(shap_values, X_sample, out_dir)

    if args.target_class is None:
        y_test = pd.read_csv(f"data_processed/{args.region}/y_test.csv").squeeze()
        target_class = int(y_test.value_counts().idxmax())
    else:
        target_class = args.target_class

    print(f"[{args.region}] 클래스 {target_class}({classes[target_class]}) 요약 생성 중...")
    plot_class_summary(shap_values, classes, target_class, out_dir)

    print(f"\n완료. 결과는 {out_dir}/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()
