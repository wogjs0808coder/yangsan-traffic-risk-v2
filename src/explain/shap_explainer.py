"""
shap_explainer.py
3주차 산출물 — SHAP 기반 모델 설명가능성

XGBoost는 트리 기반이라 TreeExplainer로 빠르게 계산 가능.
전체 요약(summary plot)과 개별 예측 설명(force plot)을 이미지로 저장한다.

실행:
    python src/shap_explainer.py
"""

import shap
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

MODEL_PATH = "data_processed/xgboost_traffic_model.json"
X_TEST_PATH = "data_processed/X_test.csv"
OUTPUT_DIR = "data_processed/shap_outputs"


def load_model_and_data():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    return model, X_test


def generate_summary_plot(model, X_test, max_display=15):
    """전체 변수 중요도를 SHAP 값 기준으로 시각화.
    '어떤 변수가 위험도 예측에 가장 큰 영향을 주는가'를 한눈에 보여준다.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/summary_plot.png", dpi=150)
    plt.close()
    print(f"저장됨: {OUTPUT_DIR}/summary_plot.png")
    return explainer, shap_values


def generate_single_case_explanation(explainer, X_test, row_index=0):
    """특정 케이스 1건에 대해 '왜 이 예측이 나왔는지'를 설명하는 force plot.
    대시보드에서 사용자가 조건을 입력했을 때, 그 케이스에 대한 개별 설명으로
    활용하면 좋다 (app.py의 예측 결과 아래에 삽입 추천).
    """
    shap_values = explainer.shap_values(X_test.iloc[[row_index]])

    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_test.iloc[row_index],
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/force_plot_case{row_index}.png", dpi=150)
    plt.close()
    print(f"저장됨: {OUTPUT_DIR}/force_plot_case{row_index}.png")


def explain_single_prediction_for_dashboard(model, input_row: pd.DataFrame) -> dict:
    """실시간 대시보드용: 사용자가 입력한 조건 1건에 대해
    '어떤 요인이 위험도를 얼마나 높였는지' Top-N을 dict로 반환한다.

    app.py에서:
        from shap_explainer import explain_single_prediction_for_dashboard
        result = explain_single_prediction_for_dashboard(model, user_input_df)
        st.write(result)
    형태로 바로 붙일 수 있다.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_row)

    contributions = dict(zip(input_row.columns, shap_values[0]))
    sorted_contrib = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    return sorted_contrib


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, X_test = load_model_and_data()
    explainer, shap_values = generate_summary_plot(model, X_test)
    generate_single_case_explanation(explainer, X_test, row_index=0)
