# 양산시 교통사고 예측 V2 🚧 (개발 중)

> V1(정적 배치 파이프라인)의 한계를 넘어, 여러 지역과 실시간 데이터를 다루는 동적 파이프라인으로 재설계하고 있습니다.
> V1 보러가기: https://github.com/wogjs0808coder/yangsan-traffic-risk
> V1 서비스 바로가기: https://fragrant-dewberry-0a3.notion.site/3c0294bdb6c580c394e1dfc48ae705a6

## V1 대비 무엇이 달라지나

| | V1 | V2 |
|---|---|---|
| 대상 지역 | 경상남도 양산시 1곳 | 서울·부산·대구·인천·대전·경남 6개 광역시도 |
| 날씨 데이터 | 로컬 CSV, 정적 | 실시간 기상 API 연동 |
| 모델 | XGBoost 단일, 기본 파라미터 | Optuna로 튜닝된 앙상블 (XGBoost + LightGBM + RandomForest) |
| 검증 방식 | 단일 train/test 분할 | Stratified 5-Fold 교차검증 |
| 설명력 | 없음 | SHAP 기반 설명가능성 (어떤 요인이 왜 영향을 줬는지) |
| 시각화 | 수치 위주 | 지도 기반 사고다발지역 시각화 (Folium) |
| 예측 대상 | 사고유형 분류 | 사고유형 분류 (지역별 개별 모델) |

## 진행 상황

- [x] 프로젝트 구조 세팅
- [x] V1 검증 로직(연월 파싱, 강수량 클리핑, 결측치 처리) 이식
- [ ] 6개 지역 데이터 병합 및 지역별 기상 매칭
- [ ] 시간 기반 파생 변수 추가 (계절, 주야)
- [ ] 모델 고도화 (Optuna 탐색 + 교차검증 + 앙상블)
- [ ] SHAP 설명가능성 적용
- [ ] 실시간 기상 API 연동
- [ ] 사고다발지역 지도 시각화
- [ ] Streamlit 배포

## 기술 스택

Python · pandas · scikit-learn · XGBoost · LightGBM · Optuna · SHAP · Folium · Streamlit

## 실행 방법

```bash
git clone https://github.com/wogjs0808coder/yangsan-traffic-risk-v2.git
cd yangsan-traffic-risk-v2
python -m venv venv
venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt
streamlit run app.py
```

## 폴더 구조

```
src/
├── data/          # 데이터 로딩, 전처리, feature engineering
├── realtime/      # 실시간 기상 API 연동
├── models/        # 모델 학습, 하이퍼파라미터 튜닝
├── explain/       # SHAP 설명가능성
└── viz/           # 지도 시각화
```

## 데이터 출처

- 교통사고 데이터: 도로교통공단 TAAS 교통사고분석시스템 (공공데이터포털)
- 기상 데이터: 기상청 기상자료개방포털
