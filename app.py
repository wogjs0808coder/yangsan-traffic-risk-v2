import streamlit as st

st.set_page_config(page_title="양산시 교통사고 예측 V2", page_icon="🚧")

st.title("🚧 양산시 교통사고 예측 V2 — 개발 중입니다")

st.write(
    """
    V2는 6개 지역 확장, 실시간 기상 연동, SHAP 설명가능성, 지도 시각화를
    단계적으로 추가하는 중입니다. 완성된 V1 서비스는 아래 링크에서 바로
    확인하실 수 있습니다.
    """
)

st.link_button("V1 바로가기 (완성된 서비스)", "https://yangsan-traffic-risk.streamlit.app")
st.link_button("V2 GitHub 저장소", "https://github.com/wogjs0808coder/yangsan-traffic-risk-v2")

st.divider()
st.caption("진행 상황은 README.md의 체크리스트에서 확인하실 수 있습니다.")
