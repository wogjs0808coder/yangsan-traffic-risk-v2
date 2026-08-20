"""
map_visualizer.py
5주차 산출물 — 사고다발지역 지도 시각화 (Folium)

TODO: 실제 accident.csv에 위도/경도 컬럼이 있는지 확인.
      없다면 '읍면동' 등 행정구역명을 기준으로 지오코딩(1회성 전처리)이
      선행되어야 합니다. 아래는 위도/경도 컬럼이 있다고 가정한 버전입니다.
"""

import pandas as pd
import folium
from folium.plugins import HeatMap

LAT_COL = "위도"   # TODO: 실제 컬럼명
LON_COL = "경도"   # TODO: 실제 컬럼명

YANGSAN_CENTER = [35.3350, 129.0378]  # 양산시청 대략 좌표


def load_accident_locations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[LAT_COL, LON_COL])
    return df


def build_heatmap(df: pd.DataFrame) -> folium.Map:
    """사고 밀집도를 히트맵으로 표현. 한눈에 위험 지역 파악용."""
    m = folium.Map(location=YANGSAN_CENTER, zoom_start=12, tiles="CartoDB positron")
    points = df[[LAT_COL, LON_COL]].values.tolist()
    HeatMap(points, radius=15, blur=20).add_to(m)
    return m


def build_marker_map(df: pd.DataFrame, risk_col: str = None) -> folium.Map:
    """개별 사고 지점을 마커로 표현. risk_col을 주면 위험도에 따라 색을 다르게.

    risk_col 값이 문자열(예: '고위험'/'중위험'/'저위험')이라고 가정.
    TODO: 실제 위험도 컬럼명/값 체계에 맞게 COLOR_MAP 수정.
    """
    COLOR_MAP = {"고위험": "red", "중위험": "orange", "저위험": "green"}
    m = folium.Map(location=YANGSAN_CENTER, zoom_start=12, tiles="CartoDB positron")

    for _, row in df.iterrows():
        color = COLOR_MAP.get(row.get(risk_col, ""), "blue") if risk_col else "blue"
        folium.CircleMarker(
            location=[row[LAT_COL], row[LON_COL]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=str(row.get(risk_col, "")),
        ).add_to(m)

    return m


def render_in_streamlit(m: folium.Map):
    """app.py에서 호출:
        from map_visualizer import load_accident_locations, build_heatmap, render_in_streamlit
        df = load_accident_locations("data/accident.csv")
        render_in_streamlit(build_heatmap(df))
    """
    from streamlit_folium import st_folium
    st_folium(m, width=700, height=500)


if __name__ == "__main__":
    df = load_accident_locations("data/accident.csv")
    m = build_heatmap(df)
    m.save("data_processed/accident_heatmap.html")
    print("저장됨: data_processed/accident_heatmap.html")
