"""
weather_api.py
4주차 산출물 — 실시간 기상 API 연동

TODO: 아래는 기상청 공공데이터포털 "초단기실황조회" API 형식을 예시로
      작성했습니다. 실제 보유하신 API의 endpoint / 파라미터 / 응답 구조가
      다르다면 fetch_current_weather()의 요청부와 파싱부만 교체하면 됩니다.
      (캐시/fallback 로직은 그대로 재사용 가능)

핵심 설계:
    1) API 호출 실패 시 서비스가 죽지 않도록 최근 캐시값으로 fallback
    2) 캐시는 파일 기반(JSON)으로 단순하게 — v3에서 DB로 옮기면 됨
    3) app.py는 get_current_weather()만 호출하면 됨 (내부 구현은 몰라도 됨)
"""

import json
import os
import time
import requests

# ── 실제 값으로 교체 ─────────────────────────────────────────
API_KEY = os.environ.get("WEATHER_API_KEY", "YOUR_API_KEY_HERE")  # TODO
API_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"  # TODO
NX, NY = 100, 55  # TODO: 양산시 격자좌표로 교체 (기상청 격자변환 필요)
# ──────────────────────────────────────────────────────────

CACHE_PATH = "data_processed/weather_cache.json"
CACHE_TTL_SEC = 60 * 30  # 30분 이내 캐시는 그대로 사용


def _read_cache():
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
    if time.time() - cache.get("_cached_at", 0) > CACHE_TTL_SEC:
        return None
    return cache


def _write_cache(data: dict):
    data = dict(data)
    data["_cached_at"] = time.time()
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def fetch_current_weather() -> dict:
    """기상 API를 직접 호출해 현재 기상 상태를 가져온다.
    TODO: 실제 API 스펙에 맞춰 params / 파싱 로직 교체.
    """
    now = time.strftime("%Y%m%d")
    base_time = time.strftime("%H00")

    params = {
        "serviceKey": API_KEY,
        "dataType": "JSON",
        "base_date": now,
        "base_time": base_time,
        "nx": NX,
        "ny": NY,
    }
    response = requests.get(API_URL, params=params, timeout=5)
    response.raise_for_status()
    raw = response.json()

    # TODO: 실제 응답 구조에 맞춰 파싱 (아래는 예시 필드명)
    items = raw["response"]["body"]["items"]["item"]
    parsed = {item["category"]: item["obsrValue"] for item in items}

    return {
        "temperature": float(parsed.get("T1H", 0)),   # 기온
        "rainfall": float(parsed.get("RN1", 0)),       # 1시간 강수량
        "humidity": float(parsed.get("REH", 0)),       # 습도
        "wind_speed": float(parsed.get("WSD", 0)),      # 풍속
    }


def get_current_weather() -> dict:
    """app.py에서 호출하는 진입점.
    API 실패 시 캐시 → 그마저 없으면 안전한 기본값을 반환해
    대시보드가 절대 죽지 않도록 한다.
    """
    try:
        weather = fetch_current_weather()
        _write_cache(weather)
        weather["_source"] = "live"
        return weather
    except Exception as e:
        print(f"[경고] 실시간 기상 API 호출 실패: {e}")
        cached = _read_cache()
        if cached:
            cached["_source"] = "cache"
            return cached
        # 캐시도 없으면 평이한 날씨로 가정 (서비스 다운 방지용 최후 fallback)
        return {
            "temperature": 15.0,
            "rainfall": 0.0,
            "humidity": 50.0,
            "wind_speed": 1.0,
            "_source": "default",
        }


if __name__ == "__main__":
    print(get_current_weather())
