import json
import os
import time

import requests

API_KEY = os.environ.get("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

REGION_COORDS = {
    "seoul": (37.5665, 126.9780),
    "busan": (35.1796, 129.0756),
    "daegu": (35.8714, 128.6014),
    "incheon": (37.4563, 126.7052),
    "daejeon": (36.3504, 127.3845),
    "yangsan": (35.3350, 129.0378),
}

CACHE_TTL_SEC = 60 * 10  # 10분 이내 캐시는 재사용 (무료 플랜 호출 제한 보호용)
RAIN_CLIP_MAX = 100.0    # legacy_utils.clip_and_flag_rainfall과 동일 기준


def _cache_path(region: str) -> str:
    return f"data_processed/{region}/weather_cache.json"


def _read_cache(region: str):
    path = _cache_path(region)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    if time.time() - cache.get("_cached_at", 0) > CACHE_TTL_SEC:
        return None
    return cache


def _write_cache(region: str, data: dict):
    data = dict(data)
    data["_cached_at"] = time.time()
    os.makedirs(os.path.dirname(_cache_path(region)), exist_ok=True)
    with open(_cache_path(region), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def parse_openweather_response(raw: dict) -> dict:
    """OpenWeatherMap 응답 JSON을 우리 프로젝트 표준 형태로 변환.

    별도 함수로 분리한 이유: 실제 API 호출 없이도 이 파싱 로직만 따로
    테스트할 수 있게 하기 위함 (아래 __main__ 참고).

    주의: 비가 안 오면 응답에 'rain' 키 자체가 아예 없다. 없으면 0으로 처리.
    """
    rainfall = raw.get("rain", {}).get("1h", 0.0)

    return {
        "temperature": raw["main"]["temp"],       # units=metric이므로 이미 섭씨
        "rainfall": rainfall,
        "humidity": raw["main"]["humidity"],
        "wind_speed": raw["wind"]["speed"],
        "weather_description": raw["weather"][0]["description"],
    }


def fetch_current_weather(region: str) -> dict:
    """OpenWeatherMap Current Weather API를 직접 호출한다.
    무료 플랜은 분당 60회 제한이 있으니, 대시보드에서 매 새로고침마다
    직접 호출하기보다 get_current_weather()의 캐시를 거치는 걸 추천.
    """
    if region not in REGION_COORDS:
        raise ValueError(f"알 수 없는 지역: {region} (가능한 값: {list(REGION_COORDS)})")

    lat, lon = REGION_COORDS[region]
    params = {
        "lat": lat, "lon": lon, "appid": API_KEY,
        "units": "metric", "lang": "kr",
    }
    response = requests.get(BASE_URL, params=params, timeout=5)
    response.raise_for_status()
    return parse_openweather_response(response.json())


def get_current_weather(region: str) -> dict:
    """app.py에서 호출하는 진입점.
    API 실패 시 캐시 → 그마저 없으면 안전한 기본값을 반환해
    대시보드가 절대 죽지 않도록 한다.
    """
    try:
        weather = fetch_current_weather(region)
        _write_cache(region, weather)
        weather["_source"] = "live"
        return weather
    except Exception as e:
        print(f"[경고] {region} 실시간 기상 API 호출 실패: {e}")
        cached = _read_cache(region)
        if cached:
            cached["_source"] = "cache"
            return cached
        return {
            "temperature": 15.0, "rainfall": 0.0, "humidity": 50.0,
            "wind_speed": 1.0, "weather_description": "정보없음", "_source": "default",
        }


def to_model_features(weather: dict) -> dict:
    """모델이 학습 때 쓴 컬럼명(preprocessor/legacy_utils 기준)에 맞춰 변환.
    app.py에서 예측용 입력 DataFrame을 만들 때 이 결과를 바로 이어붙이면 됨.
    """
    rainfall = weather["rainfall"]
    return {
        "평균기온(°C)": weather["temperature"],
        "일강수량_클립(mm)": min(rainfall, RAIN_CLIP_MAX),
        "평균 풍속(m/s)": weather["wind_speed"],
        "평균 상대습도(%)": weather["humidity"],
        "폭우_재난_플래그": 1 if rainfall > RAIN_CLIP_MAX else 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="yangsan", choices=list(REGION_COORDS))
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 API를 호출하지 않고, 모의 응답으로 파싱 로직만 테스트",
    )
    args = parser.parse_args()

    if args.dry_run:
        mock_response = {
            "main": {"temp": 18.5, "humidity": 62},
            "wind": {"speed": 2.3},
            "weather": [{"description": "실 비"}],
            "rain": {"1h": 3.2},
        }
        parsed = parse_openweather_response(mock_response)
        print("[모의 응답 파싱 결과]", parsed)
        print("[모델 입력 변환 결과]", to_model_features(parsed))
    else:
        weather = get_current_weather(args.region)
        print(weather)
        print(to_model_features(weather))
