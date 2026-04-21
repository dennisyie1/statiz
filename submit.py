import time
import hmac
import hashlib
import urllib.request
import urllib.parse
import json
import pandas as pd


API_KEY = "9cc2976dc30ff6190b7b0485536cc483"
SECRET  = "61f2cb7a045b539c7a530af70a5be50cbff571f69e6dd8fbaa3dcc1fb8835ac8"
BASE_URL = "https://api.statiz.co.kr/baseballApi"


def normalize_query(params: dict) -> str:
    safe = "-_.!~*'()"
    return "&".join(
        f"{urllib.parse.quote(str(k), safe=safe)}={urllib.parse.quote(str(params[k]), safe=safe)}"
        for k in sorted(params.keys())
        if params[k] is not None
    )


def make_signature(secret: str, payload: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

def submit_one_prediction(s_no, percent, path="prediction/gameResult", verbose=False):
    """
    한 경기 예측값 제출
    - s_no: 경기번호
    - percent: 홈팀 승리확률 (%)
    """

    method = "POST"

    # percent는 소수점 둘째 자리까지 맞춤
    percent = round(float(percent), 2)

    # form-data / x-www-form-urlencoded용 body
    body_dict = {
        "s_no": str(s_no),
        "percent": f"{percent:.2f}"
    }

    # POST body를 서명 payload에 어떻게 넣는지는
    # 실제 명세에 따라 다를 수 있음.
    # 지금은 기존 규칙과 맞춰 "METHOD|PATH||TIMESTAMP" 형태로 처리.
    # 만약 문서상 POST도 query/body를 payload에 포함하라고 되어 있으면 이 부분 수정 필요.
    normalized = ""
    timestamp = str(int(time.time()))
    payload = f"{method}|{path}|{normalized}|{timestamp}"
    signature = make_signature(SECRET, payload)

    url = f"{BASE_URL}/{path}"

    headers = {
        "X-API-KEY": API_KEY,
        "X-TIMESTAMP": timestamp,
        "X-SIGNATURE": signature,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    encoded_body = urllib.parse.urlencode(body_dict).encode("utf-8")

    if verbose:
        print("URL:", url)
        print("Payload:", payload)
        print("Body:", body_dict)

    req = urllib.request.Request(
        url,
        data=encoded_body,
        method=method,
        headers=headers
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")

    return json.loads(body)

def submit_predictions_df(pred_submit_df, path="prediction/gameResult", sleep_sec=0.2, verbose=True):
    """
    pred_submit_df 예시 컬럼
    - game_id
    - home_win_proba

    home_win_proba가 이미 % 단위(예: 30.68)라고 가정
    """
    results = []

    required_cols = {"game_id", "home_win_proba"}
    missing = required_cols - set(pred_submit_df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    for _, row in pred_submit_df.iterrows():
        s_no = str(row["game_id"])
        percent = float(row["home_win_proba"])

        try:
            response = submit_one_prediction(
                s_no=s_no,
                percent=percent,
                path=path,
                verbose=False
            )

            result_row = {
                "game_id": s_no,
                "percent": percent,
                "result_cd": response.get("result_cd"),
                "result_msg": response.get("result_msg"),
                "raw_response": response
            }

            if verbose:
                print(f"[SUBMIT OK] s_no={s_no}, percent={percent:.2f}, "
                      f"result_cd={response.get('result_cd')}, result_msg={response.get('result_msg')}")

        except Exception as e:
            result_row = {
                "game_id": s_no,
                "percent": percent,
                "result_cd": None,
                "result_msg": str(e),
                "raw_response": None
            }

            if verbose:
                print(f"[SUBMIT FAIL] s_no={s_no}, percent={percent:.2f}, error={e}")

        results.append(result_row)

        # 429 방지용 텀
        time.sleep(sleep_sec)

    return pd.DataFrame(results)