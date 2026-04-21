from IPython.display import display
import time
import hmac
import hashlib
import urllib.request
from urllib.parse import quote
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

API_KEY = "9cc2976dc30ff6190b7b0485536cc483"
SECRET  = "61f2cb7a045b539c7a530af70a5be50cbff571f69e6dd8fbaa3dcc1fb8835ac8"

BASE_URL = "https://api.statiz.co.kr/baseballApi"

# 공통 함수

def normalize_query(params: dict) -> str:
    safe = "-_.!~*'()"
    return "&".join(
        f"{quote(str(k), safe=safe)}={quote(str(params[k]), safe=safe)}"
        for k in sorted(params.keys())
        if params[k] is not None
    )


def make_signature(secret: str, payload: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def api_get(path: str, query: Dict[str, Any], timeout: int = 30, verbose: bool = False) -> Dict[str, Any]:
    method = "GET"
    normalized = normalize_query(query)
    timestamp = str(int(time.time()))
    payload = f"{method}|{path}|{normalized}|{timestamp}"
    signature = make_signature(SECRET, payload)

    url = f"{BASE_URL}/{path}"
    if normalized:
        url = f"{url}?{normalized}"

    headers = {
        "X-API-KEY": API_KEY,
        "X-TIMESTAMP": timestamp,
        "X-SIGNATURE": signature,
    }

    if verbose:
        print("URL:", url)
        print("Payload:", payload)

    req = urllib.request.Request(url, method=method, headers=headers)

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")

    data = json.loads(body)

    # 공통 실패 처리
    if isinstance(data, dict):
        result_cd = str(data.get("result_cd", ""))
        if result_cd and result_cd != "100":
            raise ValueError(f"API 실패: result_cd={result_cd}, result_msg={data.get('result_msg')}")

    return data


def numeric_key_rows(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    응답이 {'0': {...}, '1': {...}, 'result_cd': '100', ...} 형태일 때
    숫자 key row들만 추출.
    """
    rows = []
    for k, v in obj.items():
        if str(k).isdigit() and isinstance(v, dict):
            rows.append(v)
    return rows


def to_df_from_list_or_numeric_dict(obj: Dict[str, Any], list_key: Optional[str] = None) -> pd.DataFrame:
    """
    응답이
    1) {'list': [...]} 형태이거나
    2) {'0': {...}, '1': {...}} 형태일 때
    DataFrame으로 변환.
    """
    if list_key is not None and list_key in obj and isinstance(obj[list_key], list):
        return pd.DataFrame(obj[list_key])

    rows = numeric_key_rows(obj)
    if rows:
        return pd.DataFrame(rows)

    return pd.DataFrame()

def fetch_game_schedule(year: int, month: int, day: int, verbose: bool = False) -> pd.DataFrame:
    path = "prediction/gameSchedule"
    query = {
        "year": year,
        "month": month,
        "day": day
    }

    data = api_get(path, query, verbose=verbose)

    rows = []

    # 1) top-level에서 메타 키(result_cd 등) 제외하고, list value인 항목 수집
    if isinstance(data, dict):
        meta_keys = {"result_cd", "result_msg", "update_time"}
        for k, v in data.items():
            if k not in meta_keys and isinstance(v, list):
                rows.extend(v)

    df = pd.DataFrame(rows)

    if df.empty:
        if verbose:
            print("[fetch_game_schedule] rows is empty")
            print("raw keys:", list(data.keys()) if isinstance(data, dict) else type(data))
        return df

    # 컬럼명 표준화
    rename_map = {
        "s_no": "game_id",
        "gameDate": "game_date",
        "awayTeam": "away_team_code",
        "homeTeam": "home_team_code",
        "s_code": "stadium_code",
        "awaySP": "away_sp_id",
        "homeSP": "home_sp_id",
        "awaySPName": "away_sp_name",
        "homeSPName": "home_sp_name",
        "awayScore": "away_score",
        "homeScore": "home_score",
        "state": "game_state",
        "hm": "game_time",
    }
    df = df.rename(columns=rename_map)

    # game_date가 unix timestamp(초) 형태이므로 변환
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], unit="s", errors="coerce")
    else:
        df["game_date"] = pd.to_datetime(f"{year}-{month:02d}-{day:02d}", errors="coerce")

    # season 보강
    if "season" not in df.columns:
        if "year" in df.columns:
            df["season"] = pd.to_numeric(df["year"], errors="coerce")
        else:
            df["season"] = year

    # home_win 생성
    if {"home_score", "away_score"}.issubset(df.columns):
        df["home_win"] = (
            pd.to_numeric(df["home_score"], errors="coerce") >
            pd.to_numeric(df["away_score"], errors="coerce")
        ).astype("Int64")
    else:
        df["home_win"] = pd.NA

    keep_cols = [
        "game_id",
        "game_date",
        "season",
        "home_team_code",
        "away_team_code",
        "stadium_code",
        "home_sp_id",
        "away_sp_id",
        "home_sp_name",
        "away_sp_name",
        "home_score",
        "away_score",
        "home_win",
        "game_state",
        "game_time",
        "weather",
        "temperature",
        "humidity",
        "windDirection",
        "windSpeed",
        "rainprobability",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].copy()

# 여러날 일괄 수집

def collect_games(date_list: List[str], sleep_sec: float = 0.2) -> pd.DataFrame:
    all_dfs = []

    for dt in date_list:
        y, m, d = map(int, dt.split("-"))
        try:
            df = fetch_game_schedule(y, m, d)
            if not df.empty:
                all_dfs.append(df)
            print(f"[games] {dt}: {len(df)} rows")
        except Exception as e:
            print(f"[games] {dt}: ERROR -> {e}")
        time.sleep(sleep_sec)

    if all_dfs:
        out = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
    else:
        out = pd.DataFrame()

    return out

def fetch_game_lineup(game_id: str, verbose: bool = False) -> pd.DataFrame:
    path = "prediction/gameLineup"
    query = {"s_no": str(game_id)}

    data = api_get(path, query, verbose=verbose)

    rows = []

    # 메타 키 제외, 팀코드 key 아래 list 수집
    if isinstance(data, dict):
        meta_keys = {"result_cd", "result_msg", "update_time"}
        for k, v in data.items():
            if k not in meta_keys and isinstance(v, list):
                rows.extend(v)

    df = pd.DataFrame(rows)

    if df.empty:
        if verbose:
            print("[fetch_game_lineup] rows is empty")
            if isinstance(data, dict):
                print("raw keys:", list(data.keys()))
        return df

    # 컬럼 표준화
    rename_map = {
        "s_no": "game_id",
        "p_no": "player_id",
        "t_code": "team_code",
        "p_name": "player_name",
        "starting": "is_starting",
        "battingOrder": "batting_order",
        "p_throw": "throws",
        "p_bat": "bats",
        "p_backNumber": "back_number",
    }
    df = df.rename(columns=rename_map)

    # game_id 문자열 통일
    df["game_id"] = df["game_id"].astype(str)

    # starting: Y/N -> 1/0
    if "is_starting" in df.columns:
        df["is_starting"] = df["is_starting"].map({"Y": 1, "N": 0}).fillna(df["is_starting"])

    # batting_order 숫자화
    if "batting_order" in df.columns:
        df["batting_order"] = pd.to_numeric(df["batting_order"], errors="coerce")

    # 선발투수 flag
    # position==1 이 투수라고 가정
    if "position" in df.columns:
        pos = pd.to_numeric(df["position"], errors="coerce")
        if "is_starting" in df.columns:
            st = pd.to_numeric(df["is_starting"], errors="coerce")
            df["is_starting_pitcher"] = ((pos == 1) & (st == 1)).astype(int)
        else:
            df["is_starting_pitcher"] = (pos == 1).astype(int)
    else:
        df["is_starting_pitcher"] = pd.NA

    # 중복 제거
    subset_cols = [c for c in ["game_id", "team_code", "player_id"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)

    keep_cols = [
        "game_id",
        "game_date",
        "team_code",
        "player_id",
        "player_name",
        "is_starting",
        "lineupState",
        "batting_order",
        "position",
        "is_starting_pitcher",
        "throws",
        "bats",
        "back_number",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].copy()

# games에서 lineup일괄 수집

def collect_lineups(games_df: pd.DataFrame, sleep_sec: float = 0.2) -> pd.DataFrame:
    if "game_id" not in games_df.columns:
        raise ValueError("games_df에 game_id 컬럼이 필요합니다.")

    all_dfs = []

    for game_id in games_df["game_id"].astype(str).tolist():
        try:
            df = fetch_game_lineup(game_id)

            if not df.empty:
                game_date = games_df.loc[
                    games_df["game_id"].astype(str) == game_id, "game_date"
                ]
                if not game_date.empty:
                    df["game_date"] = game_date.iloc[0]

                all_dfs.append(df)

            print(f"[lineup] {game_id}: {len(df)} rows")

        except Exception as e:
            print(f"[lineup] {game_id}: ERROR -> {e}")

        time.sleep(sleep_sec)  # 429 방지

    if all_dfs:
        out = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
    else:
        out = pd.DataFrame()

    return out

def convert_ip_baseball_to_decimal(x): ## 얘 다시
    """
    예: 66.1 -> 66.333..., 66.2 -> 66.666...
    문자열/숫자 모두 대응
    """
    if pd.isna(x):
        return pd.NA
    try:
        s = str(x)
        if "." not in s:
            return float(s)
        whole, frac = s.split(".")
        whole = int(whole)
        frac = int(frac)
        if frac == 0:
            return float(whole)
        elif frac == 1:
            return whole + 1/3
        elif frac == 2:
            return whole + 2/3
        else:
            return float(s)
    except Exception:
        return pd.NA


def fetch_player_season(player_id: str, verbose: bool = False) -> Dict[str, pd.DataFrame]:
    path = "prediction/playerSeason"
    query = {"p_no": str(player_id)}

    data = api_get(path, query, verbose=verbose)

    basic_df   = pd.DataFrame(data.get("basic", {}).get("list", []))
    deepen_df  = pd.DataFrame(data.get("deepen", {}).get("list", []))
    field_df   = pd.DataFrame(data.get("fielding", {}).get("list", []))

    # year 기준 merge
    if not basic_df.empty and not deepen_df.empty and "year" in basic_df.columns and "year" in deepen_df.columns:
        merged = basic_df.merge(deepen_df, on="year", how="left", suffixes=("", "_deepen"))
    else:
        merged = basic_df.copy()

    if not merged.empty:
        merged["player_id"] = str(player_id)

    # 투수/타자 구분: 보통 p_position == 1 이면 투수
    is_pitcher = False
    if not merged.empty and "p_position" in merged.columns:
        vals = merged["p_position"].dropna().astype(str).unique().tolist()
        is_pitcher = ("1" in vals)

    # 공통 컬럼명 표준화
    if "t_code" in merged.columns:
        merged = merged.rename(columns={"t_code": "team_code"})

    # IP 변환
    if "IP" in merged.columns:
        merged["IP_decimal"] = merged["IP"].apply(convert_ip_baseball_to_decimal)

    if is_pitcher:
        keep_cols = [
            "player_id", "year", "team_code",
            "GS", "IP", "IP_decimal", "ERA", "FIP", "WHIP",
            "K9", "BB9", "HR9", "KBB", "QS"
        ]
        keep_cols = [c for c in keep_cols if c in merged.columns]
        pitcher_df = merged[keep_cols].copy()
        hitter_df = pd.DataFrame()
    else:
        keep_cols = [
            "player_id", "year", "team_code",
            "PA", "AB", "H", "HR", "RBI", "BB", "SO",
            "AVG", "OBP", "SLG", "OPS", "wOBA", "wRCplus", "BBK"
        ]
        keep_cols = [c for c in keep_cols if c in merged.columns]
        hitter_df = merged[keep_cols].copy()
        pitcher_df = pd.DataFrame()

    return {
        "hitter": hitter_df,
        "pitcher": pitcher_df,
        "raw_basic": basic_df,
        "raw_deepen": deepen_df,
        "raw_fielding": field_df
    }

# 일괄수집

def collect_player_season(player_ids: List[str], snapshot_date: str, sleep_sec: float = 0.2):
    hitter_list = []
    pitcher_list = []

    for pid in map(str, player_ids):
        try:
            res = fetch_player_season(pid)
            if not res["hitter"].empty:
                df = res["hitter"].copy()
                df["snapshot_date"] = snapshot_date
                hitter_list.append(df)
            if not res["pitcher"].empty:
                df = res["pitcher"].copy()
                df["snapshot_date"] = snapshot_date
                pitcher_list.append(df)
            print(f"[playerSeason] {pid}: OK")
        except Exception as e:
            print(f"[playerSeason] {pid}: ERROR -> {e}")
        time.sleep(sleep_sec)

    hitter_out = pd.concat(hitter_list, ignore_index=True) if hitter_list else pd.DataFrame()
    pitcher_out = pd.concat(pitcher_list, ignore_index=True) if pitcher_list else pd.DataFrame()

    return hitter_out, pitcher_out

def fetch_player_roster(date: str, code: str, t_code: str, verbose: bool = False) -> pd.DataFrame:
    path = "prediction/playerRoster"
    query = {
        "date": date,
        "code": str(code),
        "t_code": str(t_code)
    }

    data = api_get(path, query, verbose=verbose)
    df = to_df_from_list_or_numeric_dict(data, list_key="list")

    if df.empty:
        return df

    rename_map = {
        "name": "player_name",
        "p_no": "player_id",
        "t_code": "team_code",
        "pj_date": "join_date"
    }
    df = df.rename(columns=rename_map)
    df["snapshot_date"] = date

    keep_cols = ["snapshot_date", "team_code", "player_id", "player_name", "join_date"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].copy()

# 일괄 수집

def collect_rosters(date: str, team_codes: List[str], code: str = "1", sleep_sec: float = 0.2) -> pd.DataFrame:
    all_dfs = []

    for t_code in map(str, team_codes):
        try:
            df = fetch_player_roster(date=date, code=code, t_code=t_code)
            if not df.empty:
                all_dfs.append(df)
            print(f"[roster] {date} team={t_code}: {len(df)} rows")
        except Exception as e:
            print(f"[roster] {date} team={t_code}: ERROR -> {e}")
        time.sleep(sleep_sec)

    if all_dfs:
        out = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
    else:
        out = pd.DataFrame()

    return out

team_codes = ["1001", "2002", "3001", "5002", "6002", "7002", "9002", "10001", "11001", "12001"]
#               삼성,    KIA,   롯데,     LG,   두산,   한화,    SSG,    키움,      NC,     KT

# 전처리 보조 함수

def prepare_games_for_modeling(games_df: pd.DataFrame) -> pd.DataFrame:
    df = games_df.copy()

    # 정상 경기만
    df = df[
        (df["game_state"] == 3) &
        (df["home_score"].notna()) &
        (df["away_score"].notna())
    ].copy()

    df["game_id"] = df["game_id"].astype(str)
    df["home_team_code"] = df["home_team_code"].astype(str)
    df["away_team_code"] = df["away_team_code"].astype(str)

    if "home_win" not in df.columns:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    return df


def prepare_lineup(lineup_df: pd.DataFrame) -> pd.DataFrame:
    df = lineup_df.copy()

    df["game_id"] = df["game_id"].astype(str)
    df["team_code"] = df["team_code"].astype(str)
    df["player_id"] = df["player_id"].astype(str)

    if "batting_order" in df.columns:
        df["batting_order"] = pd.to_numeric(df["batting_order"], errors="coerce")

    if "position" in df.columns:
        df["position"] = pd.to_numeric(df["position"], errors="coerce")

    if "is_starting" in df.columns:
        df["is_starting"] = pd.to_numeric(df["is_starting"], errors="coerce")

    if "is_starting_pitcher" in df.columns:
        df["is_starting_pitcher"] = pd.to_numeric(df["is_starting_pitcher"], errors="coerce")

    return df


def prepare_season_hitters(season_hit_df: pd.DataFrame) -> pd.DataFrame:
    df = season_hit_df.copy()

    df["player_id"] = df["player_id"].astype(str)
    df["team_code"] = df["team_code"].astype(str)

    num_cols = ["PA", "OBP", "SLG", "OPS", "wOBA", "wRCplus", "BBK"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def prepare_season_pitchers(season_pit_df: pd.DataFrame) -> pd.DataFrame:
    df = season_pit_df.copy()

    df["player_id"] = df["player_id"].astype(str)
    df["team_code"] = df["team_code"].astype(str)

    num_cols = ["GS", "IP", "IP_decimal", "ERA", "FIP", "WHIP", "K9", "BB9", "HR9", "KBB", "QS"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# 타자 feature 생성

def build_hitter_team_features(lineup_df: pd.DataFrame, season_hit_df: pd.DataFrame) -> pd.DataFrame:
    # 타자만: 선발투수 제외, 타순 있는 선수 중심
    hitters = lineup_df.copy()
    if "is_starting_pitcher" in hitters.columns:
        hitters = hitters[hitters["is_starting_pitcher"] != 1].copy()

    hitters = hitters.merge(
        season_hit_df,
        on=["player_id", "team_code"],
        how="left",
        suffixes=("", "_season")
    )

    # lineup 전체 평균
    def weighted_mean(x, value_col, weight_col="PA"):
        tmp = x[[value_col, weight_col]].dropna()
        if tmp.empty:
            return np.nan
        w = tmp[weight_col]
        v = tmp[value_col]
        if w.sum() == 0:
            return v.mean()
        return np.average(v, weights=w)

    team_rows = []

    for (game_id, team_code), g in hitters.groupby(["game_id", "team_code"]):
        row = {
            "game_id": game_id,
            "team_code": team_code,
            "lineup_n_hitters": len(g),
        }

        for stat in ["OPS", "OBP", "SLG"]:
            if stat in g.columns:
                row[f"lineup_{stat}"] = weighted_mean(g, stat, "PA")

        # top5
        if "batting_order" in g.columns:
            top5 = g[g["batting_order"].between(1, 5, inclusive="both")].copy()
        else:
            top5 = pd.DataFrame()

        row["top5_n"] = len(top5)

        for stat in ["OPS", "OBP"]:
            if not top5.empty and stat in top5.columns:
                row[f"top5_{stat}"] = weighted_mean(top5, stat, "PA")
            else:
                row[f"top5_{stat}"] = np.nan

        team_rows.append(row)

    return pd.DataFrame(team_rows)

# 선발투수 feature 생성

def build_starting_pitcher_features(lineup_df: pd.DataFrame, season_pit_df: pd.DataFrame) -> pd.DataFrame:
    sp = lineup_df.copy()

    # 선발투수만
    if "is_starting_pitcher" in sp.columns:
        sp = sp[sp["is_starting_pitcher"] == 1].copy()
    else:
        sp = sp[sp["position"] == 1].copy()

    sp = sp.merge(
        season_pit_df,
        on=["player_id", "team_code"],
        how="left",
        suffixes=("", "_season")
    )

    keep_cols = [
        "game_id", "team_code", "player_id", "player_name",
        "ERA", "FIP", "WHIP", "K9", "BB9", "HR9", "KBB", "QS", "GS"
    ]
    keep_cols = [c for c in keep_cols if c in sp.columns]
    sp = sp[keep_cols].copy()

    rename_map = {
        "player_id": "sp_player_id",
        "player_name": "sp_player_name",
        "ERA": "sp_ERA",
        "FIP": "sp_FIP",
        "WHIP": "sp_WHIP",
        "K9": "sp_K9",
        "BB9": "sp_BB9",
        "HR9": "sp_HR9",
        "KBB": "sp_KBB",
        "QS": "sp_QS",
        "GS": "sp_GS",
    }
    sp = sp.rename(columns=rename_map)

    return sp

# 홈 원정 통합 경기단위 데이터

def make_game_level_dataset(
    games_df_clean: pd.DataFrame,
    hitter_team_features: pd.DataFrame = None,
    sp_features: pd.DataFrame = None
) -> pd.DataFrame:
    """
    수정:
    - hitter_team_features가 None/empty여도 동작
    - 선발투수 feature만으로도 pred_df 생성 가능
    """
    df = games_df_clean.copy()

    # --------------------------------------------------
    # 1) hitter feature merge (있을 때만)
    # --------------------------------------------------
    if hitter_team_features is not None and not hitter_team_features.empty:
        home_hit = hitter_team_features.copy().rename(columns={
            "team_code": "home_team_code",
            "lineup_n_hitters": "home_lineup_n_hitters",
            "lineup_OPS": "home_lineup_OPS",
            "lineup_OBP": "home_lineup_OBP",
            "lineup_SLG": "home_lineup_SLG",
            "top5_n": "home_top5_n",
            "top5_OPS": "home_top5_OPS",
            "top5_OBP": "home_top5_OBP",
        })

        away_hit = hitter_team_features.copy().rename(columns={
            "team_code": "away_team_code",
            "lineup_n_hitters": "away_lineup_n_hitters",
            "lineup_OPS": "away_lineup_OPS",
            "lineup_OBP": "away_lineup_OBP",
            "lineup_SLG": "away_lineup_SLG",
            "top5_n": "away_top5_n",
            "top5_OPS": "away_top5_OPS",
            "top5_OBP": "away_top5_OBP",
        })

        df = df.merge(home_hit, on=["game_id", "home_team_code"], how="left")
        df = df.merge(away_hit, on=["game_id", "away_team_code"], how="left")

        # hitter diff 생성
        diff_pairs_hit = [
            ("lineup_OPS", "home_lineup_OPS", "away_lineup_OPS"),
            ("lineup_OBP", "home_lineup_OBP", "away_lineup_OBP"),
            ("lineup_SLG", "home_lineup_SLG", "away_lineup_SLG"),
            ("top5_OPS", "home_top5_OPS", "away_top5_OPS"),
            ("top5_OBP", "home_top5_OBP", "away_top5_OBP"),
        ]

        for new_name, h_col, a_col in diff_pairs_hit:
            if h_col in df.columns and a_col in df.columns:
                df[f"{new_name}_diff"] = df[h_col] - df[a_col]

    # --------------------------------------------------
    # 2) pitcher feature merge (필수)
    # --------------------------------------------------
    if sp_features is not None and not sp_features.empty:
        home_sp = sp_features.copy().rename(columns={
            "team_code": "home_team_code",
            "sp_player_id": "home_sp_player_id",
            "sp_player_name": "home_sp_player_name",
            "sp_ERA": "home_sp_ERA",
            "sp_FIP": "home_sp_FIP",
            "sp_WHIP": "home_sp_WHIP",
            "sp_K9": "home_sp_K9",
            "sp_BB9": "home_sp_BB9",
            "sp_HR9": "home_sp_HR9",
            "sp_KBB": "home_sp_KBB",
            "sp_QS": "home_sp_QS",
            "sp_GS": "home_sp_GS",
        })

        away_sp = sp_features.copy().rename(columns={
            "team_code": "away_team_code",
            "sp_player_id": "away_sp_player_id",
            "sp_player_name": "away_sp_player_name",
            "sp_ERA": "away_sp_ERA",
            "sp_FIP": "away_sp_FIP",
            "sp_WHIP": "away_sp_WHIP",
            "sp_K9": "away_sp_K9",
            "sp_BB9": "away_sp_BB9",
            "sp_HR9": "away_sp_HR9",
            "sp_KBB": "away_sp_KBB",
            "sp_QS": "away_sp_QS",
            "sp_GS": "away_sp_GS",
        })

        df = df.merge(home_sp, on=["game_id", "home_team_code"], how="left")
        df = df.merge(away_sp, on=["game_id", "away_team_code"], how="left")

        # pitcher diff 생성
        diff_pairs_pit = [
            ("sp_ERA", "home_sp_ERA", "away_sp_ERA"),
            ("sp_FIP", "home_sp_FIP", "away_sp_FIP"),
            ("sp_WHIP", "home_sp_WHIP", "away_sp_WHIP"),
            ("sp_K9", "home_sp_K9", "away_sp_K9"),
            ("sp_BB9", "home_sp_BB9", "away_sp_BB9"),
            ("sp_HR9", "home_sp_HR9", "away_sp_HR9"),
            ("sp_KBB", "home_sp_KBB", "away_sp_KBB"),
        ]

        for new_name, h_col, a_col in diff_pairs_pit:
            if h_col in df.columns and a_col in df.columns:
                df[f"{new_name}_diff"] = df[h_col] - df[a_col]


    return df

def _get_last_team_code(group: pd.DataFrame):
    """
    같은 player_id, year 그룹 내에서 마지막 team_code 반환
    - API에서 들어온 행 순서를 그대로 믿고 마지막 팀으로 간주
    """
    if "team_code" not in group.columns:
        return np.nan

    vals = group["team_code"].dropna().astype(str)
    if len(vals) == 0:
        return np.nan

    return vals.iloc[-1]

def deduplicate_hitter_season_last_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    같은 선수의 같은 시즌 여러 행(팀 이동 등)을 1행으로 통합
    - team_code: 마지막 팀으로 통일
    - counting stat: 합산
    - rate stat: PA 가중평균 (불가능하면 평균)
    """
    df = df.copy()

    # year 숫자화
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # 숫자 컬럼 변환
    numeric_cols = [
        "PA", "AB", "H", "HR", "RBI", "BB", "SO",
        "AVG", "OBP", "SLG", "OPS", "wOBA", "wRCplus", "BBK"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out_rows = []

    for (player_id, year), g in df.groupby(["player_id", "year"], sort=False):
        row = {
            "player_id": str(player_id),
            "year": year,
            # 마지막 팀으로 통일
            "team_code": _get_last_team_code(g),
        }

        # 이름은 마지막 값 사용
        if "player_name" in g.columns:
            name_vals = g["player_name"].dropna()
            row["player_name"] = name_vals.iloc[-1] if len(name_vals) > 0 else np.nan

        # -------------------------
        # counting stat 합산
        # -------------------------
        count_cols = ["PA", "AB", "H", "HR", "RBI", "BB", "SO"]
        for c in count_cols:
            if c in g.columns:
                row[c] = g[c].sum(min_count=1)

        # -------------------------
        # 재계산/가중평균용 기반 값
        # -------------------------
        PA = row.get("PA", np.nan)
        AB = row.get("AB", np.nan)
        H  = row.get("H", np.nan)

        # AVG = H / AB
        if "AVG" in df.columns:
            row["AVG"] = (H / AB) if pd.notna(H) and pd.notna(AB) and AB != 0 else np.nan

        # 단순 재계산이 어려운 rate stat
        weighted_rate_cols = ["OBP", "SLG", "OPS", "wOBA", "wRCplus", "BBK"]

        for c in weighted_rate_cols:
            if c in g.columns:
                tmp = g[[c]].copy()

                # PA가 있으면 PA 가중평균
                if "PA" in g.columns:
                    tmp["PA"] = pd.to_numeric(g["PA"], errors="coerce")
                    tmp = tmp.dropna(subset=[c])

                    if not tmp.empty and tmp["PA"].fillna(0).sum() > 0:
                        row[c] = np.average(tmp[c], weights=tmp["PA"].fillna(0))
                    else:
                        row[c] = tmp[c].mean() if not tmp.empty else np.nan
                else:
                    row[c] = pd.to_numeric(g[c], errors="coerce").mean()

        out_rows.append(row)

    out = pd.DataFrame(out_rows)

    ordered_cols = [
        "player_id", "player_name", "team_code", "year",
        "PA", "AB", "H", "HR", "RBI", "BB", "SO",
        "AVG", "OBP", "SLG", "OPS", "wOBA", "wRCplus", "BBK"
    ]
    ordered_cols = [c for c in ordered_cols if c in out.columns]

    return out[ordered_cols].copy()

def deduplicate_pitcher_season_last_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    같은 선수의 같은 시즌 여러 행(팀 이동 등)을 1행으로 통합
    - team_code: 마지막 팀으로 통일
    - counting stat: 합산
    - rate stat: IP 가중평균
    """
    df = df.copy()

    # year 숫자화
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # 숫자 컬럼 변환
    numeric_cols = [
        "GS", "IP", "IP_decimal", "ERA", "FIP", "WHIP",
        "K9", "BB9", "HR9", "KBB", "QS"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out_rows = []

    for (player_id, year), g in df.groupby(["player_id", "year"], sort=False):
        row = {
            "player_id": str(player_id),
            "year": year,
            # 마지막 팀으로 통일
            "team_code": _get_last_team_code(g),
        }

        # 이름은 마지막 값 사용
        if "player_name" in g.columns:
            name_vals = g["player_name"].dropna()
            row["player_name"] = name_vals.iloc[-1] if len(name_vals) > 0 else np.nan

        # -------------------------
        # counting stat 합산
        # -------------------------
        count_cols = ["GS", "QS"]
        for c in count_cols:
            if c in g.columns:
                row[c] = g[c].sum(min_count=1)

        # -------------------------
        # IP는 decimal 기준으로 합산
        # -------------------------
        if "IP_decimal" in g.columns:
            row["IP_decimal"] = g["IP_decimal"].sum(min_count=1)
        elif "IP" in g.columns:
            row["IP"] = g["IP"].sum(min_count=1)

        # -------------------------
        # rate stat은 IP 가중평균
        # -------------------------
        weight_col = "IP_decimal" if "IP_decimal" in g.columns else ("IP" if "IP" in g.columns else None)
        weighted_rate_cols = ["ERA", "FIP", "WHIP", "K9", "BB9", "HR9", "KBB"]

        for c in weighted_rate_cols:
            if c in g.columns:
                tmp = g[[c]].copy()

                if weight_col is not None:
                    tmp[weight_col] = pd.to_numeric(g[weight_col], errors="coerce")
                    tmp = tmp.dropna(subset=[c])

                    if not tmp.empty and tmp[weight_col].fillna(0).sum() > 0:
                        row[c] = np.average(tmp[c], weights=tmp[weight_col].fillna(0))
                    else:
                        row[c] = tmp[c].mean() if not tmp.empty else np.nan
                else:
                    row[c] = pd.to_numeric(g[c], errors="coerce").mean()

        out_rows.append(row)

    out = pd.DataFrame(out_rows)

    ordered_cols = [
        "player_id", "player_name", "team_code", "year",
        "GS", "IP_decimal", "ERA", "FIP", "WHIP",
        "K9", "BB9", "HR9", "KBB", "QS"
    ]
    ordered_cols = [c for c in ordered_cols if c in out.columns]

    return out[ordered_cols].copy()

def deduplicate_pitcher_season_last_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    같은 선수의 같은 시즌 여러 행(팀 이동 등)을 1행으로 통합
    - team_code: 마지막 팀으로 통일
    - counting stat: 합산
    - rate stat: IP 가중평균
    """
    df = df.copy()

    # year 숫자화
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # 숫자 컬럼 변환
    numeric_cols = [
        "GS", "IP", "IP_decimal", "ERA", "FIP", "WHIP",
        "K9", "BB9", "HR9", "KBB", "QS"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out_rows = []

    for (player_id, year), g in df.groupby(["player_id", "year"], sort=False):
        row = {
            "player_id": str(player_id),
            "year": year,
            # 마지막 팀으로 통일
            "team_code": _get_last_team_code(g),
        }

        # 이름은 마지막 값 사용
        if "player_name" in g.columns:
            name_vals = g["player_name"].dropna()
            row["player_name"] = name_vals.iloc[-1] if len(name_vals) > 0 else np.nan

        # -------------------------
        # counting stat 합산
        # -------------------------
        count_cols = ["GS", "QS"]
        for c in count_cols:
            if c in g.columns:
                row[c] = g[c].sum(min_count=1)

        # -------------------------
        # IP는 decimal 기준으로 합산
        # -------------------------
        if "IP_decimal" in g.columns:
            row["IP_decimal"] = g["IP_decimal"].sum(min_count=1)
        elif "IP" in g.columns:
            row["IP"] = g["IP"].sum(min_count=1)

        # -------------------------
        # rate stat은 IP 가중평균
        # -------------------------
        weight_col = "IP_decimal" if "IP_decimal" in g.columns else ("IP" if "IP" in g.columns else None)
        weighted_rate_cols = ["ERA", "FIP", "WHIP", "K9", "BB9", "HR9", "KBB"]

        for c in weighted_rate_cols:
            if c in g.columns:
                tmp = g[[c]].copy()

                if weight_col is not None:
                    tmp[weight_col] = pd.to_numeric(g[weight_col], errors="coerce")
                    tmp = tmp.dropna(subset=[c])

                    if not tmp.empty and tmp[weight_col].fillna(0).sum() > 0:
                        row[c] = np.average(tmp[c], weights=tmp[weight_col].fillna(0))
                    else:
                        row[c] = tmp[c].mean() if not tmp.empty else np.nan
                else:
                    row[c] = pd.to_numeric(g[c], errors="coerce").mean()

        out_rows.append(row)

    out = pd.DataFrame(out_rows)

    ordered_cols = [
        "player_id", "player_name", "team_code", "year",
        "GS", "IP_decimal", "ERA", "FIP", "WHIP",
        "K9", "BB9", "HR9", "KBB", "QS"
    ]
    ordered_cols = [c for c in ordered_cols if c in out.columns]

    return out[ordered_cols].copy()

def prepare_season_pitchers(season_pit_df: pd.DataFrame) -> pd.DataFrame:
    df = season_pit_df.copy()

    df["player_id"] = df["player_id"].astype(str)

    if "team_code" in df.columns:
        df["team_code"] = df["team_code"].astype(str)

    # year -> season 맞춤
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["season"] = df["year"]

    num_cols = ["GS", "IP", "IP_decimal", "ERA", "FIP", "WHIP", "K9", "BB9", "HR9", "KBB", "QS"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def build_modeling_data_pipeline(
    date_list,
    team_codes=None,
    roster_code="1",
    sleep_sec=0.2,
    verbose=True
):
    """
    date_list를 입력받아
    1) games 수집
    2) 정상 경기만 필터
    3) lineup 수집
    4) roster 수집 (date_list 전체 기준)
    5) player season 수집
    6) 시즌 중 팀 이동 선수 처리 (마지막 팀으로 통일 + 시즌 성적 통합)
    7) pitcher 결측/무한대 처리
    8) 경기 단위 model_df 생성
    까지를 한 번에 수행

    Parameters
    ----------
    date_list : list[str]
        예: ["2025-04-22", "2025-04-23", "2025-04-24"]

    team_codes : list[str], optional
        팀코드 리스트. None이면 KBO 10개 구단 기본값 사용.

    roster_code : str, default "1"
        playerRoster 호출용 code 파라미터

    sleep_sec : float, default 0.2
        반복 API 호출 간 텀 (429 방지)

    verbose : bool, default True
        진행 로그 출력 여부

    Returns
    -------
    result : dict
        {
            "games_df": ...,
            "games_df_clean": ...,
            "lineup_df": ...,
            "roster_df": ...,
            "season_hit_df": ...,
            "season_pit_df": ...,
            "games_model": ...,
            "lineup_model": ...,
            "season_hit_model": ...,
            "season_pit_model": ...,
            "hitter_team_features": ...,
            "sp_features": ...,
            "model_df": ...
        }
    """

    # --------------------------------------------------
    # 0) 기본 팀코드 설정
    # --------------------------------------------------
    if team_codes is None:
        team_codes = ["1001", "2002", "3001", "5002", "6002", "7002", "9002", "10001", "11001", "12001"]
        # 삼성, KIA, 롯데, LG, 두산, 한화, SSG, 키움, NC, KT

    # --------------------------------------------------
    # 1) games 수집
    # --------------------------------------------------
    if verbose:
        print("=== 1. collect_games ===")
    games_df = collect_games(date_list, sleep_sec=sleep_sec)

    # --------------------------------------------------
    # 2) 정상 경기만 필터
    # game_state == 3 이고 점수가 존재하는 경기만 사용
    # --------------------------------------------------
    if verbose:
        print("=== 2. filter normal games ===")

    games_df_clean = games_df[
        (games_df["game_state"] == 3) &
        (games_df["home_score"].notna()) &
        (games_df["away_score"].notna())
    ].copy()

    if verbose:
        print(f"games_df.shape       = {games_df.shape}")
        print(f"games_df_clean.shape = {games_df_clean.shape}")

    # --------------------------------------------------
    # 3) lineup 수집
    # 정상 경기만 대상으로 호출
    # --------------------------------------------------
    if verbose:
        print("=== 3. collect_lineups ===")

    lineup_df = collect_lineups(games_df_clean, sleep_sec=sleep_sec)

    if verbose:
        print(f"lineup_df.shape = {lineup_df.shape}")

    # --------------------------------------------------
    # 4) roster 수집
    # 특정 하루만 수집하면 선수 누락 가능하므로
    # date_list 전체 기준으로 roster를 합침
    # --------------------------------------------------
    if verbose:
        print("=== 4. collect_rosters (all dates) ===")

    roster_dfs = []

    for dt in date_list:
        if verbose:
            print(f"roster date: {dt}")

        tmp_roster = collect_rosters(
            date=dt,
            team_codes=team_codes,
            code=roster_code,
            sleep_sec=sleep_sec
        )

        if not tmp_roster.empty:
            roster_dfs.append(tmp_roster)

    if roster_dfs:
        roster_df = pd.concat(roster_dfs, ignore_index=True).drop_duplicates()
    else:
        roster_df = pd.DataFrame(columns=["snapshot_date", "team_code", "player_id", "player_name", "join_date"])

    if verbose:
        print(f"roster_df.shape = {roster_df.shape}")

    # --------------------------------------------------
    # 5) player_ids 추출
    # --------------------------------------------------
    if roster_df.empty:
        raise ValueError("roster_df가 비어 있습니다. playerSeason 수집을 진행할 수 없습니다.")

    player_ids = roster_df["player_id"].astype(str).unique().tolist()

    if verbose:
        print(f"n_unique_player_ids = {len(player_ids)}")

    # --------------------------------------------------
    # 6) playerSeason 수집
    # --------------------------------------------------
    snapshot_date = date_list[0]

    if verbose:
        print("=== 5. collect_player_season ===")

    season_hit_df, season_pit_df = collect_player_season(
        player_ids,
        snapshot_date=snapshot_date,
        sleep_sec=sleep_sec
    )

    # --------------------------------------------------
    # 7) target season 필터
    # 현재 pipeline은 date_list가 한 시즌이라고 가정
    # --------------------------------------------------
    target_year = pd.to_datetime(date_list[0]).year

    season_hit_df["year"] = pd.to_numeric(season_hit_df["year"], errors="coerce")
    season_pit_df["year"] = pd.to_numeric(season_pit_df["year"], errors="coerce")

    season_hit_df = season_hit_df[season_hit_df["year"] == target_year].copy()
    season_pit_df = season_pit_df[season_pit_df["year"] == target_year].copy()

    # --------------------------------------------------
    # 8) 시즌 중 팀 이동 선수 처리
    # 핵심:
    # - player_id + year 기준 1행으로 압축
    # - team_code는 마지막 팀으로 통일
    # - season 성적은 통합
    # --------------------------------------------------
    if verbose:
        print("=== 6. deduplicate season stats (last team + aggregate season) ===")

    season_hit_df = deduplicate_hitter_season_last_team(season_hit_df)
    season_pit_df = deduplicate_pitcher_season_last_team(season_pit_df)

    if verbose:
        print(f"season_hit_df.shape after dedup = {season_hit_df.shape}")
        print(f"season_pit_df.shape after dedup = {season_pit_df.shape}")

    # --------------------------------------------------
    # 9) pitcher 특수값 처리
    # ERA/FIP/WHIP/KBB의 inf, 결측 등을 99.99로 치환
    # --------------------------------------------------
    if verbose:
        print("=== 7. clean pitcher season stats ===")

    pit_fill_cols = [c for c in ["ERA", "FIP", "WHIP", "KBB"] if c in season_pit_df.columns]
    if pit_fill_cols:
        season_pit_df[pit_fill_cols] = (
            season_pit_df[pit_fill_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(99.99)
        )

    # --------------------------------------------------
    # 10) 모델링용 전처리
    # --------------------------------------------------
    if verbose:
        print("=== 8. prepare modeling tables ===")

    games_model = prepare_games_for_modeling(games_df_clean)
    lineup_model = prepare_lineup(lineup_df)
    season_hit_model = prepare_season_hitters(season_hit_df)
    season_pit_model = prepare_season_pitchers(season_pit_df)

    # --------------------------------------------------
    # 11) lineup에 season 붙이기
    # season stats를 player_id + season 기준으로 merge할 수 있게 준비
    # --------------------------------------------------
    lineup_model = lineup_model.merge(
        games_model[["game_id", "season"]],
        on="game_id",
        how="left"
    )

    # --------------------------------------------------
    # 12) 팀 타자 feature 생성
    # 주의:
    # build_hitter_team_features 내부에서
    # season_hit_model과 merge할 때 on=["player_id", "season"] 기준이 되도록
    # 함수도 같이 수정해서 쓰는 것을 권장
    # --------------------------------------------------
    if verbose:
        print("=== 9. build hitter team features ===")

    hitter_team_features = build_hitter_team_features(lineup_model, season_hit_model)

    if verbose:
        print(f"hitter_team_features.shape = {hitter_team_features.shape}")

    # --------------------------------------------------
    # 13) 선발투수 feature 생성
    # 주의:
    # build_starting_pitcher_features 내부에서
    # season_pit_model과 merge할 때 on=["player_id", "season"] 기준이 되도록
    # 함수도 같이 수정해서 쓰는 것을 권장
    # --------------------------------------------------
    if verbose:
        print("=== 10. build starting pitcher features ===")

    sp_features = build_starting_pitcher_features(lineup_model, season_pit_model)

    if verbose:
        print(f"sp_features.shape = {sp_features.shape}")

        # 경기-팀당 1행인지 확인
        if not sp_features.empty:
            print("sp_features rows per (game_id, team_code):")
            print(sp_features.groupby(["game_id", "team_code"]).size().value_counts().head())

    # --------------------------------------------------
    # 14) 최종 경기 단위 데이터셋 생성
    # --------------------------------------------------
    if verbose:
        print("=== 11. make game-level dataset ===")

    model_df = make_game_level_dataset(
        games_df_clean=games_model,
        hitter_team_features=hitter_team_features,
        sp_features=sp_features
    )

    if verbose:
        print(f"model_df.shape = {model_df.shape}")

        # 같은 경기 중복 여부 확인
        print("game_id duplication check:")
        print(model_df["game_id"].value_counts().value_counts().head())

    # --------------------------------------------------
    # 15) diff feature 결측 확인
    # --------------------------------------------------
    if verbose:
        diff_cols = [c for c in model_df.columns if c.endswith("_diff")]
        if diff_cols:
            na_summary = model_df[diff_cols].isna().sum().sort_values(ascending=False)
            print("Top NA counts in diff features:")
            print(na_summary.head(10))

    return {
        "games_df": games_df,
        "games_df_clean": games_df_clean,
        "lineup_df": lineup_df,
        "roster_df": roster_df,
        "season_hit_df": season_hit_df,
        "season_pit_df": season_pit_df,
        "games_model": games_model,
        "lineup_model": lineup_model,
        "season_hit_model": season_hit_model,
        "season_pit_model": season_pit_model,
        "hitter_team_features": hitter_team_features,
        "sp_features": sp_features,
        "model_df": model_df,
    }

from datetime import datetime, timedelta

def generate_date_list(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date, "%Y-%m-%d")

    date_list = []
    cur = start

    while cur <= end:
        date_list.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    return date_list



# -----------------
def build_prediction_data_for_date(
    date_str,
    team_codes=None,
    roster_code="1",
    sleep_sec=0.2,
    verbose=True
):
    """
    특정 날짜(date_str)의 경기들을 불러와서
    모델 예측에 바로 사용할 수 있는 pred_df를 생성한다.

    Parameters
    ----------
    date_str : str
        예: "2026-04-05"

    team_codes : list[str], optional
        팀코드 리스트. None이면 KBO 10개 구단 기본값 사용.

    roster_code : str, default "1"
        playerRoster 호출용 code 파라미터

    sleep_sec : float, default 0.2
        반복 API 호출 간 텀

    verbose : bool, default True
        진행 로그 출력 여부

    Returns
    -------
    result : dict
        {
            "games_df": ...,
            "games_pred": ...,
            "lineup_df": ...,
            "roster_df": ...,
            "season_hit_df": ...,
            "season_pit_df": ...,
            "lineup_model": ...,
            "season_hit_model": ...,
            "season_pit_model": ...,
            "hitter_team_features": ...,
            "sp_features": ...,
            "pred_df": ...
        }
    """

    # --------------------------------------------------
    # 0) 기본 팀코드
    # --------------------------------------------------
    team_codes = ["1001", "2002", "3001", "5002", "6002", "7002", "9002", "10001", "11001", "12001"]
        # 삼성, KIA, 롯데, LG, 두산, 한화, SSG, 키움, NC, KT

    # --------------------------------------------------
    # 1) 해당 날짜 경기 수집
    # --------------------------------------------------
    if verbose:
        print("=== 1. collect_games ===")

    games_df = collect_games([date_str], sleep_sec=sleep_sec)

    if verbose:
        print(f"games_df.shape = {games_df.shape}")
        display(games_df.head())

    # --------------------------------------------------
    # 2) 예측 대상으로 사용할 경기 필터
    # --------------------------------------------------
    # 주의:
    # - 학습용 pipeline에서는 game_state == 3 & 점수 존재 조건을 썼음
    # - 여기서는 "어제 완료된 경기"라서 동일하게 써도 됨
    # - 만약 미래 경기 예측용이라면 점수 조건은 빼야 함. game_state != 4 활용.
    # --------------------------------------------------
    if verbose:
        print("=== 2. filter target games ===")

    games_pred = games_df[
        (games_df["game_state"] != 4)
    ].copy()

    if verbose:
        
        print(f"games_pred.shape = {games_pred.shape}")
        display(games_pred.head())

    if games_pred.empty:
        raise ValueError("예측 대상으로 사용할 경기가 없습니다. game_state를 확인해봐.")

    # --------------------------------------------------
    # 3) 라인업 수집
    # --------------------------------------------------
    if verbose:
        print("=== 3. collect_lineups ===")

    lineup_df = collect_lineups(games_pred, sleep_sec=sleep_sec)

    if verbose:
        print(f"lineup_df.shape = {lineup_df.shape}")

    # --------------------------------------------------
    # 4) roster 수집
    # --------------------------------------------------
    if verbose:
        print("=== 4. collect_rosters ===")

    roster_df = collect_rosters(
        date=date_str,
        team_codes=team_codes,
        code=roster_code,
        sleep_sec=sleep_sec
    )

    if verbose:
        print(f"roster_df.shape = {roster_df.shape}")

    if roster_df.empty:
        raise ValueError("roster_df가 비어 있습니다. playerSeason 수집이 불가능합니다.")

    # --------------------------------------------------
    # 5) player ids 추출
    # --------------------------------------------------
    player_ids = roster_df["player_id"].astype(str).unique().tolist()

    if verbose:
        print(f"n_unique_player_ids = {len(player_ids)}")

    # --------------------------------------------------
    # 6) playerSeason 수집
    # --------------------------------------------------
    if verbose:
        print("=== 5. collect_player_season ===")

    season_hit_df, season_pit_df = collect_player_season(
        player_ids,
        snapshot_date=date_str,
        sleep_sec=sleep_sec
    )

    target_year = pd.to_datetime(date_str).year

    season_hit_df["year"] = pd.to_numeric(season_hit_df["year"], errors="coerce")
    season_pit_df["year"] = pd.to_numeric(season_pit_df["year"], errors="coerce")

    season_hit_df = season_hit_df[season_hit_df["year"] == target_year].copy()
    season_pit_df = season_pit_df[season_pit_df["year"] == target_year].copy()

    # --------------------------------------------------
    # 7) 시즌 중 팀 이동 선수 처리
    # - player_id + year 기준 1행
    # - team_code는 마지막 팀으로 통일
    # --------------------------------------------------
    if verbose:
        print("=== 6. deduplicate season stats ===")

    season_hit_df = deduplicate_hitter_season_last_team(season_hit_df)
    season_pit_df = deduplicate_pitcher_season_last_team(season_pit_df)

    if verbose:
        print(f"season_hit_df.shape = {season_hit_df.shape}")
        print(f"season_pit_df.shape = {season_pit_df.shape}")

    # --------------------------------------------------
    # 8) pitcher 특수값 처리
    # --------------------------------------------------
    if verbose:
        print("=== 7. clean pitcher season stats ===")

    pit_fill_cols = [c for c in ["ERA", "FIP", "WHIP", "KBB"] if c in season_pit_df.columns]
    if pit_fill_cols:
        season_pit_df[pit_fill_cols] = (
            season_pit_df[pit_fill_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(99.99)
        )

    # --------------------------------------------------
    # 9) 전처리
    # --------------------------------------------------
    if verbose:
        print("=== 8. prepare modeling tables ===")

    # 주의:
    # 여기서는 예측용이므로 prepare_games_for_modeling 대신
    # games_pred를 그대로 쓰되 season/home_win만 맞춰도 충분
    games_pred["game_id"] = games_pred["game_id"].astype(str)
    games_pred["home_team_code"] = games_pred["home_team_code"].astype(str)
    games_pred["away_team_code"] = games_pred["away_team_code"].astype(str)

    if "season" not in games_pred.columns:
        games_pred["season"] = pd.to_datetime(games_pred["game_date"]).dt.year

    lineup_model = prepare_lineup(lineup_df)
    season_hit_model = prepare_season_hitters(season_hit_df)
    season_pit_model = prepare_season_pitchers(season_pit_df)

    # lineup에 season 붙이기
    lineup_model = lineup_model.merge(
        games_pred[["game_id", "season"]],
        on="game_id",
        how="left"
    )

    # --------------------------------------------------
    # 10) 팀 타자 feature 생성
    # --------------------------------------------------
    if verbose:
        print("=== 9. build hitter team features ===")

    hitter_team_features = build_hitter_team_features(lineup_model, season_hit_model)

    if verbose:
        print(f"hitter_team_features.shape = {hitter_team_features.shape}")

    # --------------------------------------------------
    # 11) 선발투수 feature 생성
    # --------------------------------------------------
    if verbose:
        print("=== 10. build starting pitcher features ===")

    sp_features = build_starting_pitcher_features(lineup_model, season_pit_model)

    if verbose:
        print(f"sp_features.shape = {sp_features.shape}")

    # --------------------------------------------------
    # 12) 최종 경기 단위 예측 데이터 생성
    # --------------------------------------------------
    if verbose:
        print("=== 11. make prediction dataframe ===")

    pred_df = make_game_level_dataset(
        games_df_clean=games_pred,
        hitter_team_features=hitter_team_features,
        sp_features=sp_features
    )

    if verbose:
        print(f"pred_df.shape = {pred_df.shape}")
        display(pred_df.head())

    return {
        "games_df": games_df,
        "games_pred": games_pred,
        "lineup_df": lineup_df,
        "roster_df": roster_df,
        "season_hit_df": season_hit_df,
        "season_pit_df": season_pit_df,
        "lineup_model": lineup_model,
        "season_hit_model": season_hit_model,
        "season_pit_model": season_pit_model,
        "hitter_team_features": hitter_team_features,
        "sp_features": sp_features,
        "pred_df": pred_df,
    }






def fill_and_recompute_sp_features(
    pred_df: pd.DataFrame,
    season_pit_df: pd.DataFrame,
    target_year: int,
    use_prev_year: bool = True,
    add_imputed_flag: bool = True,
    weighted_years: dict | None = None
) -> pd.DataFrame:
    """
    선발투수 결측치 채우기 + diff 재계산까지 한 번에 처리

    처리 순서:
    1) 현재 값 유지
    2) 없으면 전년도 시즌 값
    3) 그래도 없으면 가중 리그 평균
    4) diff 재계산

    Parameters
    ----------
    pred_df : pd.DataFrame
        경기 단위 예측 데이터
    season_pit_df : pd.DataFrame
        투수 시즌 데이터 (year, player_id, ERA, FIP, WHIP, K9, BB9, HR9, KBB 포함 가정)
    target_year : int
        예측 연도 (예: 2026)
    use_prev_year : bool, default True
        전년도 시즌 값을 먼저 사용할지 여부
    add_imputed_flag : bool, default True
        home/away 선발투수 결측 대체 여부 flag 추가할지 여부
    weighted_years : dict | None, default None
        리그 평균 계산용 연도 가중치
        예: {2023: 0.2, 2024: 0.3, 2025: 0.5}
        None이면 target_year 직전 3개년 기준 기본값 사용

    Returns
    -------
    pd.DataFrame
        결측치 대체 및 diff 재계산이 완료된 pred_df
    """

    df = pred_df.copy()
    pit = season_pit_df.copy()

    # ----------------------------------------
    # 0) 타입 정리
    # ----------------------------------------
    pit["year"] = pd.to_numeric(pit["year"], errors="coerce")
    pit["player_id"] = pit["player_id"].astype(str)

    stat_cols = ["ERA", "FIP", "WHIP", "K9", "BB9", "HR9", "KBB"]
    for c in stat_cols:
        if c in pit.columns:
            pit[c] = pd.to_numeric(pit[c], errors="coerce")

    # ----------------------------------------
    # 1) 전년도 데이터 준비
    # ----------------------------------------
    prev_year = target_year - 1
    prev_pit = pit[pit["year"] == prev_year].copy()

    prev_lookup = {}
    if use_prev_year and not prev_pit.empty:
        prev_pit = prev_pit.drop_duplicates(subset=["player_id"]).set_index("player_id")

        for stat in stat_cols:
            if stat in prev_pit.columns:
                prev_lookup[stat] = prev_pit[stat].to_dict()

    # ----------------------------------------
    # 2) 가중 리그 평균 계산
    # ----------------------------------------
    # 기본: target_year 직전 3개년
    # 예: target_year=2026 -> {2023:0.2, 2024:0.3, 2025:0.5}
    if weighted_years is None:
        weighted_years = {
            target_year - 3: 0.2,
            target_year - 2: 0.3,
            target_year - 1: 0.5
        }

    league_means = {}

    for stat in stat_cols:
        if stat not in pit.columns:
            league_means[stat] = np.nan
            continue

        weighted_sum = 0.0
        total_weight = 0.0

        for year, weight in weighted_years.items():
            year_df = pit[pit["year"] == year]

            if year_df.empty:
                continue

            year_mean = year_df[stat].mean(skipna=True)

            if pd.notna(year_mean):
                weighted_sum += year_mean * weight
                total_weight += weight

        # 가중평균이 가능하면 사용
        if total_weight > 0:
            league_means[stat] = weighted_sum / total_weight
        else:
            # 백업: 전체 pit 평균
            league_means[stat] = pit[stat].mean(skipna=True)

    # ----------------------------------------
    # 3) imputed flag 생성
    # ----------------------------------------
    if add_imputed_flag:
        if "home_sp_ERA" in df.columns:
            df["home_sp_imputed"] = df["home_sp_ERA"].isna().astype(int)
        if "away_sp_ERA" in df.columns:
            df["away_sp_imputed"] = df["away_sp_ERA"].isna().astype(int)

    # ----------------------------------------
    # 4) 결측치 채우기
    # ----------------------------------------
    side_map = {
        "home": "home_sp_player_id",
        "away": "away_sp_player_id"
    }

    for side, pid_col in side_map.items():

        if pid_col not in df.columns:
            continue

        df[pid_col] = df[pid_col].astype(str)

        for stat in stat_cols:
            col = f"{side}_sp_{stat}"

            if col not in df.columns:
                continue

            # 현재 결측 행
            missing_mask = df[col].isna()

            if not missing_mask.any():
                continue

            # (1) 전년도 값으로 먼저 채우기
            if use_prev_year and stat in prev_lookup:
                df.loc[missing_mask, col] = (
                    df.loc[missing_mask, pid_col].map(prev_lookup[stat])
                )

            # (2) 그래도 비면 가중 리그 평균으로 채우기
            still_missing = df[col].isna()
            if still_missing.any():
                df.loc[still_missing, col] = league_means[stat]

    # ----------------------------------------
    # 5) diff 재계산
    # ----------------------------------------
    diff_pairs = [
        ("sp_ERA_diff", "home_sp_ERA", "away_sp_ERA"),
        ("sp_FIP_diff", "home_sp_FIP", "away_sp_FIP"),
        ("sp_WHIP_diff", "home_sp_WHIP", "away_sp_WHIP"),
        ("sp_K9_diff", "home_sp_K9", "away_sp_K9"),
        ("sp_BB9_diff", "home_sp_BB9", "away_sp_BB9"),
        ("sp_HR9_diff", "home_sp_HR9", "away_sp_HR9"),
        ("sp_KBB_diff", "home_sp_KBB", "away_sp_KBB"),
    ]

    for diff_col, h_col, a_col in diff_pairs:
        if h_col in df.columns and a_col in df.columns:
            df[diff_col] = df[h_col] - df[a_col]

    return df



def build_season_pit_df_from_model_dfs(df_dict: dict[int, pd.DataFrame]) -> pd.DataFrame:

    """
    model_df_2023/2024/2025 같은 경기 단위 데이터에서
    시즌별 선발투수 요약 테이블(season_pit_df) 생성

    Parameters
    ----------
    df_dict : dict[int, pd.DataFrame]
        예: {2023: model_df_2023, 2024: model_df_2024, 2025: model_df_2025}

    Returns
    -------
    pd.DataFrame
        columns:
        ['year', 'player_id', 'player_name', 'ERA', 'FIP', 'WHIP',
         'K9', 'BB9', 'HR9', 'KBB', 'QS', 'GS']
    """

    all_pitchers = []

    for year, df in df_dict.items():
        tmp = df.copy()

        # home
        home_cols = [
            "home_sp_player_id", "home_sp_player_name",
            "home_sp_ERA", "home_sp_FIP", "home_sp_WHIP",
            "home_sp_K9", "home_sp_BB9", "home_sp_HR9", "home_sp_KBB",
            "home_sp_QS", "home_sp_GS"
        ]
        home = tmp[home_cols].copy()
        home.columns = [
            "player_id", "player_name",
            "ERA", "FIP", "WHIP",
            "K9", "BB9", "HR9", "KBB",
            "QS", "GS"
        ]
        home["year"] = year

        # away
        away_cols = [
            "away_sp_player_id", "away_sp_player_name",
            "away_sp_ERA", "away_sp_FIP", "away_sp_WHIP",
            "away_sp_K9", "away_sp_BB9", "away_sp_HR9", "away_sp_KBB",
            "away_sp_QS", "away_sp_GS"
        ]
        away = tmp[away_cols].copy()
        away.columns = [
            "player_id", "player_name",
            "ERA", "FIP", "WHIP",
            "K9", "BB9", "HR9", "KBB",
            "QS", "GS"
        ]
        away["year"] = year

        season_pitchers = pd.concat([home, away], ignore_index=True)

        # 타입 정리
        season_pitchers["year"] = pd.to_numeric(season_pitchers["year"], errors="coerce")
        season_pitchers["player_id"] = season_pitchers["player_id"].astype(str)

        num_cols = ["ERA", "FIP", "WHIP", "K9", "BB9", "HR9", "KBB", "QS", "GS"]
        for c in num_cols:
            season_pitchers[c] = pd.to_numeric(season_pitchers[c], errors="coerce")

        # player_id 없는 행 제거
        season_pitchers = season_pitchers[
            season_pitchers["player_id"].notna() &
            (season_pitchers["player_id"] != "nan")
        ].copy()

        # 같은 시즌, 같은 투수가 여러 경기에서 반복되므로 1행만 남김
        # 혹시 값이 조금이라도 다르면 GS 큰 값 기준 마지막 기록 사용
        season_pitchers = (
            season_pitchers
            .sort_values(["year", "player_id", "GS"])
            .drop_duplicates(subset=["year", "player_id"], keep="last")
        )

        all_pitchers.append(season_pitchers)

    season_pit_df = pd.concat(all_pitchers, ignore_index=True)

    # 보기 좋게 정렬
    season_pit_df = season_pit_df.sort_values(["year", "player_id"]).reset_index(drop=True)

    return season_pit_df