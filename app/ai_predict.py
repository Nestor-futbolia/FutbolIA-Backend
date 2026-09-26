import os
import math
from datetime import datetime, timezone
from typing import Any

import httpx


SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "")


FEATURES = [
    "home_goals_for_5",
    "home_goals_against_5",
    "home_points_5",
    "away_goals_for_5",
    "away_goals_against_5",
    "away_points_5",
    "goals_form_difference",
    "points_form_difference",
    "home_advantage",
]


def get_headers() -> dict:
    return {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Content-Type": "application/json",
    }


def require_config() -> None:
    if not SUPABASE_URL:
        raise RuntimeError("Falta SUPABASE_URL")

    if not SUPABASE_SECRET_KEY:
        raise RuntimeError("Falta SUPABASE_SECRET_KEY")


async def supabase_get(
    table: str,
    params: dict[str, Any] | None = None,
) -> list[dict]:
    require_config()

    url = f"{SUPABASE_URL}/rest/v1/{table}"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            url,
            headers=get_headers(),
            params=params or {},
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase GET {table}: "
            f"{response.status_code} {response.text}"
        )

    return response.json()


async def supabase_post(
    table: str,
    payload: dict[str, Any],
) -> dict:
    require_config()

    url = f"{SUPABASE_URL}/rest/v1/{table}"

    headers = get_headers()
    headers["Prefer"] = "return=representation"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            url,
            headers=headers,
            json=payload,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase POST {table}: "
            f"{response.status_code} {response.text}"
        )

    data = response.json()

    if isinstance(data, list):
        return data[0] if data else {}

    return data


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def rolling_form(
    matches: list[dict],
    team_id: int,
    before_date: str,
    limit: int = 5,
) -> tuple[float, float, float]:

    previous = []

    for match in matches:
        starting_at = match.get("starting_at")

        if not starting_at:
            continue

        if starting_at >= before_date:
            continue

        home_id = match.get("home_team_id")
        away_id = match.get("away_team_id")

        if team_id not in (home_id, away_id):
            continue

        home_goals = match.get("home_goals")
        away_goals = match.get("away_goals")

        if home_goals is None or away_goals is None:
            continue

        home_goals = safe_float(home_goals)
        away_goals = safe_float(away_goals)

        if team_id == home_id:
            goals_for = home_goals
            goals_against = away_goals

            if home_goals > away_goals:
                points = 3
            elif home_goals == away_goals:
                points = 1
            else:
                points = 0

        else:
            goals_for = away_goals
            goals_against = home_goals

            if away_goals > home_goals:
                points = 3
            elif away_goals == home_goals:
                points = 1
            else:
                points = 0

        previous.append(
            (
                goals_for,
                goals_against,
                points,
            )
        )

    previous = previous[-limit:]

    if not previous:
        return 0.0, 0.0, 0.0

    goals_for = sum(x[0] for x in previous) / len(previous)
    goals_against = sum(x[1] for x in previous) / len(previous)
    points = sum(x[2] for x in previous) / len(previous)

    return goals_for, goals_against, points


async def load_matches_for_features(
    before_date: str,
    home_team_id: int,
    away_team_id: int,
) -> list[dict]:

    params = {
        "select": (
            "id,starting_at,home_team_id,away_team_id,"
            "home_goals,away_goals,status"
        ),
        "starting_at": f"lt.{before_date}",
        "status": "in.(FT,AET,PEN)",
        "order": "starting_at.asc",
        "limit": "1000",
    }

    rows = await supabase_get(
        "matches",
        params,
    )

    relevant = []

    for row in rows:
        home_id = row.get("home_team_id")
        away_id = row.get("away_team_id")

        if (
            home_id == home_team_id
            or away_id == home_team_id
            or home_id == away_team_id
            or away_id == away_team_id
        ):
            relevant.append(row)

    return relevant


async def build_features(
    match: dict,
) -> dict[str, float]:

    home_team_id = match.get("home_team_id")
    away_team_id = match.get("away_team_id")
    starting_at = match.get("starting_at")

    if home_team_id is None:
        raise RuntimeError("El partido no tiene home_team_id")

    if away_team_id is None:
        raise RuntimeError("El partido no tiene away_team_id")

    if not starting_at:
        raise RuntimeError("El partido no tiene starting_at")

    previous_matches = await load_matches_for_features(
        starting_at,
        int(home_team_id),
        int(away_team_id),
    )

    (
        home_gf,
        home_ga,
        home_points,
    ) = rolling_form(
        previous_matches,
        int(home_team_id),
        starting_at,
        5,
    )

    (
        away_gf,
        away_ga,
        away_points,
    ) = rolling_form(
        previous_matches,
        int(away_team_id),
        starting_at,
        5,
    )

    return {
        "home_goals_for_5": home_gf,
        "home_goals_against_5": home_ga,
        "home_points_5": home_points,
        "away_goals_for_5": away_gf,
        "away_goals_against_5": away_ga,
        "away_points_5": away_points,
        "goals_form_difference": (
            (home_gf - home_ga)
            - (away_gf - away_ga)
        ),
        "points_form_difference": (
            home_points - away_points
        ),
        "home_advantage": 1.0,
    }


async def load_active_model() -> dict:
    rows = await supabase_get(
        "model_versions",
        {
            "select": "*",
            "active": "eq.true",
            "order": "trained_at.desc",
            "limit": "1",
        },
    )

    if not rows:
        raise RuntimeError(
            "No existe un modelo IA activo. "
            "Ejecuta primero train_ai.py."
        )

    model = rows[0]

    metrics = model.get("metrics")

    if not isinstance(metrics, dict):
        raise RuntimeError(
            "El modelo activo no contiene los parámetros."
        )

    if metrics.get("model_type") != "logistic_regression":
        raise RuntimeError(
            "Tipo de modelo no compatible."
        )

    return model


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)

    z = math.exp(value)
    return z / (1.0 + z)


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)

    exp_values = [
        math.exp(value - maximum)
        for value in values
    ]

    total = sum(exp_values)

    if total == 0:
        return [
            1.0 / len(values)
            for _ in values
        ]

    return [
        value / total
        for value in exp_values
    ]


def predict_from_model(
    model: dict,
    features: dict[str, float],
) -> dict[str, float]:

    metrics = model["metrics"]

    feature_names = metrics["features"]
    means = metrics["scaler_mean"]
    scales = metrics["scaler_scale"]
    coefficients = metrics["coefficients"]
    intercept = metrics["intercept"]
    classes = metrics["classes"]

    vector = []

    for feature_name in feature_names:
        value = safe_float(
            features.get(feature_name),
            0.0,
        )

        index = feature_names.index(feature_name)

        mean = safe_float(means[index])
        scale = safe_float(scales[index], 1.0)

        if scale == 0:
            scale = 1.0

        standardized = (
            (value - mean) / scale
        )

        vector.append(standardized)

    scores = []

    for class_index in range(len(classes)):
        coefficient_row = coefficients[class_index]

        score = safe_float(
            intercept[class_index]
            if isinstance(intercept, list)
            else intercept
        )

        for i, coefficient in enumerate(
            coefficient_row
        ):
            score += (
                safe_float(coefficient)
                * vector[i]
            )

        scores.append(score)

    probabilities = softmax(scores)

    result = {}

    for class_name, probability in zip(
        classes,
        probabilities,
    ):
        result[str(class_name)] = float(
            probability
        )

    return result


async def predict_match(
    match_id: int,
) -> dict:

    matches = await supabase_get(
        "matches",
        {
            "select": "*",
            "id": f"eq.{match_id}",
            "limit": "1",
        },
    )

    if not matches:
        raise RuntimeError(
            f"No existe el partido {match_id}"
        )

    match = matches[0]

    status = str(
        match.get("status") or ""
    ).upper()

    if status in {
        "FT",
        "AET",
        "PEN",
    }:
        raise RuntimeError(
            "El partido ya terminó. "
            "No se puede generar una predicción pre-partido."
        )

    features = await build_features(match)

    model = await load_active_model()

    probabilities = predict_from_model(
        model,
        features,
    )

    local_probability = probabilities.get(
        "0",
        0.0,
    )

    draw_probability = probabilities.get(
        "1",
        0.0,
    )

    away_probability = probabilities.get(
        "2",
        0.0,
    )

    payload = {
        "match_id": match_id,
        "model_version": model["version"],
        "market": "1X2",
        "selection": "HOME",
        "probability": local_probability,
        "predicted_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "features_snapshot": {
            "features": features,
            "probabilities": {
                "home": local_probability,
                "draw": draw_probability,
                "away": away_probability,
            },
        },
    }

    saved = await supabase_post(
        "predictions",
        payload,
    )

    return {
        "ok": True,
        "match_id": match_id,
        "model_version": model["version"],
        "market": "1X2",
        "probabilities": {
            "home": round(
                local_probability * 100,
                2,
            ),
            "draw": round(
                draw_probability * 100,
                2,
            ),
            "away": round(
                away_probability * 100,
                2,
            ),
        },
        "features": features,
        "prediction_id": saved.get("id"),
      }
