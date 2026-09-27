from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from app.nestor_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    build_inference_features,
    target_name,
)


SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    ""
).strip().rstrip("/")

SUPABASE_SECRET_KEY = os.getenv(
    "SUPABASE_SECRET_KEY",
    ""
).strip()


if not SUPABASE_URL:

    raise RuntimeError(
        "Falta SUPABASE_URL"
    )


if not SUPABASE_SECRET_KEY:

    raise RuntimeError(
        "Falta SUPABASE_SECRET_KEY"
    )


HEADERS = {

    "apikey":
        SUPABASE_SECRET_KEY,

    "Authorization":
        f"Bearer {SUPABASE_SECRET_KEY}",

    "Content-Type":
        "application/json",
}


def supabase_url(
    table: str
) -> str:

    return (
        f"{SUPABASE_URL}"
        f"/rest/v1/{table}"
    )


async def supabase_get(
    table: str,
    params: dict[str, Any]
) -> list[dict[str, Any]]:

    async with httpx.AsyncClient(
        timeout=60
    ) as client:

        response = await client.get(

            supabase_url(table),

            headers=HEADERS,

            params=params
        )

    if response.status_code not in (
        200,
        206
    ):

        raise RuntimeError(

            f"Supabase GET {table} "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    return (
        data
        if isinstance(data, list)
        else []
    )


async def supabase_post(
    table: str,
    payload: dict[str, Any]
) -> list[dict[str, Any]]:

    headers = {

        **HEADERS,

        "Prefer":
            "return=representation"
    }

    async with httpx.AsyncClient(
        timeout=60
    ) as client:

        response = await client.post(

            supabase_url(table),

            headers=headers,

            json=payload
        )

    if response.status_code not in (
        200,
        201
    ):

        raise RuntimeError(

            f"Supabase POST {table} "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    return (
        response.json()
        if response.text
        else []
    )


async def load_active_model() -> dict[str, Any]:

    rows = await supabase_get(

        "model_versions",

        {

            "select": (
                "version,"
                "model_name,"
                "trained_at,"
                "training_matches,"
                "metrics,"
                "artifact,"
                "active,"
                "status,"
                "parent_version,"
                "feature_schema_version"
            ),

            "active":
                "eq.true",

            "order":
                "trained_at.desc",

            "limit":
                "1",
        },
    )

    if not rows:

        raise RuntimeError(
            "No existe un modelo activo "
            "en model_versions."
        )

    row = rows[0]

    artifact = row.get(
        "artifact"
    )

    if not isinstance(
        artifact,
        dict
    ):

        metrics = row.get(
            "metrics"
        )

        if (
            isinstance(metrics, dict)
            and "coefficients" in metrics
        ):

            artifact = metrics

        else:

            artifact = None

    if not isinstance(
        artifact,
        dict
    ):

        raise RuntimeError(
            "El modelo activo no contiene "
            "un artefacto compatible."
        )

    row["artifact"] = artifact

    return row


def _softmax(
    logits: list[float],
    temperature: float = 1.0
) -> list[float]:

    t = max(
        float(temperature),
        0.05
    )

    z = [
        value / t
        for value in logits
    ]

    max_z = max(z)

    exps = [

        math.exp(
            value - max_z
        )

        for value in z
    ]

    total = sum(exps)

    return [

        value / total

        for value in exps
    ]


def _artifact_probabilities(
    artifact: dict[str, Any],
    features: list[float]
) -> list[float]:

    coefficients = artifact.get(
        "coefficients"
    )

    intercept = artifact.get(
        "intercept"
    )

    mean = artifact.get(
        "scaler_mean"
    )

    scale = artifact.get(
        "scaler_scale"
    )

    if not all(
        isinstance(
            value,
            list
        )

        for value in (
            coefficients,
            intercept,
            mean,
            scale
        )
    ):

        raise RuntimeError(
            "Artefacto de modelo incompleto."
        )

    logits: list[float] = []

    for class_index in range(3):

        value = float(
            intercept[class_index]
        )

        for feature_index, feature_value in enumerate(
            features
        ):

            denominator = float(
                scale[feature_index]
            )

            if abs(
                denominator
            ) < 1e-12:

                denominator = 1.0

            standardized = (

                float(feature_value)

                - float(
                    mean[feature_index]
                )

            ) / denominator

            value += (

                standardized

                * float(
                    coefficients[
                        class_index
                    ][feature_index]
                )
            )

        logits.append(
            value
        )

    calibration = artifact.get(
        "calibration",
        {}
    )

    if isinstance(
        calibration,
        dict
    ):

        temperature = float(
            calibration.get(
                "temperature",
                1.0
            )
        )

    else:

        temperature = 1.0

    return _softmax(
        logits,
        temperature
    )


def _entropy(
    probabilities: list[float]
) -> float:

    entropy = 0.0

    for value in probabilities:

        if value > 0:

            entropy -= (
                value
                * math.log(value)
            )

    return float(
        entropy
        / math.log(
            len(probabilities)
        )
    )


def _explanation(
    probabilities: list[float],
    snapshot: dict[str, Any]
) -> dict[str, Any]:

    labels = [
        "Local",
        "Empate",
        "Visitante"
    ]

    best_index = max(
        range(3),
        key=lambda i:
            probabilities[i]
    )

    confidence = probabilities[
        best_index
    ]

    entropy = _entropy(
        probabilities
    )

    factors: list[str] = []

    features = snapshot.get(
        "features",
        {}
    )

    if float(
        features.get(
            "points_form_difference",
            0.0
        )
    ) > 0.15:

        factors.append(
            "La forma reciente de puntos "
            "favorece al local."
        )

    elif float(
        features.get(
            "points_form_difference",
            0.0
        )
    ) < -0.15:

        factors.append(
            "La forma reciente de puntos "
            "favorece al visitante."
        )

    if float(
        features.get(
            "goals_form_difference",
            0.0
        )
    ) > 0.20:

        factors.append(
            "El balance reciente de goles "
            "favorece al local."
        )

    elif float(
        features.get(
            "goals_form_difference",
            0.0
        )
    ) < -0.20:

        factors.append(
            "El balance reciente de goles "
            "favorece al visitante."
        )

    if (
        snapshot.get("imputed_home")
        or snapshot.get("imputed_away")
    ):

        factors.append(

            "Hay datos históricos "
            "insuficientes para al menos "
            "uno de los equipos; se aplicó "
            "imputación del modelo."
        )

    if not factors:

        factors.append(

            "La salida combina forma "
            "reciente y ventaja de local "
            "según el modelo 1X2."
        )

    if entropy >= 0.85:

        uncertainty = "alta"

    elif entropy >= 0.60:

        uncertainty = "media"

    else:

        uncertainty = "baja"

    return {

        "selection":
            labels[best_index],

        "confidence":
            round(
                confidence,
                6
            ),

        "uncertainty":
            uncertainty,

        "entropy_normalized":
            round(
                entropy,
                6
            ),

        "factors":
            factors,
    }


async def _team_history(
    team_id: int,
    before: str
) -> list[dict[str, Any]]:

    rows = await supabase_get(

        "matches",

        {

            "select": (
                "id,"
                "starting_at,"
                "status,"
                "home_team_id,"
                "away_team_id,"
                "home_goals,"
                "away_goals"
            ),

            "or":
                (
                    f"(home_team_id.eq.{team_id},"
                    f"away_team_id.eq.{team_id})"
                ),

            "status":
                "in.(FT,AET,PEN)",

            "starting_at":
                f"lt.{before}",

            "order":
                "starting_at.desc,id.desc",

            "limit":
                "20",
        },
    )

    normalized: list[
        dict[str, Any]
    ] = []

    for row in rows:

        normalized.append(
            {
                **row,
                "team_id":
                    team_id
            }
        )

    return normalized


async def _existing_predictions(
    match_id: int,
    model_version: str
) -> list[dict[str, Any]]:

    return await supabase_get(

        "predictions",

        {

            "select":
                "*",

            "match_id":
                f"eq.{match_id}",

            "model_version":
                f"eq.{model_version}",

            "market":
                "eq.1X2",

            "order":
                "selection.asc",
        }
    )


async def predict_match(
    match_id: int
) -> dict[str, Any]:

    model_row = (
        await load_active_model()
    )

    artifact = model_row[
        "artifact"
    ]

    if artifact.get(
        "model_family"
    ) not in (
        None,
        "1X2"
    ):

        raise RuntimeError(

            "El modelo activo no "
            "pertenece a la familia 1X2."
        )

    matches = await supabase_get(

        "matches",

        {

            "select": (
                "id,"
                "starting_at,"
                "status,"
                "home_team_id,"
                "away_team_id,"
                "home_goals,"
                "away_goals,"
                "league_id,"
                "season_id"
            ),

            "id":
                f"eq.{match_id}",

            "limit":
                "1",
        }
    )

    if not matches:

        raise RuntimeError(

            f"No se encontró "
            f"el partido {match_id}."
        )

    match = matches[0]

    home_id = int(
        match[
            "home_team_id"
        ]
    )

    away_id = int(
        match[
            "away_team_id"
        ]
    )

    kickoff = str(
        match[
            "starting_at"
        ]
    )

    home_history = (
        await _team_history(
            home_id,
            kickoff
        )
    )

    away_history = (
        await _team_history(
            away_id,
            kickoff
        )
    )

    defaults = artifact.get(
        "feature_defaults",
        {}
    )

    if not isinstance(
        defaults,
        dict
    ):

        defaults = {}

    features, snapshot = (
        build_inference_features(
            home_history,
            away_history,
            defaults
        )
    )

    probabilities = (
        _artifact_probabilities(
            artifact,
            features
        )
    )

    explanation = (
        _explanation(
            probabilities,
            snapshot
        )
    )

    existing = (
        await _existing_predictions(
            match_id,
            str(
                model_row["version"]
            )
        )
    )

    if len(existing) >= 3:

        selected = max(

            existing,

            key=lambda row:
                float(
                    row.get(
                        "probability",
                        0.0
                    )
                )
        )

        return {

            "ok":
                True,

            "match_id":
                match_id,

            "model_version":
                model_row["version"],

            "market":
                "1X2",

            "prediction":
                selected.get(
                    "selection"
                ),

            "probability":
                float(
                    selected.get(
                        "probability"
                    )
                ),

            "probabilities": {

                row.get(
                    "selection"
                ):
                    float(
                        row.get(
                            "probability"
                        )
                    )

                for row in existing

                if row.get(
                    "selection"
                ) is not None
            },

            "features":
                snapshot,

            "explanation":
                explanation,

            "cached":
                True,
        }

    labels = {

        0: "HOME",

        1: "DRAW",

        2: "AWAY",
    }

    timestamp = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    for index in range(3):

        await supabase_post(

            "predictions",

            {

                "match_id":
                    match_id,

                "model_version":
                    model_row["version"],

                "market":
                    "1X2",

                "selection":
                    labels[index],

                "probability":
                    float(
                        probabilities[index]
                    ),

                "created_at":
                    timestamp,

                "features_snapshot": {

                    **snapshot,

                    "model_version":
                        model_row["version"],

                    "feature_schema_version":
                        artifact.get(
                            "feature_schema_version",
                            FEATURE_SCHEMA_VERSION
                        ),
                },
            },
        )

    best_index = int(
        max(
            range(3),
            key=lambda i:
                probabilities[i]
        )
    )

    return {

        "ok":
            True,

        "match_id":
            match_id,

        "model_version":
            model_row["version"],

        "market":
            "1X2",

        "prediction":
            target_name(
                best_index
            ),

        "probability":
            float(
                probabilities[
                    best_index
                ]
            ),

        "probabilities": {

            "HOME":
                float(
                    probabilities[0]
                ),

            "DRAW":
                float(
                    probabilities[1]
                ),

            "AWAY":
                float(
                    probabilities[2]
                ),
        },

        "features":
            snapshot,

        "explanation":
            explanation,

        "cached":
            False,
    }
