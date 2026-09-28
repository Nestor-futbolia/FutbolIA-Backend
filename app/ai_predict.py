from __future__ import annotations

import math
import os
from typing import Any

import httpx

from app.nestor_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    build_inference_features,
    target_name,
)


SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "").strip()

MODEL_FAMILY = "1X2"
EXPECTED_CLASSES = [0, 1, 2]
LEGACY_CLASSES = ["A", "D", "H"]


if not SUPABASE_URL:
    raise RuntimeError("Falta SUPABASE_URL")

if not SUPABASE_SECRET_KEY:
    raise RuntimeError("Falta SUPABASE_SECRET_KEY")


HEADERS = {
    "apikey": SUPABASE_SECRET_KEY,
    "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
    "Content-Type": "application/json",
}


def supabase_url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


async def supabase_get(
    table: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(
            supabase_url(table),
            headers=HEADERS,
            params=params,
        )

    if response.status_code not in (200, 206):
        raise RuntimeError(
            f"Supabase GET {table} HTTP {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()
    return data if isinstance(data, list) else []


async def supabase_post(
    table: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    headers = {
        **HEADERS,
        "Prefer": "return=representation",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            supabase_url(table),
            headers=headers,
            json=payload,
        )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Supabase POST {table} HTTP {response.status_code}: "
            f"{response.text}"
        )

    return response.json() if response.text else []


# ============================================================
# VALIDACIÓN Y NORMALIZACIÓN DEL ARTEFACTO
# ============================================================


def _as_float_array(
    value: Any,
    name: str,
) -> list[float]:
    if not isinstance(value, list):
        raise RuntimeError(
            f"El campo {name} del artefacto no es una lista."
        )

    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"El campo {name} del artefacto contiene valores inválidos."
        ) from exc


def _validate_parameter_shapes(
    coefficients: Any,
    intercept: Any,
    mean: Any,
    scale: Any,
) -> None:
    if not isinstance(coefficients, list) or len(coefficients) != 3:
        raise RuntimeError(
            "El artefacto debe contener exactamente 3 filas de coeficientes."
        )

    expected_features = len(FEATURE_NAMES)

    for row in coefficients:
        if not isinstance(row, list) or len(row) != expected_features:
            raise RuntimeError(
                "La matriz de coeficientes no coincide con FEATURE_NAMES."
            )

        for value in row:
            if not math.isfinite(float(value)):
                raise RuntimeError(
                    "El artefacto contiene coeficientes no finitos."
                )

    intercept_values = _as_float_array(intercept, "intercept")
    mean_values = _as_float_array(mean, "scaler_mean")
    scale_values = _as_float_array(scale, "scaler_scale")

    if len(intercept_values) != 3:
        raise RuntimeError(
            "El vector intercept debe tener 3 valores."
        )

    if len(mean_values) != expected_features:
        raise RuntimeError(
            "scaler_mean no coincide con FEATURE_NAMES."
        )

    if len(scale_values) != expected_features:
        raise RuntimeError(
            "scaler_scale no coincide con FEATURE_NAMES."
        )

    if any(not math.isfinite(value) for value in intercept_values):
        raise RuntimeError(
            "El artefacto contiene interceptos no finitos."
        )

    if any(not math.isfinite(value) for value in mean_values):
        raise RuntimeError(
            "El artefacto contiene scaler_mean no finitos."
        )

    if any(
        not math.isfinite(value) or abs(value) < 1e-12
        for value in scale_values
    ):
        raise RuntimeError(
            "El artefacto contiene scaler_scale inválido o cero."
        )


def _validate_temperature(artifact: dict[str, Any]) -> float:
    calibration = artifact.get("calibration", {})

    if not isinstance(calibration, dict):
        raise RuntimeError(
            "El bloque calibration del artefacto es inválido."
        )

    try:
        temperature = float(
            calibration.get("temperature", 1.0)
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "La temperatura de calibración del artefacto es inválida."
        ) from exc

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError(
            "La temperatura de calibración del artefacto es inválida."
        )

    return temperature


def _native_artifact_is_compatible(
    artifact: dict[str, Any],
) -> bool:
    if artifact.get("model_type") != "logistic_regression":
        return False

    if artifact.get("model_family") not in (
        None,
        MODEL_FAMILY,
    ):
        return False

    if artifact.get("features") != FEATURE_NAMES:
        return False

    schema = artifact.get("feature_schema_version")

    if schema not in (
        None,
        FEATURE_SCHEMA_VERSION,
    ):
        return False

    if artifact.get("classes") != EXPECTED_CLASSES:
        return False

    try:
        _validate_parameter_shapes(
            artifact.get("coefficients"),
            artifact.get("intercept"),
            artifact.get("scaler_mean"),
            artifact.get("scaler_scale"),
        )

        _validate_temperature(artifact)

    except Exception:
        return False

    return True


def _convert_legacy_artifact(
    legacy: dict[str, Any],
) -> dict[str, Any]:
    if legacy.get("model_type") != "multinomial_logistic_regression":
        raise RuntimeError(
            "El modelo activo antiguo no usa un formato compatible."
        )

    if legacy.get("classes") != LEGACY_CLASSES:
        raise RuntimeError(
            "El modelo legacy no tiene el orden de clases esperado [A,D,H]."
        )

    if legacy.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError(
            "El modelo legacy no usa el mismo esquema de features de NESTOR."
        )

    _validate_parameter_shapes(
        legacy.get("coefficients"),
        legacy.get("intercept"),
        legacy.get("scaler_mean"),
        legacy.get("scaler_scale"),
    )

    coefficients = [
        [float(value) for value in row]
        for row in legacy["coefficients"]
    ]

    intercept = [
        float(value)
        for value in legacy["intercept"]
    ]

    scaler_mean = [
        float(value)
        for value in legacy["scaler_mean"]
    ]

    scaler_scale = [
        float(value)
        for value in legacy["scaler_scale"]
    ]

    # Legacy:
    #   A = away
    #   D = draw
    #   H = home
    #
    # NESTOR:
    #   0 = home
    #   1 = draw
    #   2 = away
    #
    # Therefore:
    # [A,D,H] -> [H,D,A] -> [0,1,2]

    legacy_to_nestor = [2, 1, 0]

    converted_coefficients = [
        coefficients[index]
        for index in legacy_to_nestor
    ]

    converted_intercept = [
        intercept[index]
        for index in legacy_to_nestor
    ]

    # El modelo legacy no guardaba feature_defaults.
    # Usamos el centro de entrenamiento del scaler como fallback
    # para equipos sin historial previo.

    feature_defaults = {
        feature_name: float(value)
        for feature_name, value in zip(
            FEATURE_NAMES,
            scaler_mean,
        )
    }

    return {
        "format_version": "NESTOR-LEGACY-BRIDGE-v1",
        "model_type": "logistic_regression",
        "model_family": MODEL_FAMILY,
        "features": FEATURE_NAMES,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "classes": EXPECTED_CLASSES,
        "coefficients": converted_coefficients,
        "intercept": converted_intercept,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "calibration": {
            "method": "temperature_scaling",
            "temperature": 1.0,
        },
        "feature_defaults": feature_defaults,
        "legacy_source_model_type": legacy.get(
            "model_type"
        ),
        "legacy_source_classes": list(
            LEGACY_CLASSES
        ),
        "legacy_class_conversion": (
            "[A,D,H] -> [H,D,A] -> [0,1,2]"
        ),
    }


def normalize_model_artifact(
    row: dict[str, Any],
) -> tuple[dict[str, Any], str]:

    artifact = row.get("artifact")

    if (
        isinstance(artifact, dict)
        and _native_artifact_is_compatible(artifact)
    ):
        normalized = dict(artifact)

        normalized.setdefault(
            "model_family",
            MODEL_FAMILY,
        )

        normalized.setdefault(
            "feature_schema_version",
            FEATURE_SCHEMA_VERSION,
        )

        return normalized, "strict_native"

    metrics = row.get("metrics")

    if not isinstance(metrics, dict):
        raise RuntimeError(
            "El modelo activo no contiene artifact ni metrics compatibles."
        )

    # El campeón histórico guarda el modelo dentro
    # de metrics usando el formato legacy.

    if metrics.get(
        "model_type"
    ) == "multinomial_logistic_regression":

        return (
            _convert_legacy_artifact(metrics),
            "legacy_multinomial_bridge",
        )

    # Compatibilidad adicional con modelos antiguos
    # que guardaron el artifact nativo dentro de metrics.

    if _native_artifact_is_compatible(metrics):

        normalized = dict(metrics)

        normalized.setdefault(
            "model_family",
            MODEL_FAMILY,
        )

        normalized.setdefault(
            "feature_schema_version",
            FEATURE_SCHEMA_VERSION,
        )

        return (
            normalized,
            "native_metrics_bridge",
        )

    raise RuntimeError(
        "El modelo activo no contiene un artefacto 1X2 compatible con NESTOR."
    )


# ============================================================
# MODELO ACTIVO
# ============================================================


async def load_active_model() -> dict[str, Any]:

    rows = await supabase_get(
        "model_versions",
        {
            "select": (
                "version,model_name,trained_at,training_matches,"
                "metrics,artifact,active,status,parent_version,"
                "feature_schema_version,training_config"
            ),
            "active": "eq.true",
            "order": "trained_at.desc",
            "limit": "2",
        },
    )

    if not rows:
        raise RuntimeError(
            "No existe un modelo activo en model_versions."
        )

    if len(rows) > 1:
        versions = [
            str(row.get("version"))
            for row in rows
        ]

        raise RuntimeError(
            "Hay más de un modelo activo en model_versions: "
            + ", ".join(versions)
        )

    row = rows[0]

    artifact, compatibility = normalize_model_artifact(
        row
    )

    row["artifact"] = artifact

    row["inference_compatibility"] = (
        compatibility
    )

    row["inference_ready"] = True

    return row


# ============================================================
# PROBABILIDADES
# ============================================================


def _softmax(
    logits: list[float],
    temperature: float = 1.0,
) -> list[float]:

    t = max(
        float(temperature),
        0.05,
    )

    z = [
        value / t
        for value in logits
    ]

    max_z = max(z)

    exps = [
        math.exp(value - max_z)
        for value in z
    ]

    total = sum(exps)

    if (
        total <= 0.0
        or not math.isfinite(total)
    ):
        raise RuntimeError(
            "No se pudieron calcular las probabilidades del modelo."
        )

    return [
        value / total
        for value in exps
    ]


def _artifact_logits(
    artifact: dict[str, Any],
    features: list[float],
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

    _validate_parameter_shapes(
        coefficients,
        intercept,
        mean,
        scale,
    )

    if len(features) != len(
        FEATURE_NAMES
    ):
        raise RuntimeError(
            "El vector de features de inferencia no coincide con el esquema NESTOR."
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

            if abs(denominator) < 1e-12:
                denominator = 1.0

            standardized = (
                float(feature_value)
                - float(mean[feature_index])
            ) / denominator

            value += (
                standardized
                * float(
                    coefficients[class_index][
                        feature_index
                    ]
                )
            )

        if not math.isfinite(value):
            raise RuntimeError(
                "El modelo generó un logit no finito."
            )

        logits.append(value)

    return logits


def _artifact_probabilities(
    artifact: dict[str, Any],
    features: list[float],
) -> tuple[list[float], float]:

    logits = _artifact_logits(
        artifact,
        features,
    )

    temperature = _validate_temperature(
        artifact
    )

    return (
        _softmax(
            logits,
            temperature,
        ),
        temperature,
    )


# ============================================================
# EXPLICACIÓN NESTOR
# ============================================================


def _entropy(
    probabilities: list[float],
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
    snapshot: dict[str, Any],
) -> dict[str, Any]:

    labels = [
        "Local",
        "Empate",
        "Visitante",
    ]

    best_index = max(
        range(3),
        key=lambda index: probabilities[index],
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
        {},
    )

    if float(
        features.get(
            "points_form_difference",
            0.0,
        )
    ) > 0.15:

        factors.append(
            "La forma reciente de puntos favorece al local."
        )

    elif float(
        features.get(
            "points_form_difference",
            0.0,
        )
    ) < -0.15:

        factors.append(
            "La forma reciente de puntos favorece al visitante."
        )

    if float(
        features.get(
            "goals_form_difference",
            0.0,
        )
    ) > 0.20:

        factors.append(
            "El balance reciente de goles favorece al local."
        )

    elif float(
        features.get(
            "goals_form_difference",
            0.0,
        )
    ) < -0.20:

        factors.append(
            "El balance reciente de goles favorece al visitante."
        )

    if (
        snapshot.get("imputed_home")
        or snapshot.get("imputed_away")
    ):

        factors.append(
            "Hay datos históricos insuficientes para al menos uno de los equipos; "
            "se aplicó la imputación definida por el modelo."
        )

    if not factors:

        factors.append(
            "La salida combina forma reciente y ventaja de local según el modelo 1X2."
        )

    if entropy >= 0.85:
        uncertainty = "alta"

    elif entropy >= 0.60:
        uncertainty = "media"

    else:
        uncertainty = "baja"

    return {
        "selection": labels[
            best_index
        ],
        "confidence": round(
            confidence,
            6,
        ),
        "uncertainty": uncertainty,
        "entropy_normalized": round(
            entropy,
            6,
        ),
        "factors": factors,
    }


# ============================================================
# HISTORIAL PREPARTIDO
# ============================================================


async def _team_history(
    team_id: int,
    before: str,
) -> list[dict[str, Any]]:

    rows = await supabase_get(
        "matches",
        {
            "select": (
                "id,starting_at,status,home_team_id,away_team_id,"
                "home_goals,away_goals"
            ),
            "or": (
                f"(home_team_id.eq.{team_id},"
                f"away_team_id.eq.{team_id})"
            ),
            "status": "in.(FT,AET,PEN)",
            "starting_at": f"lt.{before}",
            "order": "starting_at.desc,id.desc",
            "limit": "20",
        },
    )

    normalized: list[
        dict[str, Any]
    ] = []

    for row in rows:

        try:

            normalized.append(
                {
                    **row,
                    "team_id": team_id,
                }
            )

        except Exception:
            continue

    return normalized


# ============================================================
# CACHE DE PREDICCIONES
# ============================================================


async def _existing_predictions(
    match_id: int,
    model_version: str,
) -> list[dict[str, Any]]:

    return await supabase_get(
        "predictions",
        {
            "select": "*",
            "match_id": f"eq.{match_id}",
            "model_version": f"eq.{model_version}",
            "market": "eq.1X2",
            "order": "selection.asc",
        },
    )


# ============================================================
# PREDICCIÓN DE PRODUCCIÓN
# ============================================================


async def predict_match(
    match_id: int,
) -> dict[str, Any]:

    model_row = await load_active_model()

    artifact = model_row[
        "artifact"
    ]

    if artifact.get(
        "model_family"
    ) not in (
        None,
        MODEL_FAMILY,
    ):

        raise RuntimeError(
            "El modelo activo no pertenece a la familia 1X2."
        )

    matches = await supabase_get(
        "matches",
        {
            "select": (
                "id,starting_at,status,home_team_id,away_team_id,"
                "home_goals,away_goals,league_id,season_id"
            ),
            "id": f"eq.{match_id}",
            "limit": "1",
        },
    )

    if not matches:

        raise RuntimeError(
            f"No se encontró el partido {match_id}."
        )

    match = matches[0]

    if match.get(
        "starting_at"
    ) is None:

        raise RuntimeError(
            f"El partido {match_id} no tiene starting_at."
        )

    home_id = int(
        match["home_team_id"]
    )

    away_id = int(
        match["away_team_id"]
    )

    kickoff = str(
        match["starting_at"]
    )

    home_history = await _team_history(
        home_id,
        kickoff,
    )

    away_history = await _team_history(
        away_id,
        kickoff,
    )

    defaults = artifact.get(
        "feature_defaults",
        {},
    )

    if not isinstance(
        defaults,
        dict,
    ):
        defaults = {}

    features, snapshot = build_inference_features(
        home_history,
        away_history,
        defaults,
    )

    probabilities, temperature = _artifact_probabilities(
        artifact,
        features,
    )

    explanation = _explanation(
        probabilities,
        snapshot,
    )

    model_version = str(
        model_row["version"]
    )

    existing = await _existing_predictions(
        match_id,
        model_version,
    )

    if len(existing) >= 3:

        selected = max(
            existing,
            key=lambda row: float(
                row.get(
                    "probability",
                    0.0,
                )
            ),
        )

        return {
            "ok": True,
            "match_id": match_id,
            "model_version": model_version,
            "model_name": model_row.get(
                "model_name"
            ),
            "market": "1X2",
            "prediction": selected.get(
                "selection"
            ),
            "probability": float(
                selected.get(
                    "probability"
                )
            ),
            "probabilities": {
                row.get(
                    "selection"
                ): float(
                    row.get(
                        "probability"
                    )
                )
                for row in existing
                if row.get(
                    "selection"
                ) is not None
            },
            "features": snapshot,
            "explanation": explanation,
            "calibration_temperature": float(
                temperature
            ),
            "inference_compatibility": model_row.get(
                "inference_compatibility"
            ),
            "cached": True,
        }

    labels = {
        0: "HOME",
        1: "DRAW",
        2: "AWAY",
    }

    for index in range(3):

        await supabase_post(
            "predictions",
            {
                "match_id": match_id,
                "model_version": model_version,
                "market": "1X2",
                "selection": labels[index],
                "probability": float(
                    probabilities[index]
                ),
                "features_snapshot": {
                    **snapshot,
                    "model_version": model_version,
                    "feature_schema_version": artifact.get(
                        "feature_schema_version",
                        FEATURE_SCHEMA_VERSION,
                    ),
                    "inference_compatibility": model_row.get(
                        "inference_compatibility"
                    ),
                    "calibration_temperature": float(
                        temperature
                    ),
                },
            },
        )

    best_index = int(
        max(
            range(3),
            key=lambda index: probabilities[index],
        )
    )

    return {
        "ok": True,
        "match_id": match_id,
        "model_version": model_version,
        "model_name": model_row.get(
            "model_name"
        ),
        "market": "1X2",
        "prediction": target_name(
            best_index
        ),
        "probability": float(
            probabilities[best_index]
        ),
        "probabilities": {
            "HOME": float(
                probabilities[0]
            ),
            "DRAW": float(
                probabilities[1]
            ),
            "AWAY": float(
                probabilities[2]
            ),
        },
        "features": snapshot,
        "explanation": explanation,
        "calibration_temperature": float(
            temperature
        ),
        "inference_compatibility": model_row.get(
            "inference_compatibility"
        ),
        "cached": False,
            } 
