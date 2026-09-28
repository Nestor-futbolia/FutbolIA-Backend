from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from app.nestor_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    build_training_dataset,
    clean_match,
)


# ============================================================
# CONFIGURACIÓN NESTOR
# ============================================================

SUPABASE_URL = (
    os.getenv("SUPABASE_URL", "")
    .strip()
    .rstrip("/")
)

SUPABASE_SECRET_KEY = (
    os.getenv("SUPABASE_SECRET_KEY", "")
    .strip()
)

MODEL_NAME = "NESTOR-1X2-LogisticRegression"

MODEL_FAMILY = "1X2"

MODEL_PROTOCOL_VERSION = "NESTOR-EVAL-v1.3"

MIN_MATCHES = 80

WINDOW = 5

PAGE_SIZE = 1000

CALIBRATION_GRID = np.linspace(
    0.50,
    3.00,
    101,
)

ACTIVATION_MARGIN = 0.001

TRAIN_RATIO = 0.60

CALIBRATION_RATIO = 0.20


# ============================================================
# VALIDACIÓN DE VARIABLES DE ENTORNO
# ============================================================

if not SUPABASE_URL:
    raise RuntimeError(
        "Falta SUPABASE_URL"
    )

if not SUPABASE_SECRET_KEY:
    raise RuntimeError(
        "Falta SUPABASE_SECRET_KEY"
    )


HEADERS = {
    "apikey": SUPABASE_SECRET_KEY,
    "Authorization": (
        f"Bearer {SUPABASE_SECRET_KEY}"
    ),
    "Content-Type": "application/json",
}


# ============================================================
# SUPABASE
# ============================================================

def supabase_url(
    table: str,
) -> str:

    return (
        f"{SUPABASE_URL}/rest/v1/{table}"
    )


def supabase_get(
    table: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:

    with httpx.Client(
        timeout=60.0
    ) as client:

        response = client.get(
            supabase_url(table),
            headers=HEADERS,
            params=params,
        )

    if response.status_code not in (
        200,
        206,
    ):

        raise RuntimeError(
            f"Supabase GET {table} "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    if not response.text:
        return []

    data = response.json()

    if not isinstance(
        data,
        list,
    ):

        return []

    return data


def supabase_post(
    table: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:

    headers = {
        **HEADERS,
        "Prefer": (
            "resolution=merge-duplicates,"
            "return=representation"
        ),
    }

    with httpx.Client(
        timeout=60.0
    ) as client:

        response = client.post(
            supabase_url(table),
            headers=headers,
            json=payload,
        )

    if response.status_code not in (
        200,
        201,
    ):

        raise RuntimeError(
            f"Supabase POST {table} "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    if not response.text:
        return []

    data = response.json()

    return (
        data
        if isinstance(data, list)
        else []
    )


def supabase_patch(
    table: str,
    params: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:

    headers = {
        **HEADERS,
        "Prefer": "return=representation",
    }

    with httpx.Client(
        timeout=60.0
    ) as client:

        response = client.patch(
            supabase_url(table),
            headers=headers,
            params=params,
            json=payload,
        )

    if response.status_code not in (
        200,
        204,
    ):

        raise RuntimeError(
            f"Supabase PATCH {table} "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    if not response.text:
        return []

    data = response.json()

    return (
        data
        if isinstance(data, list)
        else []
    )


# ============================================================
# CARGA DE PARTIDOS
# ============================================================

def load_finished_matches() -> list[
    dict[str, Any]
]:

    rows: list[
        dict[str, Any]
    ] = []

    offset = 0

    while True:

        page = supabase_get(
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
                "status": (
                    "in.(FT,AET,PEN)"
                ),
                "order": (
                    "starting_at.asc,id.asc"
                ),
                "limit": str(
                    PAGE_SIZE
                ),
                "offset": str(
                    offset
                ),
            },
        )

        if not page:
            break

        rows.extend(page)

        if len(page) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return rows


def normalize_matches(
    rows: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:

    seen: set[int] = set()

    cleaned: list[
        dict[str, Any]
    ] = []

    for row in rows:

        try:

            item = clean_match(
                row
            )

        except Exception:

            continue

        if item is None:
            continue

        try:

            match_id = int(
                item["id"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            continue

        if match_id in seen:
            continue

        seen.add(match_id)

        cleaned.append(item)

    cleaned.sort(
        key=lambda item: (
            str(
                item["starting_at"]
            ),
            int(
                item["id"]
            ),
        )
    )

    return cleaned


# ============================================================
# DATASET FINGERPRINT
# ============================================================

def dataset_fingerprint(
    match_ids: list[int],
) -> str:

    raw = ":".join(
        str(value)
        for value in match_ids
    ).encode("utf-8")

    return hashlib.sha256(
        raw
    ).hexdigest()[:32]


# ============================================================
# MÉTRICAS
# ============================================================

def brier_multiclass(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    one_hot = np.zeros_like(
        probabilities
    )

    for idx, target in enumerate(
        y_true
    ):

        target_int = int(
            target
        )

        if (
            target_int < 0
            or target_int >= probabilities.shape[1]
        ):

            raise ValueError(
                "Clase objetivo fuera de rango "
                "en Brier."
            )

        one_hot[
            idx,
            target_int
        ] = 1.0

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - one_hot
                ) ** 2,
                axis=1,
            )
        )
    )


# ============================================================
# SOFTMAX Y CALIBRACIÓN
# ============================================================

def softmax(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:

    logits = np.asarray(
        logits,
        dtype=float,
    )

    if logits.ndim != 2:
        raise ValueError(
            "Los logits deben ser una matriz 2D."
        )

    temperature = max(
        float(temperature),
        0.05,
    )

    z = logits / temperature

    z = (
        z
        - np.max(
            z,
            axis=1,
            keepdims=True,
        )
    )

    exp_z = np.exp(z)

    denominator = np.sum(
        exp_z,
        axis=1,
        keepdims=True,
    )

    return (
        exp_z
        / np.clip(
            denominator,
            1e-12,
            None,
        )
    )


def fit_temperature(
    logits: np.ndarray,
    y: np.ndarray,
) -> float:

    logits = np.asarray(
        logits,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    best_temperature = 1.0

    best_loss = math.inf

    for candidate in (
        CALIBRATION_GRID
    ):

        probabilities = softmax(
            logits,
            float(candidate),
        )

        try:

            current_loss = float(
                log_loss(
                    y,
                    probabilities,
                    labels=[
                        0,
                        1,
                        2,
                    ],
                )
            )

        except Exception:

            continue

        if (
            current_loss
            < best_loss - 1e-12
        ):

            best_loss = (
                current_loss
            )

            best_temperature = (
                float(candidate)
            )

    return best_temperature


# ============================================================
# FEATURES
# ============================================================

def feature_defaults(
    X_train: np.ndarray,
) -> dict[str, float]:

    X_train = np.asarray(
        X_train,
        dtype=float,
    )

    medians = np.median(
        X_train,
        axis=0,
    )

    return {
        name: float(value)
        for name, value in zip(
            FEATURE_NAMES,
            medians,
        )
    }


# ============================================================
# SERIALIZACIÓN
# ============================================================

def serialize_artifact(
    model: LogisticRegression,
    scaler: StandardScaler,
    temperature: float,
    validation_metrics: dict[str, Any],
    train_size: int,
    calibration_size: int,
    holdout_size: int,
    defaults: dict[str, float],
) -> dict[str, Any]:

    return {

        "format_version": (
            "NESTOR-MODEL-v1.3"
        ),

        "model_type": (
            "logistic_regression"
        ),

        "model_family": (
            MODEL_FAMILY
        ),

        "model_protocol_version": (
            MODEL_PROTOCOL_VERSION
        ),

        "feature_schema_version": (
            FEATURE_SCHEMA_VERSION
        ),

        "features": list(
            FEATURE_NAMES
        ),

        "classes": [
            0,
            1,
            2,
        ],

        "coefficients": [
            [
                float(value)
                for value in row
            ]
            for row in model.coef_
        ],

        "intercept": [
            float(value)
            for value in model.intercept_
        ],

        "scaler_mean": [
            float(value)
            for value in scaler.mean_
        ],

        "scaler_scale": [
            float(value)
            for value in scaler.scale_
        ],

        "calibration": {

            "method": (
                "temperature_scaling"
            ),

            "temperature": float(
                temperature
            ),
        },

        "feature_defaults": (
            defaults
        ),

        "validation_metrics": (
            validation_metrics
        ),

        "training_examples": int(
            train_size
        ),

        "calibration_examples": int(
            calibration_size
        ),

        "holdout_examples": int(
            holdout_size
        ),
    }


# ============================================================
# MODELO ACTIVO
# ============================================================

def load_active_model() -> (
    dict[str, Any] | None
):

    rows = supabase_get(
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
                "validation_start,"
                "validation_end,"
                "feature_schema_version,"
                "training_config"
            ),

            "active": "eq.true",

            "order": (
                "trained_at.desc"
            ),

            "limit": "1",
        },
    )

    if not rows:
        return None

    return rows[0]


# ============================================================
# CONVERSIÓN DEL MODELO LEGACY
# ============================================================

def convert_legacy_artifact(
    artifact: dict[str, Any] | None,
) -> tuple[
    dict[str, Any] | None,
    str,
]:

    if not isinstance(
        artifact,
        dict,
    ):

        return (
            None,
            "artifact_missing",
        )

    model_type = artifact.get(
        "model_type"
    )

    if model_type != (
        "multinomial_logistic_regression"
    ):

        return (
            None,
            "unsupported_legacy_model_type",
        )

    classes = artifact.get(
        "classes"
    )

    if classes != [
        "A",
        "D",
        "H",
    ]:

        return (
            None,
            "legacy_class_order_incompatible",
        )

    feature_names = artifact.get(
        "feature_names"
    )

    if feature_names != list(
        FEATURE_NAMES
    ):

        return (
            None,
            "legacy_features_incompatible",
        )

    try:

        coefficients = np.asarray(
            artifact[
                "coefficients"
            ],
            dtype=float,
        )

        intercept = np.asarray(
            artifact[
                "intercept"
            ],
            dtype=float,
        )

        scaler_mean = np.asarray(
            artifact[
                "scaler_mean"
            ],
            dtype=float,
        )

        scaler_scale = np.asarray(
            artifact[
                "scaler_scale"
            ],
            dtype=float,
        )

    except Exception:

        return (
            None,
            "legacy_parameters_invalid",
        )

    expected_features = len(
        FEATURE_NAMES
    )

    if coefficients.shape != (
        3,
        expected_features,
    ):

        return (
            None,
            "legacy_coefficient_shape_invalid",
        )

    if intercept.shape != (3,):

        return (
            None,
            "legacy_intercept_shape_invalid",
        )

    if scaler_mean.shape != (
        expected_features,
    ):

        return (
            None,
            "legacy_scaler_mean_invalid",
        )

    if scaler_scale.shape != (
        expected_features,
    ):

        return (
            None,
            "legacy_scaler_scale_invalid",
        )

    # ========================================================
    # ORDEN LEGACY
    #
    # Legacy:
    #   [A, D, H]
    #
    # NESTOR:
    #   [HOME, DRAW, AWAY]
    #   [0,    1,    2]
    #
    # Por tanto:
    #
    #   A -> 2
    #   D -> 1
    #   H -> 0
    #
    # Filas legacy:
    #   0=A
    #   1=D
    #   2=H
    #
    # Filas NESTOR:
    #   H -> legacy 2
    #   D -> legacy 1
    #   A -> legacy 0
    # ========================================================

    legacy_to_nestor = [
        2,
        1,
        0,
    ]

    converted_coefficients = (
        coefficients[
            legacy_to_nestor
        ]
    )

    converted_intercept = (
        intercept[
            legacy_to_nestor
        ]
    )

    converted = {

        "format_version": (
            "NESTOR-LEGACY-BRIDGE-v1"
        ),

        "model_type": (
            "logistic_regression"
        ),

        "model_family": (
            MODEL_FAMILY
        ),

        "model_protocol_version": (
            MODEL_PROTOCOL_VERSION
        ),

        "feature_schema_version": (
            FEATURE_SCHEMA_VERSION
        ),

        "features": list(
            FEATURE_NAMES
        ),

        "classes": [
            0,
            1,
            2,
        ],

        "coefficients": (
            converted_coefficients.tolist()
        ),

        "intercept": (
            converted_intercept.tolist()
        ),

        "scaler_mean": (
            scaler_mean.tolist()
        ),

        "scaler_scale": (
            scaler_scale.tolist()
        ),

        "calibration": {

            "method": (
                "temperature_scaling"
            ),

            "temperature": 1.0,
        },

        "feature_defaults": {},

        "legacy_source_model_type": (
            model_type
        ),

        "legacy_source_classes": (
            classes
        ),

        "legacy_class_conversion": (
            "[A,D,H] -> [H,D,A] -> [0,1,2]"
        ),
    }

    return (
        converted,
        "legacy_multinomial_bridge",
    )


# ============================================================
# NORMALIZACIÓN DEL ACTIVO
# ============================================================

def normalize_active_artifact(
    row: dict[str, Any],
) -> tuple[
    dict[str, Any] | None,
    str,
]:

    artifact = row.get(
        "artifact"
    )

    if isinstance(
        artifact,
        dict,
    ):

        model_type = artifact.get(
            "model_type"
        )

        features = artifact.get(
            "features"
        )

        if (
            model_type
            == "logistic_regression"
            and features
            == list(FEATURE_NAMES)
        ):

            coefficients = artifact.get(
                "coefficients"
            )

            intercept = artifact.get(
                "intercept"
            )

            scaler_mean = artifact.get(
                "scaler_mean"
            )

            scaler_scale = artifact.get(
                "scaler_scale"
            )

            if all(
                isinstance(
                    value,
                    list,
                )
                for value in (
                    coefficients,
                    intercept,
                    scaler_mean,
                    scaler_scale,
                )
            ):

                return (
                    artifact,
                    "native_artifact",
                )

    metrics = row.get(
        "metrics"
    )

    if not isinstance(
        metrics,
        dict,
    ):

        return (
            None,
            "metrics_missing",
        )

    legacy_artifact, reason = (
        convert_legacy_artifact(
            metrics
        )
    )

    if legacy_artifact is not None:

        return (
            legacy_artifact,
            reason,
        )

    return (
        None,
        reason,
    )


# ============================================================
# PROBABILIDADES DESDE ARTIFACT
# ============================================================

def probabilities_from_artifact(
    artifact: dict[str, Any],
    X: np.ndarray,
    temperature_override: float | None = None,
) -> np.ndarray:

    coefficients = np.asarray(
        artifact[
            "coefficients"
        ],
        dtype=float,
    )

    intercept = np.asarray(
        artifact[
            "intercept"
        ],
        dtype=float,
    )

    mean = np.asarray(
        artifact[
            "scaler_mean"
        ],
        dtype=float,
    )

    scale = np.asarray(
        artifact[
            "scaler_scale"
        ],
        dtype=float,
    )

    X = np.asarray(
        X,
        dtype=float,
    )

    if X.ndim != 2:

        raise ValueError(
            "X debe ser una matriz 2D."
        )

    if mean.shape != (
        X.shape[1],
    ):

        raise ValueError(
            "El número de features del "
            "modelo activo no coincide "
            "con X."
        )

    safe_scale = np.where(
        np.abs(scale) < 1e-12,
        1.0,
        scale,
    )

    X_scaled = (
        X - mean
    ) / safe_scale

    logits = (
        X_scaled
        @ coefficients.T
        + intercept
    )

    if (
        temperature_override
        is None
    ):

        calibration = (
            artifact.get(
                "calibration",
                {},
            )
        )

        if not isinstance(
            calibration,
            dict,
        ):

            calibration = {}

        temperature = float(
            calibration.get(
                "temperature",
                1.0,
            )
        )

    else:

        temperature = float(
            temperature_override
        )

    return softmax(
        logits,
        temperature,
    )


def logits_from_artifact(
    artifact: dict[str, Any],
    X: np.ndarray,
) -> np.ndarray:

    coefficients = np.asarray(
        artifact[
            "coefficients"
        ],
        dtype=float,
    )

    intercept = np.asarray(
        artifact[
            "intercept"
        ],
        dtype=float,
    )

    mean = np.asarray(
        artifact[
            "scaler_mean"
        ],
        dtype=float,
    )

    scale = np.asarray(
        artifact[
            "scaler_scale"
        ],
        dtype=float,
    )

    X = np.asarray(
        X,
        dtype=float,
    )

    safe_scale = np.where(
        np.abs(scale) < 1e-12,
        1.0,
        scale,
    )

    X_scaled = (
        X - mean
    ) / safe_scale

    return (
        X_scaled
        @ coefficients.T
        + intercept
    )


# ============================================================
# EVALUACIÓN DEL CAMPEÓN
# ============================================================

def evaluate_active_model(
    active_row: dict[str, Any] | None,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    holdout_ids: list[int],
) -> dict[str, Any] | None:

    if not active_row:

        return None

    artifact, compatibility = (
        normalize_active_artifact(
            active_row
        )
    )

    if artifact is None:

        print(
            "Modelo activo no compatible: "
            f"{compatibility}"
        )

        return None

    try:

        # ----------------------------------------------------
        # RAW DEL CAMPEÓN
        # ----------------------------------------------------

        calibration_logits = (
            logits_from_artifact(
                artifact,
                X_calibration,
            )
        )

        holdout_logits = (
            logits_from_artifact(
                artifact,
                X_holdout,
            )
        )

        raw_calibration = softmax(
            calibration_logits,
            1.0,
        )

        raw_holdout = softmax(
            holdout_logits,
            1.0,
        )

        # ----------------------------------------------------
        # CALIBRACIÓN DEL CAMPEÓN
        #
        # MUY IMPORTANTE:
        # se usa exactamente el mismo conjunto
        # de calibración que utiliza el challenger.
        # ----------------------------------------------------

        champion_temperature = (
            fit_temperature(
                calibration_logits,
                y_calibration,
            )
        )

        calibrated_holdout = softmax(
            holdout_logits,
            champion_temperature,
        )

        # ----------------------------------------------------
        # MÉTRICAS RAW
        # ----------------------------------------------------

        raw_predictions = np.argmax(
            raw_holdout,
            axis=1,
        )

        raw_accuracy = float(
            accuracy_score(
                y_holdout,
                raw_predictions,
            )
        )

        raw_logloss = float(
            log_loss(
                y_holdout,
                raw_holdout,
                labels=[
                    0,
                    1,
                    2,
                ],
            )
        )

        raw_brier = brier_multiclass(
            y_holdout,
            raw_holdout,
        )

        # ----------------------------------------------------
        # MÉTRICAS CALIBRADAS
        # ----------------------------------------------------

        calibrated_predictions = (
            np.argmax(
                calibrated_holdout,
                axis=1,
            )
        )

        calibrated_accuracy = float(
            accuracy_score(
                y_holdout,
                calibrated_predictions,
            )
        )

        calibrated_logloss = float(
            log_loss(
                y_holdout,
                calibrated_holdout,
                labels=[
                    0,
                    1,
                    2,
                ],
            )
        )

        calibrated_brier = (
            brier_multiclass(
                y_holdout,
                calibrated_holdout,
            )
        )

        fingerprint = (
            dataset_fingerprint(
                holdout_ids
            )
        )

        return {

            "raw_accuracy": (
                raw_accuracy
            ),

            "raw_log_loss": (
                raw_logloss
            ),

            "raw_brier": (
                raw_brier
            ),

            "calibrated_accuracy": (
                calibrated_accuracy
            ),

            "calibrated_log_loss": (
                calibrated_logloss
            ),

            "calibrated_brier": (
                calibrated_brier
            ),

            "calibration_temperature": (
                float(
                    champion_temperature
                )
            ),

            "accuracy": (
                calibrated_accuracy
            ),

            "log_loss": (
                calibrated_logloss
            ),

            "brier": (
                calibrated_brier
            ),

            "evaluation_dataset_fingerprint": (
                fingerprint
            ),

            "evaluation_protocol_version": (
                MODEL_PROTOCOL_VERSION
            ),

            "evaluation_compatibility": (
                compatibility
            ),

            "evaluation_holdout_matches": (
                int(
                    len(y_holdout)
                )
            ),
        }

    except Exception as exc:

        print(
            "No se pudo reevaluar "
            "el modelo activo: "
            f"{exc}"
        )

        return None


# ============================================================
# GUARDADO Y PROMOCIÓN
# ============================================================

def save_candidate(
    model: LogisticRegression,
    scaler: StandardScaler,
    temperature: float,
    candidate_metrics: dict[str, Any],
    active_current: dict[str, Any] | None,
    active_row: dict[str, Any] | None,
    holdout_ids: list[int],
    validation_start: str,
    validation_end: str,
    defaults: dict[str, float],
) -> dict[str, Any]:

    now = datetime.now(
        timezone.utc
    )

    version = (
        "NESTOR-1X2-"
        + now.strftime(
            "%Y%m%d-%H%M%S"
        )
    )

    fingerprint = (
        dataset_fingerprint(
            holdout_ids
        )
    )

    candidate_raw = float(
        candidate_metrics[
            "raw_holdout_log_loss"
        ]
    )

    candidate_calibrated = float(
        candidate_metrics[
            "holdout_log_loss"
        ]
    )

    raw_improvement = None

    calibrated_improvement = None

    should_activate = False

    reason = (
        "challenger_protocol_not_comparable"
    )

    # ========================================================
    # PRIMER MODELO
    # ========================================================

    if active_row is None:

        should_activate = True

        reason = (
            "first_nestor_model"
        )

    # ========================================================
    # COMPARACIÓN CONTRA CAMPEÓN
    # ========================================================

    elif active_current is None:

        should_activate = False

        reason = (
            "active_model_not_comparable"
        )

    else:

        active_fingerprint = (
            active_current.get(
                "evaluation_dataset_fingerprint"
            )
        )

        if (
            active_fingerprint
            != fingerprint
        ):

            should_activate = False

            reason = (
                "evaluation_fingerprint_mismatch"
            )

        else:

            raw_improvement = (
                float(
                    active_current[
                        "raw_log_loss"
                    ]
                )
                - candidate_raw
            )

            calibrated_improvement = (
                float(
                    active_current[
                        "calibrated_log_loss"
                    ]
                )
                - candidate_calibrated
            )

            raw_pass = (
                raw_improvement
                >= ACTIVATION_MARGIN
            )

            calibrated_pass = (
                calibrated_improvement
                >= ACTIVATION_MARGIN
            )

            # =================================================
            # PROMOCIÓN:
            #
            # Debe mejorar BOTH:
            # RAW
            # CALIBRATED
            #
            # mínimo 0.001 en ambos.
            # =================================================

            if (
                raw_pass
                and calibrated_pass
            ):

                should_activate = True

                reason = (
                    "better_than_active_on_same_holdout_raw_and_calibrated"
                )

            elif (
                not raw_pass
                and not calibrated_pass
            ):

                should_activate = False

                reason = (
                    "not_better_than_active_on_same_holdout_raw_or_calibrated"
                )

            elif not raw_pass:

                should_activate = False

                reason = (
                    "raw_not_better_than_active"
                )

            else:

                should_activate = False

                reason = (
                    "calibrated_not_better_than_active"
                )

    # ========================================================
    # MÉTRICAS PÚBLICAS
    # ========================================================

    candidate_public = dict(
        candidate_metrics
    )

    candidate_public.pop(
        "_X_holdout",
        None,
    )

    candidate_public.pop(
        "_y_holdout",
        None,
    )

    candidate_public[
        "evaluation_dataset_fingerprint"
    ] = fingerprint

    candidate_public[
        "evaluation_protocol_version"
    ] = MODEL_PROTOCOL_VERSION

    candidate_public[
        "activation_margin"
    ] = float(
        ACTIVATION_MARGIN
    )

    candidate_public[
        "promotion_rule"
    ] = (
        "raw_and_calibrated_log_loss"
    )

    candidate_public[
        "raw_vs_active_log_loss_improvement"
    ] = (
        None
        if raw_improvement is None
        else float(raw_improvement)
    )

    candidate_public[
        "calibrated_vs_active_log_loss_improvement"
    ] = (
        None
        if calibrated_improvement is None
        else float(
            calibrated_improvement
        )
    )

    # ========================================================
    # MÉTRICAS PARA ARTIFACT
    # ========================================================

    validation_metrics = {

        "raw_holdout_log_loss": (
            candidate_public[
                "raw_holdout_log_loss"
            ]
        ),

        "holdout_log_loss": (
            candidate_public[
                "holdout_log_loss"
            ]
        ),

        "holdout_accuracy": (
            candidate_public[
                "holdout_accuracy"
            ]
        ),

        "holdout_brier": (
            candidate_public[
                "holdout_brier"
            ]
        ),

        "calibration_temperature": (
            candidate_public[
                "calibration_temperature"
            ]
        ),

        "raw_vs_active_log_loss_improvement": (
            candidate_public[
                "raw_vs_active_log_loss_improvement"
            ]
        ),

        "calibrated_vs_active_log_loss_improvement": (
            candidate_public[
                "calibrated_vs_active_log_loss_improvement"
            ]
        ),
    }

    artifact = serialize_artifact(
        model=model,
        scaler=scaler,
        temperature=temperature,
        validation_metrics=validation_metrics,
        train_size=int(
            candidate_public[
                "training_matches"
            ]
        ),
        calibration_size=int(
            candidate_public[
                "calibration_matches"
            ]
        ),
        holdout_size=int(
            candidate_public[
                "holdout_matches"
            ]
        ),
        defaults=defaults,
    )

    artifact[
        "evaluation_dataset_fingerprint"
    ] = fingerprint

    artifact[
        "promotion_rule"
    ] = (
        "raw_and_calibrated_log_loss"
    )

    artifact[
        "activation_margin"
    ] = float(
        ACTIVATION_MARGIN
    )

    # ========================================================
    # ESTADO INICIAL
    # ========================================================

    if should_activate:

        initial_status = (
            "promotion_pending"
        )

    elif active_current is None:

        initial_status = (
            "challenger"
        )

    else:

        initial_status = (
            "rejected"
        )

    parent_version = (
        active_row.get(
            "version"
        )
        if active_row
        else None
    )

    row = {

        "version": version,

        "model_name": MODEL_NAME,

        "trained_at": now.isoformat(),

        "training_matches": int(
            candidate_public[
                "training_matches"
            ]
        ),

        "metrics": {

            **candidate_public,

            "active_current_holdout": (
                active_current
            ),

            "activation_reason": (
                reason
            ),

            "parent_version": (
                parent_version
            ),

            "protocol_version": (
                MODEL_PROTOCOL_VERSION
            ),

            "evaluation_dataset_fingerprint": (
                fingerprint
            ),

            "feature_schema_version": (
                FEATURE_SCHEMA_VERSION
            ),
        },

        "artifact": artifact,

        "active": False,

        "status": initial_status,

        "parent_version": (
            parent_version
        ),

        "validation_start": (
            validation_start
        ),

        "validation_end": (
            validation_end
        ),

        "feature_schema_version": (
            FEATURE_SCHEMA_VERSION
        ),

        "training_config": {

            "window": WINDOW,

            "model": MODEL_NAME,

            "solver": "lbfgs",

            "C": 1.0,

            "max_iter": 3000,

            "random_state": 42,

            "evaluation_protocol_version": (
                MODEL_PROTOCOL_VERSION
            ),

            "train_ratio": (
                TRAIN_RATIO
            ),

            "calibration_ratio": (
                CALIBRATION_RATIO
            ),

            "holdout_ratio": (
                1.0
                - TRAIN_RATIO
                - CALIBRATION_RATIO
            ),

            "calibration_method": (
                "temperature_scaling"
            ),

            "calibration_grid_start": 0.50,

            "calibration_grid_end": 3.00,

            "calibration_grid_steps": 101,

            "activation_margin_log_loss": (
                ACTIVATION_MARGIN
            ),

            "promotion_rule": (
                "raw_and_calibrated_log_loss"
            ),

            "evaluation_dataset_fingerprint": (
                fingerprint
            ),
        },
    }

    # ========================================================
    # GUARDADO
    # ========================================================

    inserted = supabase_post(
        "model_versions",
        row,
    )

    if not inserted:

        raise RuntimeError(
            "El challenger no pudo "
            "guardarse en model_versions."
        )

    # ========================================================
    # PROMOCIÓN AUTOMÁTICA
    # ========================================================

    if should_activate:

        # Primero retiramos cualquier campeón.
        supabase_patch(
            "model_versions",
            {
                "active": "eq.true",
            },
            {
                "active": False,
                "status": "retired",
            },
        )

        # Luego activamos exclusivamente
        # el challenger que pasó las dos pruebas.
        activated = supabase_patch(
            "model_versions",
            {
                "version": (
                    f"eq.{version}"
                ),
            },
            {
                "active": True,
                "status": "active",
            },
        )

        if not activated:

            raise RuntimeError(
                "El challenger fue guardado "
                "pero no pudo activarse."
            )

    else:

        supabase_patch(
            "model_versions",
            {
                "version": (
                    f"eq.{version}"
                ),
            },
            {
                "active": False,
                "status": (
                    "rejected"
                    if active_current
                    else "challenger"
                ),
            },
        )

    final_status = (
        "active"
        if should_activate
        else (
            "rejected"
            if active_current
            else "challenger"
        )
    )

    # ========================================================
    # INFORME
    # ========================================================

    print("")
    print(
        "============================================================"
    )

    print(
        "COMPARACIÓN NESTOR-EVAL-v1.3"
    )

    print(
        "============================================================"
    )

    if active_current is None:

        print("")
        print(
            "CAMPEÓN: NO COMPARABLE"
        )

        if active_row:

            print(
                "Motivo: el modelo activo "
                "no pudo evaluarse bajo "
                "el protocolo actual."
            )

    else:

        print("")
        print(
            "CAMPEÓN RAW"
        )

        print(
            "  Accuracy: "
            f"{active_current['raw_accuracy']:.6f}"
        )

        print(
            "  Log loss: "
            f"{active_current['raw_log_loss']:.6f}"
        )

        print(
            "  Brier: "
            f"{active_current['raw_brier']:.6f}"
        )

        print("")
        print(
            "CAMPEÓN CALIBRADO"
        )

        print(
            "  Temperatura: "
            f"{active_current['calibration_temperature']:.2f}"
        )

        print(
            "  Accuracy: "
            f"{active_current['calibrated_accuracy']:.6f}"
        )

        print(
            "  Log loss: "
            f"{active_current['calibrated_log_loss']:.6f}"
        )

        print(
            "  Brier: "
            f"{active_current['calibrated_brier']:.6f}"
        )

    print("")
    print(
        "CHALLENGER RAW"
    )

    print(
        "  Accuracy: "
        f"{candidate_public['raw_holdout_accuracy']:.6f}"
    )

    print(
        "  Log loss: "
        f"{candidate_public['raw_holdout_log_loss']:.6f}"
    )

    print(
        "  Brier: "
        f"{candidate_public['raw_holdout_brier']:.6f}"
    )

    print("")
    print(
        "CHALLENGER CALIBRADO"
    )

    print(
        "  Temperatura: "
        f"{candidate_public['calibration_temperature']:.2f}"
    )

    print(
        "  Accuracy: "
        f"{candidate_public['holdout_accuracy']:.6f}"
    )

    print(
        "  Log loss: "
        f"{candidate_public['holdout_log_loss']:.6f}"
    )

    print(
        "  Brier: "
        f"{candidate_public['holdout_brier']:.6f}"
    )

    print("")

    if raw_improvement is not None:

        print(
            "MEJORA RAW: "
            f"{raw_improvement:.6f}"
        )

        print(
            "MEJORA CALIBRADA: "
            f"{calibrated_improvement:.6f}"
        )

        print(
            "MARGEN REQUERIDO: "
            f"{ACTIVATION_MARGIN:.6f}"
        )

        print("")

        print(
            "RAW PASS: "
            f"{raw_improvement >= ACTIVATION_MARGIN}"
        )

        print(
            "CALIBRATED PASS: "
            f"{calibrated_improvement >= ACTIVATION_MARGIN}"
        )

    print("")

    print(
        "ACTIVAR: "
        f"{should_activate}"
    )

    print(
        "RAZÓN: "
        f"{reason}"
    )

    print("")

    print(
        "VERSION: "
        f"{version}"
    )

    print(
        "============================================================"
    )

    return {

        "version": version,

        "active": bool(
            should_activate
        ),

        "status": final_status,

        "activation_reason": reason,

        "candidate_holdout": (
            candidate_public
        ),

        "active_current_holdout": (
            active_current
        ),

        "parent_version": (
            parent_version
        ),

        "evaluation_dataset_fingerprint": (
            fingerprint
        ),

        "evaluation_protocol_version": (
            MODEL_PROTOCOL_VERSION
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "============================================================"
    )

    print(
        "NESTOR — MOTOR DE INTELIGENCIA "
        "FUTBOLÍSTICA EVOLUTIVA"
    )

    print(
        "Entrenamiento formal 1X2 v1.3"
    )

    print(
        "Protocolo:",
        MODEL_PROTOCOL_VERSION,
    )

    print(
        "Feature schema:",
        FEATURE_SCHEMA_VERSION,
    )

    print(
        "============================================================"
    )

    # ========================================================
    # CARGAR PARTIDOS
    # ========================================================

    raw_rows = (
        load_finished_matches()
    )

    matches = (
        normalize_matches(
            raw_rows
        )
    )

    print(
        "Partidos terminados cargados: "
        f"{len(matches)}"
    )

    if len(matches) < MIN_MATCHES:

        raise RuntimeError(
            f"Se necesitan al menos "
            f"{MIN_MATCHES} partidos terminados."
        )

    # ========================================================
    # DATASET
    # ========================================================

    X_list, y_list, match_ids = (
        build_training_dataset(
            matches
        )
    )

    if len(X_list) < 60:

        raise RuntimeError(
            "No hay suficientes ejemplos "
            "utilizables después del "
            "historial rodante."
        )

    X = np.asarray(
        X_list,
        dtype=float,
    )

    y = np.asarray(
        y_list,
        dtype=int,
    )

    match_ids = [
        int(value)
        for value in match_ids
    ]

    if (
        len(X)
        != len(y)
        or len(X)
        != len(match_ids)
    ):

        raise RuntimeError(
            "X, y y match_ids no tienen "
            "el mismo número de ejemplos."
        )

    # ========================================================
    # VALIDACIÓN DE CLASES
    # ========================================================

    unique_classes = sorted(
        set(
            int(value)
            for value in y.tolist()
        )
    )

    if unique_classes != [
        0,
        1,
        2,
    ]:

        raise RuntimeError(
            "El dataset no contiene "
            "correctamente las tres clases "
            "0=HOME, 1=DRAW, 2=AWAY."
        )

    # ========================================================
    # SPLIT TEMPORAL 60/20/20
    # ========================================================

    total = len(X)

    train_end = int(
        total
        * TRAIN_RATIO
    )

    calibration_end = int(
        total
        * (
            TRAIN_RATIO
            + CALIBRATION_RATIO
        )
    )

    if (
        train_end < 30
        or calibration_end
        - train_end < 10
        or total
        - calibration_end < 10
    ):

        raise RuntimeError(
            "El dataset no permite "
            "una separación temporal "
            "60/20/20 segura."
        )

    X_train = X[
        :train_end
    ]

    y_train = y[
        :train_end
    ]

    X_calibration = X[
        train_end:
        calibration_end
    ]

    y_calibration = y[
        train_end:
        calibration_end
    ]

    X_holdout = X[
        calibration_end:
    ]

    y_holdout = y[
        calibration_end:
    ]

    holdout_ids = match_ids[
        calibration_end:
    ]

    evaluation_fingerprint = (
        dataset_fingerprint(
            holdout_ids
        )
    )

    print(
        "Split temporal:"
    )

    print(
        f"  Entrenamiento: "
        f"{len(X_train)}"
    )

    print(
        f"  Calibración:   "
        f"{len(X_calibration)}"
    )

    print(
        f"  Holdout:       "
        f"{len(X_holdout)}"
    )

    print(
        "  Fingerprint:   "
        f"{evaluation_fingerprint}"
    )

    # ========================================================
    # ENTRENAMIENTO
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_calibration_scaled = (
        scaler.transform(
            X_calibration
        )
    )

    X_holdout_scaled = (
        scaler.transform(
            X_holdout
        )
    )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        C=1.0,
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    # ========================================================
    # LOGITS DEL CHALLENGER
    # ========================================================

    calibration_logits = (
        X_calibration_scaled
        @ model.coef_.T
        + model.intercept_
    )

    holdout_logits = (
        X_holdout_scaled
        @ model.coef_.T
        + model.intercept_
    )

    # ========================================================
    # CALIBRACIÓN CHALLENGER
    # ========================================================

    temperature = (
        fit_temperature(
            calibration_logits,
            y_calibration,
        )
    )

    # ========================================================
    # CHALLENGER RAW
    # ========================================================

    raw_holdout_probabilities = (
        softmax(
            holdout_logits,
            1.0,
        )
    )

    raw_holdout_predictions = (
        np.argmax(
            raw_holdout_probabilities,
            axis=1,
        )
    )

    raw_accuracy = float(
        accuracy_score(
            y_holdout,
            raw_holdout_predictions,
        )
    )

    raw_logloss = float(
        log_loss(
            y_holdout,
            raw_holdout_probabilities,
            labels=[
                0,
                1,
                2,
            ],
        )
    )

    raw_brier = (
        brier_multiclass(
            y_holdout,
            raw_holdout_probabilities,
        )
    )

    # ========================================================
    # CHALLENGER CALIBRADO
    # ========================================================

    calibrated_holdout_probabilities = (
        softmax(
            holdout_logits,
            temperature,
        )
    )

    calibrated_predictions = (
        np.argmax(
            calibrated_holdout_probabilities,
            axis=1,
        )
    )

    calibrated_accuracy = float(
        accuracy_score(
            y_holdout,
            calibrated_predictions,
        )
    )

    calibrated_logloss = float(
        log_loss(
            y_holdout,
            calibrated_holdout_probabilities,
            labels=[
                0,
                1,
                2,
            ],
        )
    )

    calibrated_brier = (
        brier_multiclass(
            y_holdout,
            calibrated_holdout_probabilities,
        )
    )

    print("")
    print(
        "=== CHALLENGER ==="
    )

    print(
        "Raw accuracy: "
        f"{raw_accuracy:.6f}"
    )

    print(
        "Raw log loss: "
        f"{raw_logloss:.6f}"
    )

    print(
        "Raw Brier: "
        f"{raw_brier:.6f}"
    )

    print(
        "Calibrated accuracy: "
        f"{calibrated_accuracy:.6f}"
    )

    print(
        "Calibrated log loss: "
        f"{calibrated_logloss:.6f}"
    )

    print(
        "Calibrated Brier: "
        f"{calibrated_brier:.6f}"
    )

    print(
        "Calibration temperature: "
        f"{temperature:.2f}"
    )

    # ========================================================
    # DEFAULTS
    # ========================================================

    defaults = (
        feature_defaults(
            X_train
        )
    )

    # ========================================================
    # MÉTRICAS CANDIDATO
    # ========================================================

    candidate_metrics: dict[
        str,
        Any,
    ] = {

        "holdout_accuracy": (
            calibrated_accuracy
        ),

        "holdout_log_loss": (
            calibrated_logloss
        ),

        "holdout_brier": (
            calibrated_brier
        ),

        "raw_holdout_accuracy": (
            raw_accuracy
        ),

        "raw_holdout_log_loss": (
            raw_logloss
        ),

        "raw_holdout_brier": (
            raw_brier
        ),

        "calibrated_holdout_accuracy": (
            calibrated_accuracy
        ),

        "calibrated_holdout_log_loss": (
            calibrated_logloss
        ),

        "calibrated_holdout_brier": (
            calibrated_brier
        ),

        "calibration_temperature": (
            float(temperature)
        ),

        "training_matches": (
            int(len(X_train))
        ),

        "calibration_matches": (
            int(
                len(X_calibration)
            )
        ),

        "holdout_matches": (
            int(
                len(X_holdout)
            )
        ),

        "usable_examples": (
            int(len(X))
        ),

        "total_finished_matches": (
            int(len(matches))
        ),

        "skipped_for_history": (
            int(
                len(matches)
                - len(X)
            )
        ),

        "evaluation_dataset_fingerprint": (
            evaluation_fingerprint
        ),

        "evaluation_protocol_version": (
            MODEL_PROTOCOL_VERSION
        ),

        "_X_holdout": (
            X_holdout.tolist()
        ),

        "_y_holdout": (
            y_holdout.tolist()
        ),
    }

    # ========================================================
    # FECHAS DEL HOLDOUT
    # ========================================================

    match_dates = {

        int(item["id"]): (
            item["starting_at"]
        )

        for item in matches
    }

    if not holdout_ids:

        raise RuntimeError(
            "El holdout está vacío."
        )

    validation_start = (
        match_dates.get(
            int(
                holdout_ids[0]
            )
        )
    )

    validation_end = (
        match_dates.get(
            int(
                holdout_ids[-1]
            )
        )
    )

    if (
        validation_start is None
        or validation_end is None
    ):

        raise RuntimeError(
            "No se pudieron determinar "
            "las fechas del holdout."
        )

    # ========================================================
    # CARGAR CAMPEÓN
    #
    # ESTA ERA LA FUNCIÓN QUE FALTABA
    # EN EL ARCHIVO QUE PRODUJO EL ERROR.
    # ========================================================

    active_row = (
        load_active_model()
    )

    if active_row:

        print("")
        print(
            "Modelo activo detectado: "
            f"{active_row.get('version')}"
        )

    else:

        print("")
        print(
            "No existe modelo activo."
        )

    # ========================================================
    # EVALUAR CAMPEÓN EN EL MISMO HOLDOUT
    # ========================================================

    active_current = (
        evaluate_active_model(
            active_row=active_row,
            X_calibration=X_calibration,
            y_calibration=y_calibration,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            holdout_ids=holdout_ids,
        )
    )

    # ========================================================
    # CREAR ARTIFACT Y GUARDAR CANDIDATO
    # ========================================================

    result = save_candidate(
        model=model,
        scaler=scaler,
        temperature=temperature,
        candidate_metrics=candidate_metrics,
        active_current=active_current,
        active_row=active_row,
        holdout_ids=holdout_ids,
        validation_start=validation_start,
        validation_end=validation_end,
        defaults=defaults,
    )

    # ========================================================
    # RESULTADO FINAL JSON
    # ========================================================

    print("")

    print(
        "============================================================"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "============================================================"
    )

    print(
        "ENTRENAMIENTO NESTOR FINALIZADO"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":

    main()
