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
    WINDOW,
    build_training_dataset,
    clean_match,
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "").strip()

MODEL_NAME = "NESTOR-1X2-LogisticRegression"
MODEL_FAMILY = "1X2"

MODEL_PROTOCOL_VERSION = "NESTOR-EVAL-v1.3"

MIN_MATCHES = 80
PAGE_SIZE = 1000

CALIBRATION_GRID = np.linspace(0.50, 3.00, 101)

ACTIVATION_MARGIN = 0.001

TRAIN_RATIO = 0.60
CALIBRATION_RATIO = 0.20


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


def supabase_get(
    table: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:

    with httpx.Client(timeout=60.0) as client:
        response = client.get(
            supabase_url(table),
            headers=HEADERS,
            params=params,
        )

    if response.status_code not in (200, 206):
        raise RuntimeError(
            f"Supabase GET {table} "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    return data if isinstance(data, list) else []


def supabase_post(
    table: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:

    headers = {
        **HEADERS,
        "Prefer": "return=representation",
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            supabase_url(table),
            headers=headers,
            json=payload,
        )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Supabase POST {table} "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    return response.json() if response.text else []


def supabase_patch(
    table: str,
    params: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:

    headers = {
        **HEADERS,
        "Prefer": "return=representation",
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.patch(
            supabase_url(table),
            headers=headers,
            params=params,
            json=payload,
        )

    if response.status_code not in (200, 204):
        raise RuntimeError(
            f"Supabase PATCH {table} "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    return response.json() if response.text else []


def load_finished_matches() -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []

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
                "status": "in.(FT,AET,PEN)",
                "order": "starting_at.asc,id.asc",
                "limit": str(PAGE_SIZE),
                "offset": str(offset),
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
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    seen: set[int] = set()

    cleaned: list[dict[str, Any]] = []

    for row in rows:

        try:
            item = clean_match(row)
        except Exception:
            continue

        if item is None:
            continue

        try:
            match_id = int(item["id"])
        except Exception:
            continue

        if match_id in seen:
            continue

        item["id"] = match_id
        seen.add(match_id)

        cleaned.append(item)

    cleaned.sort(
        key=lambda item: (
            item["starting_at"],
            item["id"],
        )
    )

    return cleaned


def dataset_fingerprint(
    match_ids: list[int],
) -> str:

    raw = ":".join(
        str(value)
        for value in match_ids
    ).encode("utf-8")

    return hashlib.sha256(
        raw
    ).hexdigest()[:24]


def brier_multiclass(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:

    if probabilities.ndim != 2:
        raise RuntimeError(
            "Las probabilidades no tienen forma matricial."
        )

    if probabilities.shape[1] != 3:
        raise RuntimeError(
            "Se esperaban exactamente 3 clases."
        )

    one_hot = np.zeros_like(
        probabilities,
        dtype=float,
    )

    for idx, target in enumerate(y_true):

        target_int = int(target)

        if target_int not in (0, 1, 2):
            raise RuntimeError(
                f"Clase objetivo inválida: {target_int}"
            )

        one_hot[idx, target_int] = 1.0

    return float(
        np.mean(
            np.sum(
                (probabilities - one_hot) ** 2,
                axis=1,
            )
        )
    )


def softmax(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:

    logits = np.asarray(
        logits,
        dtype=float,
    )

    if logits.ndim != 2:
        raise RuntimeError(
            "Los logits deben ser una matriz 2D."
        )

    t = max(
        float(temperature),
        0.05,
    )

    z = logits / t

    z = z - np.max(
        z,
        axis=1,
        keepdims=True,
    )

    exp_z = np.exp(z)

    denominator = np.sum(
        exp_z,
        axis=1,
        keepdims=True,
    )

    return exp_z / np.clip(
        denominator,
        1e-12,
        None,
    )


def fit_temperature(
    logits: np.ndarray,
    y: np.ndarray,
) -> float:

    best_t = 1.0
    best_loss = math.inf

    logits = np.asarray(
        logits,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    for candidate in CALIBRATION_GRID:

        probabilities = softmax(
            logits,
            float(candidate),
        )

        loss = float(
            log_loss(
                y,
                probabilities,
                labels=[0, 1, 2],
            )
        )

        if loss < best_loss - 1e-12:

            best_loss = loss

            best_t = float(candidate)

    return best_t


def feature_defaults(
    X_train: np.ndarray,
) -> dict[str, float]:

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


def serialize_artifact(
    model: LogisticRegression,
    scaler: StandardScaler,
    temperature: float,
    accuracy: float,
    logloss_value: float,
    train_size: int,
    calibration_size: int,
    holdout_size: int,
    defaults: dict[str, float],
) -> dict[str, Any]:

    if model.classes_.tolist() != [0, 1, 2]:
        raise RuntimeError(
            "El modelo entrenado no tiene "
            "el orden de clases esperado [0, 1, 2]."
        )

    return {
        "format_version": "NESTOR-MODEL-v1.3",

        "model_type": "logistic_regression",

        "model_family": MODEL_FAMILY,

        "model_protocol_version": (
            MODEL_PROTOCOL_VERSION
        ),

        "feature_schema_version": (
            FEATURE_SCHEMA_VERSION
        ),

        "features": FEATURE_NAMES,

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
            "method": "temperature_scaling",
            "temperature": float(
                temperature
            ),
        },

        "feature_defaults": defaults,

        "validation_accuracy": float(
            accuracy
        ),

        "validation_log_loss": float(
            logloss_value
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


def convert_legacy_artifact(
    artifact: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str]:

    if not isinstance(
        artifact,
        dict,
    ):
        return None, "artifact_missing"

    model_type = artifact.get(
        "model_type"
    )

    if model_type != "multinomial_logistic_regression":
        return None, "unsupported_legacy_model_type"

    classes = artifact.get(
        "classes"
    )

    if classes != [
        "A",
        "D",
        "H",
    ]:
        return None, "legacy_class_order_incompatible"

    feature_names = artifact.get(
        "feature_names"
    )

    if feature_names != FEATURE_NAMES:
        return None, "legacy_features_incompatible"

    try:

        coefficients = np.asarray(
            artifact["coefficients"],
            dtype=float,
        )

        intercept = np.asarray(
            artifact["intercept"],
            dtype=float,
        )

        scaler_mean = np.asarray(
            artifact["scaler_mean"],
            dtype=float,
        )

        scaler_scale = np.asarray(
            artifact["scaler_scale"],
            dtype=float,
        )

    except Exception:

        return None, "legacy_parameters_invalid"

    expected_features = len(
        FEATURE_NAMES
    )

    if coefficients.shape != (
        3,
        expected_features,
    ):
        return None, "legacy_coefficient_shape_invalid"

    if intercept.shape != (3,):
        return None, "legacy_intercept_shape_invalid"

    if scaler_mean.shape != (
        expected_features,
    ):
        return None, "legacy_scaler_mean_invalid"

    if scaler_scale.shape != (
        expected_features,
    ):
        return None, "legacy_scaler_scale_invalid"

    if not np.all(
        np.isfinite(coefficients)
    ):
        return None, "legacy_coefficients_nonfinite"

    if not np.all(
        np.isfinite(intercept)
    ):
        return None, "legacy_intercept_nonfinite"

    if not np.all(
        np.isfinite(scaler_mean)
    ):
        return None, "legacy_scaler_mean_nonfinite"

    if not np.all(
        np.isfinite(scaler_scale)
    ):
        return None, "legacy_scaler_scale_nonfinite"

    if np.any(
        np.abs(scaler_scale) < 1e-12
    ):
        return None, "legacy_scaler_scale_zero"

    # ------------------------------------------------------------
    # LEGACY CLASS ORDER
    #
    # Antiguo:
    #   A = away
    #   D = draw
    #   H = home
    #
    # NESTOR actual:
    #   0 = home
    #   1 = draw
    #   2 = away
    #
    # Por tanto:
    #
    #   [A, D, H]
    #       ↓
    #   [H, D, A]
    #
    # Reordenamos filas de coeficientes e interceptos.
    # ------------------------------------------------------------

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

        "model_family": MODEL_FAMILY,

        "model_protocol_version": (
            MODEL_PROTOCOL_VERSION
        ),

        "feature_schema_version": (
            FEATURE_SCHEMA_VERSION
        ),

        "features": FEATURE_NAMES,

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
            "method": "temperature_scaling",
            "temperature": 1.0,
        },

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


def validate_native_artifact(
    artifact: dict[str, Any],
) -> tuple[bool, str]:

    if artifact.get(
        "model_type"
    ) != "logistic_regression":

        return False, "native_model_type_incompatible"

    if artifact.get(
        "features"
    ) != FEATURE_NAMES:

        return False, "native_features_incompatible"

    if artifact.get(
        "feature_schema_version"
    ) != FEATURE_SCHEMA_VERSION:

        return False, "native_feature_schema_incompatible"

    if artifact.get(
        "classes"
    ) != [0, 1, 2]:

        return False, "native_class_order_incompatible"

    try:

        coefficients = np.asarray(
            artifact["coefficients"],
            dtype=float,
        )

        intercept = np.asarray(
            artifact["intercept"],
            dtype=float,
        )

        scaler_mean = np.asarray(
            artifact["scaler_mean"],
            dtype=float,
        )

        scaler_scale = np.asarray(
            artifact["scaler_scale"],
            dtype=float,
        )

    except Exception:

        return False, "native_parameters_invalid"

    expected_features = len(
        FEATURE_NAMES
    )

    if coefficients.shape != (
        3,
        expected_features,
    ):
        return False, "native_coefficient_shape_invalid"

    if intercept.shape != (3,):
        return False, "native_intercept_shape_invalid"

    if scaler_mean.shape != (
        expected_features,
    ):
        return False, "native_scaler_mean_invalid"

    if scaler_scale.shape != (
        expected_features,
    ):
        return False, "native_scaler_scale_invalid"

    if not np.all(
        np.isfinite(coefficients)
    ):
        return False, "native_coefficients_nonfinite"

    if not np.all(
        np.isfinite(intercept)
    ):
        return False, "native_intercept_nonfinite"

    if not np.all(
        np.isfinite(scaler_mean)
    ):
        return False, "native_scaler_mean_nonfinite"

    if not np.all(
        np.isfinite(scaler_scale)
    ):
        return False, "native_scaler_scale_nonfinite"

    if np.any(
        np.abs(scaler_scale) < 1e-12
    ):
        return False, "native_scaler_scale_zero"

    calibration = artifact.get(
        "calibration",
        {},
    )

    if not isinstance(
        calibration,
        dict,
    ):
        return False, "native_calibration_invalid"

    try:

        temperature = float(
            calibration.get(
                "temperature",
                1.0,
            )
        )

    except Exception:

        return False, "native_calibration_temperature_invalid"

    if not math.isfinite(
        temperature
    ) or temperature <= 0.0:

        return False, "native_calibration_temperature_invalid"

    return True, "strict"


def normalize_active_artifact(
    row: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:

    artifact = row.get(
        "artifact"
    )

    if isinstance(
        artifact,
        dict,
    ):

        valid, reason = (
            validate_native_artifact(
                artifact
            )
        )

        if valid:

            return (
                artifact,
                reason,
            )

    metrics = row.get(
        "metrics"
    )

    if not isinstance(
        metrics,
        dict,
    ):

        return None, "metrics_missing"

    # El campeón antiguo guarda el modelo
    # dentro de metrics.
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

    return None, reason


def logits_from_artifact(
    artifact: dict[str, Any],
    X: np.ndarray,
) -> np.ndarray:

    coefficients = np.asarray(
        artifact["coefficients"],
        dtype=float,
    )

    intercept = np.asarray(
        artifact["intercept"],
        dtype=float,
    )

    mean = np.asarray(
        artifact["scaler_mean"],
        dtype=float,
    )

    scale = np.asarray(
        artifact["scaler_scale"],
        dtype=float,
    )

    expected_features = len(
        FEATURE_NAMES
    )

    if X.ndim != 2:
        raise RuntimeError(
            "X debe ser una matriz 2D."
        )

    if X.shape[1] != expected_features:
        raise RuntimeError(
            "El número de features de X "
            "no coincide con FEATURE_NAMES."
        )

    scale = np.where(
        np.abs(scale) < 1e-12,
        1.0,
        scale,
    )

    Xs = (
        X - mean
    ) / scale

    return (
        Xs @ coefficients.T
        + intercept
    )


def probabilities_from_artifact(
    artifact: dict[str, Any],
    X: np.ndarray,
) -> np.ndarray:

    logits = logits_from_artifact(
        artifact,
        X,
    )

    calibration = artifact.get(
        "calibration",
        {},
    )

    temperature = float(
        calibration.get(
            "temperature",
            1.0,
        )
    )

    return softmax(
        logits,
        temperature,
    )


def evaluate_probabilities(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1, 2],
            )
        ),

        "brier": brier_multiclass(
            y_true,
            probabilities,
        ),
    }


def current_holdout_evaluation(
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
            "Modelo activo no compatible "
            "con el holdout común: "
            f"{compatibility}"
        )

        return None

    try:

        # IMPORTANTÍSIMO:
        # Para comparar campeón y challenger de forma
        # simétrica, la temperatura del campeón también
        # se vuelve a ajustar usando EL MISMO conjunto
        # de calibración actual.
        calibration_logits = (
            logits_from_artifact(
                artifact,
                X_calibration,
            )
        )

        active_temperature = fit_temperature(
            calibration_logits,
            y_calibration,
        )

        holdout_logits = (
            logits_from_artifact(
                artifact,
                X_holdout,
            )
        )

        raw_probabilities = softmax(
            holdout_logits,
            1.0,
        )

        calibrated_probabilities = softmax(
            holdout_logits,
            active_temperature,
        )

        raw_metrics = evaluate_probabilities(
            raw_probabilities,
            y_holdout,
        )

        calibrated_metrics = (
            evaluate_probabilities(
                calibrated_probabilities,
                y_holdout,
            )
        )

        fingerprint = dataset_fingerprint(
            holdout_ids
        )

        return {
            "raw_accuracy": raw_metrics[
                "accuracy"
            ],

            "raw_log_loss": raw_metrics[
                "log_loss"
            ],

            "raw_brier": raw_metrics[
                "brier"
            ],

            "calibrated_accuracy": (
                calibrated_metrics[
                    "accuracy"
                ]
            ),

            "calibrated_log_loss": (
                calibrated_metrics[
                    "log_loss"
                ]
            ),

            "calibrated_brier": (
                calibrated_metrics[
                    "brier"
                ]
            ),

            "calibration_temperature": (
                float(active_temperature)
            ),

            # Compatibilidad histórica:
            # log_loss = calibrado.
            "accuracy": calibrated_metrics[
                "accuracy"
            ],

            "log_loss": calibrated_metrics[
                "log_loss"
            ],

            "brier": calibrated_metrics[
                "brier"
            ],

            "evaluation_dataset_fingerprint": (
                fingerprint
            ),

            "evaluation_protocol_version": (
                MODEL_PROTOCOL_VERSION
            ),

            "evaluation_compatibility": (
                compatibility
            ),

            "evaluation_calibration_matches": (
                int(len(y_calibration))
            ),

            "evaluation_holdout_matches": (
                int(len(y_holdout))
            ),
        }

    except Exception as exc:

        print(
            "No se pudo reevaluar el "
            "activo en el holdout común: "
            f"{exc}"
        )

        return None


def save_candidate(
    artifact: dict[str, Any],
    active_row: dict[str, Any] | None,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    holdout_ids: list[int],
    candidate_metrics: dict[str, Any],
    validation_start: str,
    validation_end: str,
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

    fingerprint = dataset_fingerprint(
        holdout_ids
    )

    active_current = None

    if active_row:

        active_current = (
            current_holdout_evaluation(
                active_row,
                X_calibration,
                y_calibration,
                X_holdout,
                y_holdout,
                holdout_ids,
            )
        )

    candidate_public = dict(
        candidate_metrics
    )

    should_activate = False

    reason = "challenger"

    raw_improvement = None

    calibrated_improvement = None

    if active_row is None:

        should_activate = True

        reason = (
            "first_nestor_model"
        )

    elif active_current is None:

        reason = (
            "challenger_protocol_not_comparable"
        )

    else:

        common_fingerprint = (
            active_current[
                "evaluation_dataset_fingerprint"
            ]
            == fingerprint
        )

        if not common_fingerprint:

            reason = (
                "evaluation_fingerprint_mismatch"
            )

        else:

            raw_improvement = float(
                active_current[
                    "raw_log_loss"
                ]
                - candidate_public[
                    "raw_holdout_log_loss"
                ]
            )

            calibrated_improvement = float(
                active_current[
                    "calibrated_log_loss"
                ]
                - candidate_public[
                    "holdout_log_loss"
                ]
            )

            candidate_public[
                "raw_vs_active_log_loss_improvement"
            ] = raw_improvement

            candidate_public[
                "calibrated_vs_active_log_loss_improvement"
            ] = calibrated_improvement

            raw_pass = (
                raw_improvement
                >= ACTIVATION_MARGIN
            )

            calibrated_pass = (
                calibrated_improvement
                >= ACTIVATION_MARGIN
            )

            if raw_pass and calibrated_pass:

                should_activate = True

                reason = (
                    "improved_active_on_same_holdout_raw_and_calibrated"
                )

            elif not raw_pass and not calibrated_pass:

                reason = (
                    "not_better_than_active_on_same_holdout_raw_or_calibrated"
                )

            elif not raw_pass:

                reason = (
                    "raw_not_better_than_active"
                )

            else:

                reason = (
                    "calibrated_not_better_than_active"
                )

    artifact[
        "evaluation_dataset_fingerprint"
    ] = fingerprint

    artifact[
        "model_protocol_version"
    ] = MODEL_PROTOCOL_VERSION

    initial_status = (
        "promotion_pending"
        if should_activate
        else (
            "rejected"
            if active_current
            else "challenger"
        )
    )

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

            "activation_reason": reason,

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
            active_row.get("version")
            if active_row
            else None
        ),

        "validation_start": validation_start,

        "validation_end": validation_end,

        "feature_schema_version": (
            FEATURE_SCHEMA_VERSION
        ),

        "training_config": {

            "window": WINDOW,

            "model": MODEL_NAME,

            "evaluation_protocol_version": (
                MODEL_PROTOCOL_VERSION
            ),

            "train_ratio": TRAIN_RATIO,

            "calibration_ratio": (
                CALIBRATION_RATIO
            ),

            "holdout_ratio": (
                1.0
                - TRAIN_RATIO
                - CALIBRATION_RATIO
            ),

            "activation_margin_log_loss": (
                ACTIVATION_MARGIN
            ),

            "evaluation_dataset_fingerprint": (
                fingerprint
            ),

            "promotion_rule": (
                "raw_and_calibrated_log_loss"
            ),
        },
    }

    inserted = supabase_post(
        "model_versions",
        row,
    )

    if not inserted:

        raise RuntimeError(
            "El challenger no pudo guardarse "
            "en model_versions."
        )

    if should_activate:

        previous_active_version = (
            active_row.get("version")
            if active_row
            else None
        )

        try:

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

            activated = supabase_patch(
                "model_versions",
                {
                    "version": f"eq.{version}",
                },
                {
                    "active": True,
                    "status": "active",
                },
            )

            if not activated:

                raise RuntimeError(
                    "El challenger fue guardado, "
                    "pero no pudo activarse."
                )

        except Exception as exc:

            # Intento de recuperación para no dejar
            # el sistema sin campeón.
            if previous_active_version:

                try:

                    supabase_patch(
                        "model_versions",
                        {
                            "version": (
                                f"eq.{previous_active_version}"
                            ),
                        },
                        {
                            "active": True,
                            "status": "active",
                        },
                    )

                except Exception as rollback_exc:

                    raise RuntimeError(
                        "Falló la promoción y también "
                        "falló la recuperación del campeón "
                        f"anterior: {rollback_exc}"
                    ) from exc

            raise

    else:

        supabase_patch(
            "model_versions",
            {
                "version": f"eq.{version}",
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

    return {

        "version": version,

        "active": should_activate,

        "status": final_status,

        "activation_reason": reason,

        "candidate_holdout": (
            candidate_public
        ),

        "active_current_holdout": (
            active_current
        ),

        "parent_version": (
            row["parent_version"]
        ),

        "evaluation_dataset_fingerprint": (
            fingerprint
        ),

        "evaluation_protocol_version": (
            MODEL_PROTOCOL_VERSION
        ),
    }


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

    raw_rows = load_finished_matches()

    matches = normalize_matches(
        raw_rows
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

    X_list, y_list, match_ids = (
        build_training_dataset(
            matches
        )
    )

    if len(X_list) < 60:

        raise RuntimeError(
            "No hay suficientes ejemplos "
            "utilizables después del historial rodante."
        )

    X = np.asarray(
        X_list,
        dtype=float,
    )

    y = np.asarray(
        y_list,
        dtype=int,
    )

    if X.ndim != 2:

        raise RuntimeError(
            "El dataset X no tiene forma 2D."
        )

    if X.shape[1] != len(
        FEATURE_NAMES
    ):

        raise RuntimeError(
            "El número de columnas de X "
            "no coincide con FEATURE_NAMES."
        )

    if len(X) != len(y):

        raise RuntimeError(
            "X e y no tienen la misma cantidad "
            "de ejemplos."
        )

    if len(X) != len(match_ids):

        raise RuntimeError(
            "X y match_ids no tienen la misma "
            "cantidad de ejemplos."
        )

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
            "El dataset no contiene correctamente "
            "las tres clases [0,1,2]."
        )

    train_end = int(
        len(X) * TRAIN_RATIO
    )

    calibration_end = int(
        len(X)
        * (
            TRAIN_RATIO
            + CALIBRATION_RATIO
        )
    )

    if (
        train_end < 30
        or calibration_end - train_end < 10
        or len(X) - calibration_end < 10
    ):

        raise RuntimeError(
            "El dataset no permite una "
            "separación temporal 60/20/20 segura."
        )

    X_train = X[
        :train_end
    ]

    y_train = y[
        :train_end
    ]

    train_classes = sorted(
        set(
            int(value)
            for value in y_train.tolist()
        )
    )

    if train_classes != [
        0,
        1,
        2,
    ]:

        raise RuntimeError(
            "El conjunto de entrenamiento no contiene "
            "las tres clases [0,1,2]. No se puede "
            "entrenar LogisticRegression de forma segura."
        )

    X_cal = X[
        train_end:calibration_end
    ]

    y_cal = y[
        train_end:calibration_end
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
        f"  Entrenamiento: {len(X_train)}"
    )

    print(
        f"  Calibración:   {len(X_cal)}"
    )

    print(
        f"  Holdout:       {len(X_holdout)}"
    )

    print(
        "  Fingerprint:   "
        f"{evaluation_fingerprint}"
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_cal_scaled = scaler.transform(
        X_cal
    )

    X_holdout_scaled = scaler.transform(
        X_holdout
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

    if model.classes_.tolist() != [
        0,
        1,
        2,
    ]:

        raise RuntimeError(
            "LogisticRegression produjo un "
            "orden de clases distinto de [0,1,2]."
        )

    cal_logits = (
        X_cal_scaled
        @ model.coef_.T
        + model.intercept_
    )

    temperature = fit_temperature(
        cal_logits,
        y_cal,
    )

    holdout_logits = (
        X_holdout_scaled
        @ model.coef_.T
        + model.intercept_
    )

    raw_holdout_probabilities = softmax(
        holdout_logits,
        1.0,
    )

    calibrated_holdout_probabilities = (
        softmax(
            holdout_logits,
            temperature,
        )
    )

    raw_metrics = evaluate_probabilities(
        raw_holdout_probabilities,
        y_holdout,
    )

    calibrated_metrics = (
        evaluate_probabilities(
            calibrated_holdout_probabilities,
            y_holdout,
        )
    )

    print("")
    print("=== CHALLENGER ===")
    print(
        "Raw accuracy: "
        f"{raw_metrics['accuracy']:.6f}"
    )
    print(
        "Raw log loss: "
        f"{raw_metrics['log_loss']:.6f}"
    )
    print(
        "Raw Brier: "
        f"{raw_metrics['brier']:.6f}"
    )
    print(
        "Calibrated accuracy: "
        f"{calibrated_metrics['accuracy']:.6f}"
    )
    print(
        "Calibrated log loss: "
        f"{calibrated_metrics['log_loss']:.6f}"
    )
    print(
        "Calibrated Brier: "
        f"{calibrated_metrics['brier']:.6f}"
    )
    print(
        "Calibration temperature: "
        f"{temperature:.2f}"
    )

    defaults = feature_defaults(
        X_train
    )

    artifact = serialize_artifact(
        model,
        scaler,
        temperature,
        calibrated_metrics[
            "accuracy"
        ],
        calibrated_metrics[
            "log_loss"
        ],
        len(X_train),
        len(X_cal),
        len(X_holdout),
        defaults,
    )

    active_row = load_active_model()

    if active_row:

        print(
            "Modelo activo detectado: "
            f"{active_row.get('version')}"
        )

    else:

        print(
            "No existe modelo activo."
        )

    candidate_metrics: dict[str, Any] = {

        "holdout_accuracy": calibrated_metrics[
            "accuracy"
        ],

        "holdout_log_loss": calibrated_metrics[
            "log_loss"
        ],

        "holdout_brier": calibrated_metrics[
            "brier"
        ],

        "raw_holdout_accuracy": raw_metrics[
            "accuracy"
        ],

        "raw_holdout_log_loss": raw_metrics[
            "log_loss"
        ],

        "raw_holdout_brier": raw_metrics[
            "brier"
        ],

        "calibrated_holdout_accuracy": (
            calibrated_metrics[
                "accuracy"
            ]
        ),

        "calibrated_holdout_log_loss": (
            calibrated_metrics[
                "log_loss"
            ]
        ),

        "calibrated_holdout_brier": (
            calibrated_metrics[
                "brier"
            ]
        ),

        "calibration_temperature": (
            float(temperature)
        ),

        "training_matches": int(
            len(X_train)
        ),

        "calibration_matches": int(
            len(X_cal)
        ),

        "holdout_matches": int(
            len(X_holdout)
        ),

        "usable_examples": int(
            len(X)
        ),

        "total_finished_matches": int(
            len(matches)
        ),

        "skipped_for_history": int(
            len(matches) - len(X)
        ),

        "evaluation_dataset_fingerprint": (
            evaluation_fingerprint
        ),

        "evaluation_protocol_version": (
            MODEL_PROTOCOL_VERSION
        ),
    }

    match_dates = {
        int(item["id"]): item["starting_at"]
        for item in matches
    }

    if not holdout_ids:

        raise RuntimeError(
            "El holdout quedó vacío."
        )

    validation_start = match_dates.get(
        int(holdout_ids[0]),
        matches[0]["starting_at"],
    )

    validation_end = match_dates.get(
        int(holdout_ids[-1]),
        matches[-1]["starting_at"],
    )

    result = save_candidate(
        artifact,
        active_row,
        X_cal,
        y_cal,
        X_holdout,
        y_holdout,
        holdout_ids,
        candidate_metrics,
        validation_start,
        validation_end,
    )

    print("")
    print(
        "============================================================"
    )

    if result[
        "active_current_holdout"
    ] is not None:

        champion = result[
            "active_current_holdout"
        ]

        candidate = result[
            "candidate_holdout"
        ]

        print(
            "CAMPEON RAW"
        )

        print(
            "  Accuracy: "
            f"{champion['raw_accuracy']:.6f}"
        )

        print(
            "  Log loss: "
            f"{champion['raw_log_loss']:.6f}"
        )

        print(
            "  Brier: "
            f"{champion['raw_brier']:.6f}"
        )

        print("")

        print(
            "CAMPEON CALIBRADO"
        )

        print(
            "  Temperatura: "
            f"{champion['calibration_temperature']:.2f}"
        )

        print(
            "  Accuracy: "
            f"{champion['calibrated_accuracy']:.6f}"
        )

        print(
            "  Log loss: "
            f"{champion['calibrated_log_loss']:.6f}"
        )

        print(
            "  Brier: "
            f"{champion['calibrated_brier']:.6f}"
        )

        print("")

        print(
            "CHALLENGER RAW"
        )

        print(
            "  Accuracy: "
            f"{candidate['raw_holdout_accuracy']:.6f}"
        )

        print(
            "  Log loss: "
            f"{candidate['raw_holdout_log_loss']:.6f}"
        )

        print(
            "  Brier: "
            f"{candidate['raw_holdout_brier']:.6f}"
        )

        print("")

        print(
            "CHALLENGER CALIBRADO"
        )

        print(
            "  Temperatura: "
            f"{candidate['calibration_temperature']:.2f}"
        )

        print(
            "  Accuracy: "
            f"{candidate['calibrated_holdout_accuracy']:.6f}"
        )

        print(
            "  Log loss: "
            f"{candidate['calibrated_holdout_log_loss']:.6f}"
        )

        print(
            "  Brier: "
            f"{candidate['calibrated_holdout_brier']:.6f}"
        )

        print("")

        print(
            "MEJORA RAW: "
            f"{candidate.get('raw_vs_active_log_loss_improvement')}"
        )

        print(
            "MEJORA CALIBRADA: "
            f"{candidate.get('calibrated_vs_active_log_loss_improvement')}"
        )

        print(
            "MARGEN REQUERIDO: "
            f"{ACTIVATION_MARGIN:.6f}"
        )

    print("")

    print(
        "ACTIVAR: "
        f"{result['active']}"
    )

    print(
        "RAZON: "
        f"{result['activation_reason']}"
    )

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
