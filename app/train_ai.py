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

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "").strip()

MODEL_NAME = "NESTOR-1X2-LogisticRegression"
MODEL_FAMILY = "1X2"
MODEL_PROTOCOL_VERSION = "NESTOR-EVAL-v1.0"

MIN_MATCHES = 80
WINDOW = 5
PAGE_SIZE = 1000

CALIBRATION_GRID = np.linspace(0.50, 3.00, 101)

ACTIVATION_MARGIN = 0.001


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
    params: dict[str, Any]
) -> list[dict[str, Any]]:

    with httpx.Client(timeout=60.0) as client:
        response = client.get(
            supabase_url(table),
            headers=HEADERS,
            params=params
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
    payload: dict[str, Any]
) -> list[dict[str, Any]]:

    headers = {
        **HEADERS,
        "Prefer": "return=representation"
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            supabase_url(table),
            headers=headers,
            json=payload
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
    payload: dict[str, Any]
) -> list[dict[str, Any]]:

    headers = {
        **HEADERS,
        "Prefer": "return=representation"
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.patch(
            supabase_url(table),
            headers=headers,
            params=params,
            json=payload
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
    rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    seen: set[int] = set()

    cleaned: list[dict[str, Any]] = []

    for row in rows:

        item = clean_match(row)

        if item is None:
            continue

        if item["id"] in seen:
            continue

        seen.add(item["id"])

        cleaned.append(item)

    cleaned.sort(
        key=lambda item: (
            item["starting_at"],
            item["id"]
        )
    )

    return cleaned


def dataset_fingerprint(
    match_ids: list[int]
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
    probabilities: np.ndarray
) -> float:

    one_hot = np.zeros_like(probabilities)

    for idx, target in enumerate(y_true):
        one_hot[idx, int(target)] = 1.0

    return float(
        np.mean(
            np.sum(
                (probabilities - one_hot) ** 2,
                axis=1
            )
        )
    )


def softmax(
    logits: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:

    t = max(
        float(temperature),
        0.05
    )

    z = logits / t

    z = z - np.max(
        z,
        axis=1,
        keepdims=True
    )

    exp_z = np.exp(z)

    return exp_z / np.sum(
        exp_z,
        axis=1,
        keepdims=True
    )


def fit_temperature(
    logits: np.ndarray,
    y: np.ndarray
) -> float:

    best_t = 1.0
    best_loss = math.inf

    for candidate in CALIBRATION_GRID:

        probabilities = softmax(
            logits,
            float(candidate)
        )

        loss = float(
            log_loss(
                y,
                probabilities,
                labels=[0, 1, 2]
            )
        )

        if loss < best_loss - 1e-12:
            best_loss = loss
            best_t = float(candidate)

    return best_t


def model_probabilities(
    model: LogisticRegression,
    scaler: StandardScaler,
    X: np.ndarray,
    temperature: float
) -> np.ndarray:

    Xs = scaler.transform(X)

    logits = (
        Xs @ model.coef_.T
        + model.intercept_
    )

    return softmax(
        logits,
        temperature
    )


def feature_defaults(
    X_train: np.ndarray
) -> dict[str, float]:

    medians = np.median(
        X_train,
        axis=0
    )

    return {
        name: float(value)
        for name, value in zip(
            FEATURE_NAMES,
            medians
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

    return {
        "format_version": "NESTOR-MODEL-v1.0",

        "model_type": "logistic_regression",

        "model_family": MODEL_FAMILY,

        "feature_schema_version": (
            FEATURE_SCHEMA_VERSION
        ),

        "features": FEATURE_NAMES,

        "classes": [
            0,
            1,
            2
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


def probabilities_from_artifact(
    artifact: dict[str, Any],
    X: np.ndarray
) -> np.ndarray:

    coefficients = np.asarray(
        artifact["coefficients"],
        dtype=float
    )

    intercept = np.asarray(
        artifact["intercept"],
        dtype=float
    )

    mean = np.asarray(
        artifact["scaler_mean"],
        dtype=float
    )

    scale = np.asarray(
        artifact["scaler_scale"],
        dtype=float
    )

    scale = np.where(
        np.abs(scale) < 1e-12,
        1.0,
        scale
    )

    Xs = (
        X - mean
    ) / scale

    logits = (
        Xs @ coefficients.T
        + intercept
    )

    temperature = float(
        artifact
        .get("calibration", {})
        .get("temperature", 1.0)
    )

    return softmax(
        logits,
        temperature
    )


def load_active_model() -> dict[str, Any] | None:

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
                "feature_schema_version,"
                "training_config,"
                "validation_start,"
                "validation_end"
            ),
            "active": "eq.true",
            "order": "trained_at.desc",
            "limit": "1",
        },
    )

    if not rows:
        return None

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

    row["artifact"] = artifact

    return row


def current_holdout_evaluation(
    active_row: dict[str, Any] | None,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray
) -> dict[str, float] | None:

    if (
        not active_row
        or not isinstance(
            active_row.get("artifact"),
            dict
        )
    ):
        return None

    artifact = active_row["artifact"]

    if (
        artifact.get(
            "feature_schema_version"
        )
        != FEATURE_SCHEMA_VERSION
    ):
        return None

    try:

        probabilities = probabilities_from_artifact(
            artifact,
            X_holdout
        )

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        return {
            "accuracy": float(
                accuracy_score(
                    y_holdout,
                    predictions
                )
            ),

            "log_loss": float(
                log_loss(
                    y_holdout,
                    probabilities,
                    labels=[0, 1, 2]
                )
            ),

            "brier": brier_multiclass(
                y_holdout,
                probabilities
            ),
        }

    except Exception as exc:

        print(
            "No se pudo reevaluar el "
            "activo en el holdout actual: "
            f"{exc}"
        )

        return None


def save_candidate(
    artifact: dict[str, Any],
    active_row: dict[str, Any] | None,
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

    active_current = None

    if active_row:

        active_current = (
            current_holdout_evaluation(
                active_row,
                np.asarray(
                    candidate_metrics[
                        "_X_holdout"
                    ],
                    dtype=float
                ),
                np.asarray(
                    candidate_metrics[
                        "_y_holdout"
                    ],
                    dtype=int
                ),
            )
        )

    candidate_public = {
        key: value
        for key, value in candidate_metrics.items()
        if not key.startswith("_")
    }

    should_activate = False

    reason = "challenger"

    if active_current is None:

        if active_row is None:

            should_activate = True

            reason = "first_nestor_model"

        else:

            reason = (
                "challenger_protocol_not_comparable"
            )

    else:

        improvement = float(
            active_current["log_loss"]
            - candidate_public[
                "holdout_log_loss"
            ]
        )

        if improvement > ACTIVATION_MARGIN:

            should_activate = True

            reason = (
                "improved_current_holdout_log_loss"
            )

        else:

            reason = (
                "not_better_than_active_on_same_holdout"
            )

    if should_activate:

        supabase_patch(
            "model_versions",
            {
                "active": "eq.true"
            },
            {
                "active": False,
                "status": "retired"
            }
        )

    artifact[
        "evaluation_dataset_fingerprint"
    ] = dataset_fingerprint(
        holdout_ids
    )

    artifact[
        "model_protocol_version"
    ] = MODEL_PROTOCOL_VERSION

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
                artifact[
                    "evaluation_dataset_fingerprint"
                ]
            ),
        },

        "artifact": artifact,

        "active": should_activate,

        "status": (
            "active"
            if should_activate
            else (
                "rejected"
                if active_current
                else "challenger"
            )
        ),

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

            "activation_margin_log_loss": (
                ACTIVATION_MARGIN
            ),
        },
    }

    supabase_post(
        "model_versions",
        row
    )

    return {

        "version": version,

        "active": should_activate,

        "status": row["status"],

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
            artifact[
                "evaluation_dataset_fingerprint"
            ]
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
        "Entrenamiento formal 1X2 v1.0"
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
        dtype=float
    )

    y = np.asarray(
        y_list,
        dtype=int
    )

    train_end = int(
        len(X) * 0.60
    )

    calibration_end = int(
        len(X) * 0.80
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

    X_train = X[:train_end]
    y_train = y[:train_end]

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
        y_train
    )

    cal_logits = (
        X_cal_scaled
        @ model.coef_.T
        + model.intercept_
    )

    temperature = fit_temperature(
        cal_logits,
        y_cal
    )

    holdout_logits = (
        X_holdout_scaled
        @ model.coef_.T
        + model.intercept_
    )

    holdout_probabilities = softmax(
        holdout_logits,
        temperature
    )

    holdout_predictions = np.argmax(
        holdout_probabilities,
        axis=1
    )

    accuracy = float(
        accuracy_score(
            y_holdout,
            holdout_predictions
        )
    )

    holdout_logloss = float(
        log_loss(
            y_holdout,
            holdout_probabilities,
            labels=[0, 1, 2]
        )
    )

    brier = brier_multiclass(
        y_holdout,
        holdout_probabilities
    )

    raw_holdout_probabilities = softmax(
        holdout_logits,
        1.0
    )

    raw_logloss = float(
        log_loss(
            y_holdout,
            raw_holdout_probabilities,
            labels=[0, 1, 2]
        )
    )

    defaults = feature_defaults(
        X_train
    )

    artifact = serialize_artifact(
        model,
        scaler,
        temperature,
        accuracy,
        holdout_logloss,
        len(X_train),
        len(X_cal),
        len(X_holdout),
        defaults,
    )

    active_row = load_active_model()

    candidate_metrics: dict[str, Any] = {

        "holdout_accuracy": accuracy,

        "holdout_log_loss": (
            holdout_logloss
        ),

        "holdout_brier": brier,

        "raw_holdout_log_loss": (
            raw_logloss
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

        "_X_holdout": (
            X_holdout.tolist()
        ),

        "_y_holdout": (
            y_holdout.tolist()
        ),
    }

    match_dates = {
        int(item["id"]): item["starting_at"]
        for item in matches
    }

    validation_start = match_dates.get(
        int(holdout_ids[0]),
        matches[0]["starting_at"]
    )

    validation_end = match_dates.get(
        int(holdout_ids[-1]),
        matches[-1]["starting_at"]
    )

    result = save_candidate(
        artifact,
        active_row,
        holdout_ids,
        candidate_metrics,
        validation_start,
        validation_end,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
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
