import os
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.ai_predict import predict_match


router = APIRouter(prefix="/ai", tags=["AI"])


SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "")

FINAL_STATUSES = {"FT", "AET", "PEN"}


def supabase_headers() -> dict[str, str]:
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise RuntimeError(
            "Faltan SUPABASE_URL o SUPABASE_SECRET_KEY"
        )

    return {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Content-Type": "application/json",
    }


async def supabase_get(
    table: str,
    params: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    url = f"{SUPABASE_URL}/rest/v1/{table}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            url,
            headers=supabase_headers(),
            params=params or {},
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Supabase GET {table}: "
                f"{response.status_code} {response.text}"
            ),
        )

    data = response.json()

    if not isinstance(data, list):
        return []

    return data


async def supabase_post(
    table: str,
    payload: Any,
) -> list[dict[str, Any]]:
    url = f"{SUPABASE_URL}/rest/v1/{table}"

    headers = supabase_headers()
    headers["Prefer"] = "return=representation"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            headers=headers,
            json=payload,
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Supabase POST {table}: "
                f"{response.status_code} {response.text}"
            ),
        )

    if not response.text:
        return []

    data = response.json()

    if isinstance(data, list):
        return data

    return []


def get_actual_result(
    home_goals: Optional[int],
    away_goals: Optional[int],
) -> Optional[str]:

    if home_goals is None or away_goals is None:
        return None

    if home_goals > away_goals:
        return "HOME"

    if home_goals == away_goals:
        return "DRAW"

    return "AWAY"


# ============================================================
# PREDICCIONES PARA PARTIDOS FUTUROS
# ============================================================

@router.get("/predict/upcoming")
async def predict_upcoming(
    limit: int = Query(
        10,
        ge=1,
        le=50,
        description="Cantidad máxima de partidos futuros",
    )
):
    """
    Busca partidos cuyo inicio todavía no ha ocurrido
    y genera automáticamente una predicción 1X2.

    No utiliza el marcador final porque estos partidos
    todavía no han comenzado.
    """

    now_utc = datetime.now(timezone.utc)

    matches = await supabase_get(
        "matches",
        {
            "select": (
                "id,status,home_goals,away_goals,"
                "starting_at,home_team_id,away_team_id"
            ),
            "starting_at": f"gte.{now_utc.isoformat()}",
            "order": "starting_at.asc",
            "limit": str(limit),
        },
    )

    selected = len(matches)
    predicted = 0
    skipped = 0
    failed = 0

    details = []

    for match in matches:
        match_id = match.get("id")

        if match_id is None:
            continue

        match_id = int(match_id)

        try:
            existing = await supabase_get(
                "predictions",
                {
                    "select": (
                        "id,model_version,"
                        "market,selection,probability"
                    ),
                    "match_id": f"eq.{match_id}",
                    "market": "eq.1X2",
                    "limit": "1",
                },
            )

            if existing:
                skipped += 1

                details.append(
                    {
                        "match_id": match_id,
                        "starting_at": match.get(
                            "starting_at"
                        ),
                        "ok": True,
                        "status": "skipped",
                        "reason": (
                            "Ya existen predicciones 1X2"
                        ),
                    }
                )

                continue

            result = await predict_match(match_id)

            predicted += 1

            details.append(
                {
                    "match_id": match_id,
                    "starting_at": match.get(
                        "starting_at"
                    ),
                    "ok": True,
                    "status": "predicted",
                    "prediction": result.get(
                        "prediction"
                    ),
                    "prediction_percentage": result.get(
                        "prediction_percentage"
                    ),
                    "probabilities": result.get(
                        "probabilities"
                    ),
                    "model_version": result.get(
                        "model_version"
                    ),
                }
            )

        except Exception as exc:
            failed += 1

            details.append(
                {
                    "match_id": match_id,
                    "starting_at": match.get(
                        "starting_at"
                    ),
                    "ok": False,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return {
        "ok": True,
        "current_time_utc": now_utc.isoformat(),
        "selected": selected,
        "predicted": predicted,
        "skipped": skipped,
        "failed": failed,
        "details": details,
    }


# ============================================================
# GENERAR PREDICCIONES 1X2 EN LOTE
# ============================================================

@router.get("/predict/batch")
async def predict_batch(
    limit: int = Query(
        10,
        ge=1,
        le=50,
        description="Cantidad máxima de partidos",
    )
):
    matches = await supabase_get(
        "matches",
        {
            "select": (
                "id,status,home_goals,away_goals,"
                "starting_at,home_team_id,away_team_id"
            ),
            "status": "in.(FT,AET,PEN)",
            "order": "starting_at.desc",
            "limit": str(limit),
        },
    )

    selected = len(matches)
    predicted = 0
    skipped = 0
    failed = 0
    details = []

    for match in matches:
        match_id = match.get("id")

        if match_id is None:
            continue

        match_id = int(match_id)

        try:
            existing = await supabase_get(
                "predictions",
                {
                    "select": "id",
                    "match_id": f"eq.{match_id}",
                    "market": "eq.1X2",
                    "limit": "1",
                },
            )

            if existing:
                skipped += 1

                details.append(
                    {
                        "match_id": match_id,
                        "ok": True,
                        "status": "skipped",
                        "reason": (
                            "Ya existen predicciones 1X2"
                        ),
                    }
                )

                continue

            result = await predict_match(match_id)

            predicted += 1

            details.append(
                {
                    "match_id": match_id,
                    "ok": True,
                    "status": "predicted",
                    "prediction": result.get(
                        "prediction"
                    ),
                    "prediction_percentage": result.get(
                        "prediction_percentage"
                    ),
                    "model_version": result.get(
                        "model_version"
                    ),
                }
            )

        except Exception as exc:
            failed += 1

            details.append(
                {
                    "match_id": match_id,
                    "ok": False,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return {
        "ok": True,
        "selected": selected,
        "predicted": predicted,
        "skipped": skipped,
        "failed": failed,
        "details": details,
    }


# ============================================================
# EVALUAR UN PARTIDO
# ============================================================

async def evaluate_match_internal(
    match_id: int,
) -> dict[str, Any]:

    matches = await supabase_get(
        "matches",
        {
            "select": (
                "id,status,home_goals,away_goals,"
                "starting_at,home_team_id,away_team_id"
            ),
            "id": f"eq.{match_id}",
            "limit": "1",
        },
    )

    if not matches:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No existe el partido {match_id} "
                "en Supabase"
            ),
        )

    match = matches[0]

    status = str(
        match.get("status") or ""
    ).upper()

    if status not in FINAL_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"El partido {match_id} todavía "
                f"no está finalizado. Estado: {status}"
            ),
        )

    actual_result = get_actual_result(
        match.get("home_goals"),
        match.get("away_goals"),
    )

    if actual_result is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"El partido {match_id} no tiene "
                "marcador final."
            ),
        )

    predictions = await supabase_get(
        "predictions",
        {
            "select": (
                "id,match_id,model_version,market,"
                "selection,probability,predicted_at"
            ),
            "match_id": f"eq.{match_id}",
            "market": "eq.1X2",
            "order": "id.asc",
        },
    )

    if not predictions:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No existen predicciones 1X2 "
                f"para el partido {match_id}"
            ),
        )

    prediction_ids = [
        int(p["id"])
        for p in predictions
        if p.get("id") is not None
    ]

    existing_results = []

    if prediction_ids:
        id_values = ",".join(
            str(x) for x in prediction_ids
        )

        existing_results = await supabase_get(
            "prediction_results",
            {
                "select": (
                    "id,prediction_id,outcome,"
                    "actual_value,evaluated_at"
                ),
                "prediction_id": f"in.({id_values})",
            },
        )

    existing_by_prediction = {
        int(r["prediction_id"]): r
        for r in existing_results
        if r.get("prediction_id") is not None
    }

    rows_to_insert = []

    for prediction in predictions:
        prediction_id = prediction.get("id")

        if prediction_id is None:
            continue

        prediction_id = int(prediction_id)

        if prediction_id in existing_by_prediction:
            continue

        selection = str(
            prediction.get("selection") or ""
        ).upper()

        rows_to_insert.append(
            {
                "prediction_id": prediction_id,
                "outcome": (
                    selection == actual_result
                ),
                "actual_value": actual_result,
            }
        )

    inserted_rows = []

    if rows_to_insert:
        inserted_rows = await supabase_post(
            "prediction_results",
            rows_to_insert,
        )

    evaluations = []

    for prediction in predictions:
        prediction_id = prediction.get("id")

        if prediction_id is None:
            continue

        prediction_id = int(prediction_id)

        selection = str(
            prediction.get("selection") or ""
        ).upper()

        result_row = existing_by_prediction.get(
            prediction_id
        )

        if result_row is None:
            for inserted in inserted_rows:
                if (
                    int(
                        inserted.get(
                            "prediction_id",
                            -1,
                        )
                    )
                    == prediction_id
                ):
                    result_row = inserted
                    break

        evaluations.append(
            {
                "prediction_id": prediction_id,
                "model_version": prediction.get(
                    "model_version"
                ),
                "selection": selection,
                "probability": prediction.get(
                    "probability"
                ),
                "correct": (
                    selection == actual_result
                ),
                "actual_result": actual_result,
                "result_id": (
                    result_row.get("id")
                    if result_row
                    else None
                ),
            }
        )

    correct_predictions = sum(
        1
        for item in evaluations
        if item["correct"]
    )

    return {
        "ok": True,
        "match_id": match_id,
        "status": status,
        "score": {
            "home": match.get("home_goals"),
            "away": match.get("away_goals"),
        },
        "actual_result": actual_result,
        "predictions_found": len(predictions),
        "results_inserted": len(inserted_rows),
        "results_already_exist": len(
            existing_results
        ),
        "correct_predictions": correct_predictions,
        "evaluations": evaluations,
    }


# ============================================================
# EVALUAR VARIOS PARTIDOS
# ============================================================

@router.get("/evaluate/batch")
async def evaluate_batch(
    limit: int = Query(
        10,
        ge=1,
        le=50,
        description="Cantidad máxima de partidos",
    )
):
    matches = await supabase_get(
        "matches",
        {
            "select": (
                "id,status,home_goals,away_goals,"
                "starting_at,home_team_id,away_team_id"
            ),
            "status": "in.(FT,AET,PEN)",
            "order": "starting_at.desc",
            "limit": str(limit),
        },
    )

    selected = len(matches)
    evaluated = 0
    failed = 0
    results_inserted = 0

    details = []

    for match in matches:
        match_id = match.get("id")

        if match_id is None:
            continue

        try:
            result = await evaluate_match_internal(
                int(match_id)
            )

            evaluated += 1

            results_inserted += int(
                result.get(
                    "results_inserted",
                    0,
                )
            )

            details.append(
                {
                    "match_id": int(match_id),
                    "ok": True,
                    "actual_result": result.get(
                        "actual_result"
                    ),
                    "predictions_found": result.get(
                        "predictions_found",
                        0,
                    ),
                    "results_inserted": result.get(
                        "results_inserted",
                        0,
                    ),
                    "results_already_exist": result.get(
                        "results_already_exist",
                        0,
                    ),
                    "correct_predictions": result.get(
                        "correct_predictions",
                        0,
                    ),
                }
            )

        except Exception as exc:
            failed += 1

            details.append(
                {
                    "match_id": int(match_id),
                    "ok": False,
                    "error": str(exc),
                }
            )

    return {
        "ok": True,
        "selected": selected,
        "evaluated": evaluated,
        "failed": failed,
        "results_inserted": results_inserted,
        "details": details,
    }


# ============================================================
# EVALUAR UN SOLO PARTIDO
# ============================================================

@router.get("/evaluate/{match_id}")
async def evaluate_match(match_id: int):
    return await evaluate_match_internal(
        match_id
    )


# ============================================================
# OBTENER RESULTADOS DE PREDICCIONES
# ============================================================

async def get_all_prediction_results(
    max_rows: int = 5000,
) -> list[dict[str, Any]]:

    all_rows = []

    page_size = 1000
    offset = 0

    while len(all_rows) < max_rows:
        remaining = max_rows - len(all_rows)

        current_limit = min(
            page_size,
            remaining,
        )

        rows = await supabase_get(
            "prediction_results",
            {
                "select": (
                    "id,prediction_id,outcome,"
                    "actual_value,evaluated_at"
                ),
                "order": "id.asc",
                "offset": str(offset),
                "limit": str(current_limit),
            },
        )

        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < current_limit:
            break

        offset += len(rows)

    return all_rows[:max_rows]


async def get_predictions_by_ids(
    prediction_ids: list[int],
) -> list[dict[str, Any]]:

    if not prediction_ids:
        return []

    unique_ids = sorted(
        {
            int(x)
            for x in prediction_ids
        }
    )

    all_predictions = []

    chunk_size = 200

    for start in range(
        0,
        len(unique_ids),
        chunk_size,
    ):
        chunk = unique_ids[
            start:start + chunk_size
        ]

        id_values = ",".join(
            str(x)
            for x in chunk
        )

        rows = await supabase_get(
            "predictions",
            {
                "select": (
                    "id,match_id,model_version,"
                    "market,selection,probability,"
                    "predicted_at"
                ),
                "id": f"in.({id_values})",
            },
        )

        all_predictions.extend(rows)

    return all_predictions


# ============================================================
# RENDIMIENTO
# ============================================================

@router.get("/performance")
async def ai_performance(
    max_matches: int = Query(
        5000,
        ge=1,
        le=5000,
    ),
    model_version: Optional[str] = Query(
        None,
    ),
):
    max_result_rows = max_matches * 3

    results = await get_all_prediction_results(
        max_rows=max_result_rows
    )

    if not results:
        return {
            "ok": True,
            "evaluated_matches": 0,
            "correct_matches": 0,
            "accuracy": None,
            "accuracy_percent": None,
            "log_loss": None,
            "brier_score": None,
            "models": [],
            "message": (
                "Todavía no existen resultados "
                "evaluados."
            ),
        }

    prediction_ids = [
        int(row["prediction_id"])
        for row in results
        if row.get("prediction_id") is not None
    ]

    predictions = await get_predictions_by_ids(
        prediction_ids
    )

    prediction_map = {
        int(p["id"]): p
        for p in predictions
        if p.get("id") is not None
    }

    grouped = defaultdict(list)

    for result in results:
        prediction_id = result.get(
            "prediction_id"
        )

        if prediction_id is None:
            continue

        prediction = prediction_map.get(
            int(prediction_id)
        )

        if not prediction:
            continue

        if str(
            prediction.get("market") or ""
        ).upper() != "1X2":
            continue

        if (
            model_version is not None
            and prediction.get("model_version")
            != model_version
        ):
            continue

        grouped[
            int(prediction["match_id"])
        ].append(
            {
                "result": result,
                "prediction": prediction,
            }
        )

    match_records = []

    for match_id, rows in grouped.items():
        valid_rows = []

        for row in rows:
            probability = row[
                "prediction"
            ].get("probability")

            selection = str(
                row["prediction"].get(
                    "selection"
                ) or ""
            ).upper()

            actual = str(
                row["result"].get(
                    "actual_value"
                ) or ""
            ).upper()

            if probability is None:
                continue

            try:
                probability = float(
                    probability
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if not math.isfinite(
                probability
            ):
                continue

            if selection not in {
                "HOME",
                "DRAW",
                "AWAY",
            }:
                continue

            if actual not in {
                "HOME",
                "DRAW",
                "AWAY",
            }:
                continue

            valid_rows.append(
                {
                    "selection": selection,
                    "probability": probability,
                    "actual": actual,
                    "model_version": row[
                        "prediction"
                    ].get(
                        "model_version"
                    ),
                }
            )

        if not valid_rows:
            continue

        best = max(
            valid_rows,
            key=lambda x: x["probability"],
        )

        match_records.append(best)

    if not match_records:
        return {
            "ok": True,
            "evaluated_matches": 0,
            "correct_matches": 0,
            "accuracy": None,
            "accuracy_percent": None,
            "log_loss": None,
            "brier_score": None,
            "models": [],
            "message": (
                "No hay predicciones 1X2 "
                "válidas para analizar."
            ),
        }

    evaluated_matches = len(
        match_records
    )

    correct_matches = sum(
        1
        for record in match_records
        if record["selection"]
        == record["actual"]
    )

    accuracy = (
        correct_matches
        / evaluated_matches
    )

    log_loss_values = []

    for record in match_records:
        probability = max(
            1e-15,
            min(
                1.0 - 1e-15,
                float(
                    record["probability"]
                ),
            ),
        )

        if (
            record["selection"]
            == record["actual"]
        ):
            value = probability
        else:
            value = max(
                1e-15,
                (1.0 - probability)
                / 2.0,
            )

        log_loss_values.append(
            -math.log(value)
        )

    log_loss = (
        sum(log_loss_values)
        / len(log_loss_values)
    )

    # ========================================================
    # BRIER SCORE
    # ========================================================

    brier_values = []

    grouped_brier = defaultdict(list)

    for match_id, rows in grouped.items():
        for row in rows:
            prediction = row[
                "prediction"
            ]

            probability = prediction.get(
                "probability"
            )

            if probability is None:
                continue

            try:
                probability = float(
                    probability
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            selection = str(
                prediction.get(
                    "selection"
                ) or ""
            ).upper()

            actual = str(
                row["result"].get(
                    "actual_value"
                ) or ""
            ).upper()

            if selection not in {
                "HOME",
                "DRAW",
                "AWAY",
            }:
                continue

            if actual not in {
                "HOME",
                "DRAW",
                "AWAY",
            }:
                continue

            grouped_brier[match_id].append(
                {
                    "selection": selection,
                    "probability": probability,
                    "actual": actual,
                }
            )

    for match_id, rows in grouped_brier.items():
        probabilities = {
            "HOME": 0.0,
            "DRAW": 0.0,
            "AWAY": 0.0,
        }

        actual = None

        for row in rows:
            probabilities[
                row["selection"]
            ] = max(
                0.0,
                min(
                    1.0,
                    row["probability"],
                ),
            )

            actual = row["actual"]

        if actual is None:
            continue

        score = 0.0

        for selection in (
            "HOME",
            "DRAW",
            "AWAY",
        ):
            target = (
                1.0
                if selection == actual
                else 0.0
            )

            score += (
                probabilities[
                    selection
                ] - target
            ) ** 2

        brier_values.append(score)

    brier_score = (
        sum(brier_values)
        / len(brier_values)
        if brier_values
        else None
    )

    # ========================================================
    # RESUMEN POR MODELO
    # ========================================================

    models_grouped = defaultdict(list)

    for record in match_records:
        models_grouped[
            record["model_version"]
            or "unknown"
        ].append(record)

    models_summary = []

    for version, records in (
        models_grouped.items()
    ):
        total = len(records)

        correct = sum(
            1
            for record in records
            if record["selection"]
            == record["actual"]
        )

        model_accuracy = (
            correct / total
            if total
            else None
        )

        models_summary.append(
            {
                "model_version": version,
                "evaluated_matches": total,
                "correct_matches": correct,
                "accuracy": model_accuracy,
                "accuracy_percent": (
                    round(
                        model_accuracy * 100,
                        2,
                    )
                    if model_accuracy
                    is not None
                    else None
                ),
            }
        )

    return {
        "ok": True,
        "evaluated_matches": evaluated_matches,
        "correct_matches": correct_matches,
        "accuracy": accuracy,
        "accuracy_percent": round(
            accuracy * 100,
            2,
        ),
        "log_loss": log_loss,
        "brier_score": brier_score,
        "models": models_summary,
    }
