import math
import os
from collections import defaultdict
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException


router = APIRouter(
    prefix="/ai",
    tags=["AI"]
)


SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    ""
).rstrip("/")

SUPABASE_SECRET_KEY = os.getenv(
    "SUPABASE_SECRET_KEY",
    "")


FINAL_STATUSES = {
    "FT",
    "AET",
    "PEN"
}


# ============================================================
# SUPABASE
# ============================================================

def supabase_headers(
    prefer: str | None = None
):
    if (
        not SUPABASE_URL
        or not SUPABASE_SECRET_KEY
    ):
        raise RuntimeError(
            "Faltan SUPABASE_URL o SUPABASE_SECRET_KEY"
        )

    headers = {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": (
            f"Bearer {SUPABASE_SECRET_KEY}"
        ),
        "Content-Type": "application/json"
    }

    if prefer:
        headers["Prefer"] = prefer

    return headers


async def supabase_get(
    table: str,
    params: dict
):
    url = (
        f"{SUPABASE_URL}"
        f"/rest/v1/{table}"
    )

    async with httpx.AsyncClient(
        timeout=60.0
    ) as client:

        response = await client.get(
            url,
            headers=supabase_headers(),
            params=params
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase GET {table}: "
            f"{response.status_code} "
            f"{response.text}"
        )

    if not response.text:
        return []

    return response.json()


async def supabase_post(
    table: str,
    payload
):
    url = (
        f"{SUPABASE_URL}"
        f"/rest/v1/{table}"
    )

    async with httpx.AsyncClient(
        timeout=60.0
    ) as client:

        response = await client.post(
            url,
            headers=supabase_headers(
                "return=representation"
            ),
            json=payload
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase POST {table}: "
            f"{response.status_code} "
            f"{response.text}"
        )

    if not response.text:
        return []

    return response.json()


# ============================================================
# UTILIDADES
# ============================================================

def get_actual_result(
    home_goals,
    away_goals
):
    if (
        home_goals is None
        or away_goals is None
    ):
        return None

    if home_goals > away_goals:
        return "HOME"

    if home_goals < away_goals:
        return "AWAY"

    return "DRAW"


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# EVALUAR UN PARTIDO
# ============================================================

async def evaluate_match_internal(
    match_id: int
):

    matches = await supabase_get(
        "matches",
        {
            "select": (
                "id,status,home_goals,"
                "away_goals,starting_at,"
                "home_team_id,away_team_id"
            ),
            "id": f"eq.{match_id}",
            "limit": "1"
        }
    )

    if not matches:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No existe el partido "
                f"{match_id} en Supabase"
            )
        )

    match = matches[0]

    status = str(
        match.get("status") or ""
    ).upper()

    if status not in FINAL_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"El partido {match_id} "
                f"todavía no está finalizado. "
                f"Estado actual: {status}"
            )
        )

    home_goals = match.get(
        "home_goals"
    )

    away_goals = match.get(
        "away_goals"
    )

    actual_result = get_actual_result(
        home_goals,
        away_goals
    )

    if actual_result is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"El partido {match_id} "
                "no tiene goles registrados"
            )
        )

    predictions = await supabase_get(
        "predictions",
        {
            "select": (
                "id,match_id,model_version,"
                "market,selection,probability,"
                "predicted_at"
            ),
            "match_id": f"eq.{match_id}",
            "market": "eq.1X2",
            "order": "id.asc"
        }
    )

    if not predictions:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No existen predicciones "
                f"1X2 para el partido {match_id}"
            )
        )

    prediction_ids = [
        p["id"]
        for p in predictions
    ]

    existing_results = await supabase_get(
        "prediction_results",
        {
            "select": "prediction_id",
            "prediction_id": (
                "in.("
                + ",".join(
                    str(x)
                    for x in prediction_ids
                )
                + ")"
            )
        }
    )

    existing_ids = {
        row["prediction_id"]
        for row in existing_results
    }

    new_results = []

    timestamp = utc_now()

    for prediction in predictions:

        prediction_id = prediction["id"]

        if prediction_id in existing_ids:
            continue

        selection = str(
            prediction.get("selection") or ""
        ).upper()

        correct = (
            selection == actual_result
        )

        new_results.append(
            {
                "prediction_id": prediction_id,
                "outcome": correct,
                "actual_value": actual_result,
                "evaluated_at": timestamp
            }
        )

    inserted = []

    if new_results:
        inserted = await supabase_post(
            "prediction_results",
            new_results
        )

    evaluation = []

    for prediction in predictions:

        selection = str(
            prediction.get("selection") or ""
        ).upper()

        probability = prediction.get(
            "probability"
        )

        evaluation.append(
            {
                "prediction_id": prediction["id"],
                "selection": selection,
                "probability": probability,
                "correct": (
                    selection == actual_result
                )
            }
        )

    correct_predictions = sum(
        1
        for item in evaluation
        if item["correct"]
    )

    return {
        "match_id": match_id,
        "status": status,
        "score": {
            "home": home_goals,
            "away": away_goals
        },
        "actual_result": actual_result,
        "predictions_found": len(
            predictions
        ),
        "results_inserted": len(
            inserted
        ),
        "results_already_exist": (
            len(predictions)
            - len(inserted)
        ),
        "correct_predictions": (
            correct_predictions
        ),
        "evaluation": evaluation
    }


# ============================================================
# EVALUAR UN PARTIDO
# ============================================================

@router.get(
    "/evaluate/{match_id}"
)
async def evaluate_match(
    match_id: int
):

    try:

        result = await evaluate_match_internal(
            match_id
        )

        return {
            "ok": True,
            **result
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# EVALUACIÓN AUTOMÁTICA POR LOTES
# ============================================================

@router.get(
    "/evaluate/batch"
)
async def evaluate_batch(
    limit: int = 10
):

    if limit < 1 or limit > 50:
        raise HTTPException(
            status_code=400,
            detail=(
                "El límite debe estar "
                "entre 1 y 50"
            )
        )

    try:

        matches = await supabase_get(
            "matches",
            {
                "select": (
                    "id,status,starting_at,"
                    "home_goals,away_goals"
                ),
                "status": (
                    "in.(FT,AET,PEN)"
                ),
                "order": (
                    "starting_at.desc"
                ),
                "limit": str(limit)
            }
        )

        if not matches:
            return {
                "ok": True,
                "message": (
                    "No hay partidos finalizados"
                ),
                "selected": 0,
                "evaluated": 0,
                "failed": 0,
                "details": []
            }

        evaluated = []
        failed = []

        for match in matches:

            match_id = match.get("id")

            if match_id is None:
                continue

            try:

                result = (
                    await evaluate_match_internal(
                        int(match_id)
                    )
                )

                evaluated.append(
                    result
                )

            except HTTPException as exc:

                failed.append(
                    {
                        "match_id": match_id,
                        "status_code": (
                            exc.status_code
                        ),
                        "detail": exc.detail
                    }
                )

            except Exception as exc:

                failed.append(
                    {
                        "match_id": match_id,
                        "detail": str(exc)
                    }
                )

        return {
            "ok": True,
            "selected": len(matches),
            "evaluated": len(evaluated),
            "failed": len(failed),
            "results_inserted": sum(
                item["results_inserted"]
                for item in evaluated
            ),
            "details": {
                "evaluated": evaluated,
                "failed": failed
            }
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# OBTENER PREDICCIONES EVALUADAS
# ============================================================

async def get_all_prediction_results(
    max_rows: int = 5000
):

    page_size = 1000
    offset = 0
    rows = []

    while len(rows) < max_rows:

        current_limit = min(
            page_size,
            max_rows - len(rows)
        )

        page = await supabase_get(
            "prediction_results",
            {
                "select": (
                    "id,prediction_id,"
                    "outcome,actual_value,"
                    "evaluated_at"
                ),
                "order": "id.asc",
                "limit": str(current_limit),
                "offset": str(offset)
            }
        )

        if not page:
            break

        rows.extend(page)

        if len(page) < current_limit:
            break

        offset += len(page)

    return rows


async def get_predictions_by_ids(
    prediction_ids: list[int]
):

    if not prediction_ids:
        return []

    all_predictions = []

    chunk_size = 200

    for start in range(
        0,
        len(prediction_ids),
        chunk_size
    ):

        chunk = prediction_ids[
            start:start + chunk_size
        ]

        filter_value = (
            "in.("
            + ",".join(
                str(x)
                for x in chunk
            )
            + ")"
        )

        rows = await supabase_get(
            "predictions",
            {
                "select": (
                    "id,match_id,model_version,"
                    "market,selection,probability,"
                    "predicted_at"
                ),
                "id": filter_value,
                "limit": str(
                    len(chunk)
                )
            }
        )

        all_predictions.extend(
            rows
        )

    return all_predictions


# ============================================================
# RENDIMIENTO DEL MODELO
# ============================================================

@router.get(
    "/performance"
)
async def ai_performance(
    max_matches: int = 5000,
    model_version: str | None = None
):

    if max_matches < 1 or max_matches > 5000:
        raise HTTPException(
            status_code=400,
            detail=(
                "max_matches debe estar "
                "entre 1 y 5000"
            )
        )

    try:

        results = (
            await get_all_prediction_results(
                max_rows=max_matches * 3
            )
        )

        if not results:
            return {
                "ok": True,
                "message": (
                    "Todavía no existen "
                    "resultados evaluados"
                ),
                "evaluated_matches": 0
            }

        prediction_ids = [
            int(row["prediction_id"])
            for row in results
            if row.get("prediction_id")
            is not None
        ]

        predictions = (
            await get_predictions_by_ids(
                prediction_ids
            )
        )

        predictions_by_id = {
            int(p["id"]): p
            for p in predictions
        }

        match_predictions = defaultdict(
            list
        )

        for result in results:

            prediction_id = result.get(
                "prediction_id"
            )

            prediction = (
                predictions_by_id.get(
                    int(prediction_id)
                )
                if prediction_id is not None
                else None
            )

            if not prediction:
                continue

            if str(
                prediction.get("market") or ""
            ).upper() != "1X2":
                continue

            current_model = str(
                prediction.get(
                    "model_version"
                ) or ""
            )

            if (
                model_version
                and current_model
                != model_version
            ):
                continue

            match_id = prediction.get(
                "match_id"
            )

            if match_id is None:
                continue

            match_predictions[
                int(match_id)
            ].append(
                {
                    "prediction": prediction,
                    "result": result
                }
            )

        accuracy_total = 0
        log_loss_total = 0.0
        brier_total = 0.0

        evaluated_matches = 0

        by_model = defaultdict(
            lambda: {
                "matches": 0,
                "correct": 0,
                "log_loss": 0.0,
                "brier_score": 0.0
            }
        )

        for match_id, rows in (
            match_predictions.items()
        ):

            probability_by_selection = {}

            actual_result = None
            selected_prediction = None

            for row in rows:

                prediction = row[
                    "prediction"
                ]

                result = row[
                    "result"
                ]

                selection = str(
                    prediction.get(
                        "selection"
                    ) or ""
                ).upper()

                probability = (
                    prediction.get(
                        "probability"
                    )
                )

                if probability is None:
                    continue

                try:
                    probability = float(
                        probability
                    )
                except Exception:
                    continue

                probability = max(
                    0.0,
                    min(
                        1.0,
                        probability
                    )
                )

                probability_by_selection[
                    selection
                ] = probability

                actual_result = str(
                    result.get(
                        "actual_value"
                    ) or ""
                ).upper()

                if (
                    selected_prediction
                    is None
                    or probability
                    > float(
                        selected_prediction[
                            "probability"
                        ]
                    )
                ):
                    selected_prediction = {
                        "selection": selection,
                        "probability": probability,
                        "model_version": (
                            prediction.get(
                                "model_version"
                            )
                        )
                    }

            if (
                actual_result
                not in {
                    "HOME",
                    "DRAW",
                    "AWAY"
                }
            ):
                continue

            if not probability_by_selection:
                continue

            evaluated_matches += 1

            predicted_result = (
                selected_prediction[
                    "selection"
                ]
                if selected_prediction
                else None
            )

            correct = (
                predicted_result
                == actual_result
            )

            if correct:
                accuracy_total += 1

            actual_probability = (
                probability_by_selection.get(
                    actual_result,
                    0.0
                )
            )

            epsilon = 1e-15

            actual_probability = max(
                epsilon,
                min(
                    1.0 - epsilon,
                    actual_probability
                )
            )

            match_log_loss = -math.log(
                actual_probability
            )

            log_loss_total += (
                match_log_loss
            )

            brier_score = 0.0

            for selection in (
                "HOME",
                "DRAW",
                "AWAY"
            ):

                probability = (
                    probability_by_selection.get(
                        selection,
                        0.0
                    )
                )

                expected = (
                    1.0
                    if selection
                    == actual_result
                    else 0.0
                )

                brier_score += (
                    probability
                    - expected
                ) ** 2

            brier_total += brier_score

            model_name = str(
                selected_prediction.get(
                    "model_version"
                )
                or "unknown"
            )

            by_model[model_name][
                "matches"
            ] += 1

            if correct:
                by_model[model_name][
                    "correct"
                ] += 1

            by_model[model_name][
                "log_loss"
            ] += match_log_loss

            by_model[model_name][
                "brier_score"
            ] += brier_score

        if evaluated_matches == 0:
            return {
                "ok": True,
                "message": (
                    "No hay suficientes "
                    "predicciones 1X2 evaluadas"
                ),
                "evaluated_matches": 0
            }

        model_summary = {}

        for name, data in (
            by_model.items()
        ):

            matches = data["matches"]

            model_summary[name] = {
                "matches": matches,
                "accuracy": round(
                    data["correct"]
                    / matches,
                    6
                ),
                "accuracy_percent": round(
                    (
                        data["correct"]
                        / matches
                    ) * 100,
                    2
                ),
                "log_loss": round(
                    data["log_loss"]
                    / matches,
                    6
                ),
                "brier_score": round(
                    data["brier_score"]
                    / matches,
                    6
                )
            }

        return {
            "ok": True,
            "market": "1X2",
            "model_version": (
                model_version
                if model_version
                else "ALL"
            ),
            "evaluated_matches": (
                evaluated_matches
            ),
            "correct_matches": (
                accuracy_total
            ),
            "accuracy": round(
                accuracy_total
                / evaluated_matches,
                6
            ),
            "accuracy_percent": round(
                (
                    accuracy_total
                    / evaluated_matches
                ) * 100,
                2
            ),
            "log_loss": round(
                log_loss_total
                / evaluated_matches,
                6
            ),
            "brier_score": round(
                brier_total
                / evaluated_matches,
                6
            ),
            "models": model_summary
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )
