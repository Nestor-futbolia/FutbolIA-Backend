import os
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException


router = APIRouter(prefix="/ai", tags=["AI"])


SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "")


def supabase_headers():
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_SECRET_KEY")

    return {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Content-Type": "application/json",
    }


async def supabase_get(table: str, params: dict):
    url = f"{SUPABASE_URL}/rest/v1/{table}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            url,
            headers=supabase_headers(),
            params=params,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase GET {table}: "
            f"{response.status_code} {response.text}"
        )

    return response.json()


async def supabase_post(table: str, payload):
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
        raise RuntimeError(
            f"Supabase POST {table}: "
            f"{response.status_code} {response.text}"
        )

    return response.json()


def get_actual_result(home_goals, away_goals):
    if home_goals is None or away_goals is None:
        return None

    if home_goals > away_goals:
        return "HOME"

    if home_goals < away_goals:
        return "AWAY"

    return "DRAW"


@router.get("/evaluate/{match_id}")
async def evaluate_match(match_id: int):
    """
    Evalúa automáticamente las predicciones 1X2 de un partido finalizado.

    El proceso:
    1. Busca el partido.
    2. Comprueba que tenga resultado final.
    3. Determina HOME/DRAW/AWAY.
    4. Busca las predicciones guardadas.
    5. Inserta prediction_results que todavía no existan.
    6. Devuelve el resultado de la evaluación.
    """

    try:
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
                detail=f"No existe el partido {match_id} en Supabase",
            )

        match = matches[0]

        status = str(match.get("status") or "").upper()

        final_statuses = {"FT", "AET", "PEN"}

        if status not in final_statuses:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El partido {match_id} todavía no está finalizado. "
                    f"Estado actual: {status}"
                ),
            )

        home_goals = match.get("home_goals")
        away_goals = match.get("away_goals")

        actual_result = get_actual_result(
            home_goals,
            away_goals,
        )

        if actual_result is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El partido {match_id} está finalizado, "
                    "pero no tiene goles registrados."
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
                    f"No existen predicciones 1X2 guardadas "
                    f"para el partido {match_id}"
                ),
            )

        prediction_ids = [p["id"] for p in predictions]

        existing_results = await supabase_get(
            "prediction_results",
            {
                "select": "prediction_id",
                "prediction_id": (
                    "in.(" +
                    ",".join(str(x) for x in prediction_ids) +
                    ")"
                ),
            },
        )

        existing_ids = {
            row["prediction_id"]
            for row in existing_results
        }

        new_results = []

        now = datetime.now(timezone.utc).isoformat()

        for prediction in predictions:
            prediction_id = prediction["id"]

            if prediction_id in existing_ids:
                continue

            selection = str(
                prediction.get("selection") or ""
            ).upper()

            correct = selection == actual_result

            new_results.append(
                {
                    "prediction_id": prediction_id,
                    "outcome": correct,
                    "actual_value": actual_result,
                    "evaluated_at": now,
                }
            )

        inserted = []

        if new_results:
            inserted = await supabase_post(
                "prediction_results",
                new_results,
            )

        evaluation = []

        for prediction in predictions:
            selection = str(
                prediction.get("selection") or ""
            ).upper()

            evaluation.append(
                {
                    "prediction_id": prediction["id"],
                    "selection": selection,
                    "probability": prediction.get("probability"),
                    "correct": selection == actual_result,
                }
            )

        correct_predictions = sum(
            1
            for item in evaluation
            if item["correct"]
        )

        return {
            "ok": True,
            "match_id": match_id,
            "status": status,
            "score": {
                "home": home_goals,
                "away": away_goals,
            },
            "actual_result": actual_result,
            "predictions_found": len(predictions),
            "results_inserted": len(inserted),
            "results_already_exist": (
                len(predictions) - len(inserted)
            ),
            "correct_predictions": correct_predictions,
            "evaluation": evaluation,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
      )
