import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException


router = APIRouter(
    prefix="/backup/football-data",
    tags=["Fuente de respaldo"]
)

BASE_URL = "https://api.football-data.org/v4"


def get_token() -> str:
    token = os.getenv(
        "FOOTBALL_DATA_TOKEN",
        ""
    ).strip()

    if not token:
        raise HTTPException(
            status_code=503,
            detail=(
                "Falta configurar "
                "FOOTBALL_DATA_TOKEN en Render."
            )
        )

    return token


async def football_data_get(
    endpoint: str,
    params: Optional[dict] = None
) -> dict:

    token = get_token()

    try:
        async with httpx.AsyncClient(
            timeout=30
        ) as client:

            response = await client.get(
                f"{BASE_URL}{endpoint}",
                headers={
                    "X-Auth-Token": token
                },
                params=params or {}
            )

    except httpx.RequestError:
        raise HTTPException(
            status_code=502,
            detail=(
                "No fue posible conectar "
                "con football-data.org."
            )
        )

    if response.status_code == 401:
        raise HTTPException(
            status_code=502,
            detail=(
                "football-data.org rechazó "
                "el token. Comprueba la variable "
                "FOOTBALL_DATA_TOKEN en Render."
            )
        )

    if response.status_code == 403:
        raise HTTPException(
            status_code=502,
            detail=(
                "La cuenta no tiene permiso "
                "para consultar este recurso "
                "o competición."
            )
        )

    if response.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail=(
                "Se alcanzó el límite de "
                "solicitudes de football-data.org. "
                "Inténtalo más tarde."
            )
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={
                "source": "football-data.org",
                "provider_status": response.status_code,
                "message": (
                    "El proveedor rechazó "
                    "la solicitud."
                )
            }
        )

    try:
        return response.json()

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail=(
                "football-data.org devolvió "
                "una respuesta no válida."
            )
        )


@router.get("/status")
async def backup_status():

    data = await football_data_get(
        "/competitions"
    )

    competitions = data.get(
        "competitions",
        []
    )

    available = [
        {
            "id": item.get("id"),
            "code": item.get("code"),
            "name": item.get("name"),
            "area": (
                item.get("area") or {}
            ).get("name"),
            "plan": item.get("plan")
        }
        for item in competitions
    ]

    return {
        "ok": True,
        "source": "football-data.org",
        "connected": True,
        "competitions_received": len(
            available
        ),
        "competitions": available
    }


@router.get("/matches")
async def backup_matches(
    competition: str = "PL",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None
):

    params = {}

    if date_from:
        params["dateFrom"] = date_from

    if date_to:
        params["dateTo"] = date_to

    if status:
        params["status"] = status

    data = await football_data_get(
        f"/competitions/{competition}/matches",
        params
    )

    return {
        "ok": True,
        "source": "football-data.org",
        "competition": data.get(
            "competition"
        ),
        "resultSet": data.get(
            "resultSet"
        ),
        "matches": data.get(
            "matches",
            []
        )
    }


@router.get("/standings")
async def backup_standings(
    competition: str = "PL"
):

    data = await football_data_get(
        f"/competitions/{competition}/standings"
    )

    return {
        "ok": True,
        "source": "football-data.org",
        "competition": data.get(
            "competition"
        ),
        "season": data.get(
            "season"
        ),
        "standings": data.get(
            "standings",
            []
        )
    }
