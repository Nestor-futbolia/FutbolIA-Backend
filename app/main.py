import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Fútbol IA 2.0 API",
    version="0.2.0"
)

BASE_URL = "https://v3.football.api-sports.io"


def get_api_key() -> str:
    key = os.getenv("API_FOOTBALL_KEY")

    if not key:
        raise HTTPException(
            status_code=503,
            detail="API_FOOTBALL_KEY no configurada"
        )

    return key


async def football_get(
    path: str,
    params: Optional[dict] = None
):
    headers = {
        "x-apisports-key": get_api_key(),
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            BASE_URL + path,
            params=params or {},
            headers=headers
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text[:1000]
        )

    data = response.json()

    if data.get("errors"):
        raise HTTPException(
            status_code=502,
            detail=data["errors"]
        )

    return data


@app.get("/")
async def root():
    return {
        "app": "Fútbol IA 2.0",
        "status": "online",
        "provider": "API-Football",
        "utc": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health")
async def health():
    return {
        "ok": True
    }


@app.get("/countries")
async def countries():
    return await football_get("/countries")


@app.get("/leagues")
async def leagues(
    country: Optional[str] = None,
    season: Optional[int] = None,
    current: Optional[bool] = None
):
    params = {}

    if country:
        params["country"] = country

    if season:
        params["season"] = season

    if current is not None:
        params["current"] = str(current).lower()

    return await football_get("/leagues", params)


@app.get("/fixtures")
async def fixtures(
    date: Optional[str] = None,
    league: Optional[int] = None,
    season: Optional[int] = None,
    team: Optional[int] = None
):
    params = {}

    if date:
        params["date"] = date

    if league:
        params["league"] = league

    if season:
        params["season"] = season

    if team:
        params["team"] = team

    if not params:
        raise HTTPException(
            status_code=400,
            detail="Indica date, league, season o team"
        )

    return await football_get("/fixtures", params)


@app.get("/fixtures/{fixture_id}")
async def fixture(fixture_id: int):
    return await football_get(
        "/fixtures",
        {"id": fixture_id}
    )


@app.get("/teams/{team_id}")
async def team(team_id: int):
    return await football_get(
        "/teams",
        {"id": team_id}
    )


@app.get("/standings")
async def standings(
    league: int,
    season: int
):
    return await football_get(
        "/standings",
        {
            "league": league,
            "season": season
        }
    )


@app.get("/injuries")
async def injuries(
    league: Optional[int] = None,
    season: Optional[int] = None,
    fixture: Optional[int] = None,
    team: Optional[int] = None
):
    params = {}

    if league:
        params["league"] = league

    if season:
        params["season"] = season

    if fixture:
        params["fixture"] = fixture

    if team:
        params["team"] = team

    return await football_get("/injuries", params)


@app.get("/odds")
async def odds(
    fixture: Optional[int] = None,
    league: Optional[int] = None,
    season: Optional[int] = None
):
    params = {}

    if fixture:
        params["fixture"] = fixture

    if league:
        params["league"] = league

    if season:
        params["season"] = season

    return await football_get("/odds", params)
