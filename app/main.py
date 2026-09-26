import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Fútbol IA 2.0 API",
    version="0.3.0"
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


def get_supabase_config():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SECRET_KEY")

    if not supabase_url or not supabase_key:
        raise HTTPException(
            status_code=503,
            detail="SUPABASE_URL o SUPABASE_SECRET_KEY no configurada"
        )

    return supabase_url, supabase_key


def get_supabase_headers(supabase_key: str):
    return {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }


@app.get("/")
async def root():
    return {
        "app": "Fútbol IA 2.0 API",
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

    if not params:
        raise HTTPException(
            status_code=400,
            detail="Indica fixture, league o season"
        )

    return await football_get("/odds", params)


@app.get("/sync/league")
async def sync_league(
    league: int,
    season: int
):
    data = await football_get(
        "/leagues",
        {
            "id": league,
            "season": season
        }
    )

    if not data.get("response"):
        raise HTTPException(
            status_code=404,
            detail="Liga no encontrada en API-Football"
        )

    item = data["response"][0]

    league_info = item["league"]
    country_info = item.get("country", {})

    row = {
        "id": league_info["id"],
        "name": league_info["name"],
        "country": country_info.get("name"),
        "type": league_info.get("type"),
        "active": True
    }

    supabase_url, supabase_key = get_supabase_config()
    headers = get_supabase_headers(supabase_key)

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{supabase_url}/rest/v1/leagues",
            headers=headers,
            json=row
        )

    if response.status_code >= 300:
        raise HTTPException(
            status_code=502,
            detail=response.text[:1000]
        )

    return {
        "ok": True,
        "message": "Liga sincronizada correctamente",
        "league": row
    }


@app.get("/sync/fixtures")
async def sync_fixtures(
    league: int,
    season: int
):
    data = await football_get(
        "/fixtures",
        {
            "league": league,
            "season": season
        }
    )

    fixtures_data = data.get("response", [])

    if not fixtures_data:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron partidos para esta liga y temporada"
        )

    supabase_url, supabase_key = get_supabase_config()
    headers = get_supabase_headers(supabase_key)

    rows = []

    for item in fixtures_data:
        fixture_info = item.get("fixture", {})
        league_info = item.get("league", {})
        teams_info = item.get("teams", {})
        goals_info = item.get("goals", {})
        score_info = item.get("score", {})
        halftime_info = score_info.get("halftime", {})

        home_team = teams_info.get("home", {})
        away_team = teams_info.get("away", {})

        starting_at = fixture_info.get("date")

        row = {
            "id": fixture_info.get("id"),
            "league_id": league_info.get("id"),
            "season_id": league_info.get("season"),
            "home_team_id": home_team.get("id"),
            "away_team_id": away_team.get("id"),
            "starting_at": starting_at,
            "status": fixture_info.get("status", {}).get("short"),
            "home_goals": goals_info.get("home"),
            "away_goals": goals_info.get("away"),
            "home_ht_goals": halftime_info.get("home"),
            "away_ht_goals": halftime_info.get("away")
        }

        if row["id"] is not None:
            rows.append(row)

    if not rows:
        raise HTTPException(
            status_code=502,
            detail="API-Football devolvió partidos sin identificadores válidos"
        )

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{supabase_url}/rest/v1/matches",
            headers=headers,
            json=rows
        )

    if response.status_code >= 300:
        raise HTTPException(
            status_code=502,
            detail=response.text[:2000]
        )

    return {
        "ok": True,
        "message": "Partidos sincronizados correctamente",
        "league": league,
        "season": season,
        "matches_received": len(fixtures_data),
        "matches_saved": len(rows)
    }
