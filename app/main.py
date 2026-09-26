import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Fútbol IA 2.0 API",
    version="0.5.0"
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

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(
            BASE_URL + path,
            params=params or {},
            headers=headers
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text[:2000]
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


async def supabase_upsert(
    table: str,
    rows,
    supabase_url: str,
    supabase_key: str
):
    headers = get_supabase_headers(supabase_key)

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{supabase_url}/rest/v1/{table}",
            headers=headers,
            json=rows
        )

    if response.status_code >= 300:
        raise HTTPException(
            status_code=502,
            detail=f"Error Supabase en {table}: {response.text[:3000]}"
        )

    return response


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

    await supabase_upsert(
        "leagues",
        row,
        supabase_url,
        supabase_key
    )

    season_info = item.get("seasons", [])

    season_rows = []

    for season_item in season_info:
        season_id = season_item.get("year")

        if season_id is None:
            continue

        season_rows.append({
            "id": season_id,
            "league_id": league_info["id"],
            "name": str(season_item.get("year")),
            "starting_at": season_item.get("start"),
            "ending_at": season_item.get("end")
        })

    if season_rows:
        await supabase_upsert(
            "seasons",
            season_rows,
            supabase_url,
            supabase_key
        )

    return {
        "ok": True,
        "message": "Liga y temporadas sincronizadas correctamente",
        "league": row,
        "seasons_saved": len(season_rows)
    }


@app.get("/sync/teams")
async def sync_teams(
    league: int,
    season: int
):
    data = await football_get(
        "/teams",
        {
            "league": league,
            "season": season
        }
    )

    teams_data = data.get("response", [])

    if not teams_data:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron equipos"
        )

    supabase_url, supabase_key = get_supabase_config()

    rows = []

    for item in teams_data:
        team_info = item.get("team", {})
        venue_info = item.get("venue", {})

        team_id = team_info.get("id")

        if team_id is None:
            continue

        rows.append({
            "id": team_id,
            "name": team_info.get("name"),
            "short_code": team_info.get("code"),
            "country": team_info.get("country"),
            "venue_name": venue_info.get("name"),
            "league_id": league
        })

    if not rows:
        raise HTTPException(
            status_code=502,
            detail="API-Football no devolvió equipos válidos"
        )

    await supabase_upsert(
        "teams",
        rows,
        supabase_url,
        supabase_key
    )

    return {
        "ok": True,
        "message": "Equipos sincronizados correctamente",
        "league": league,
        "season": season,
        "teams_saved": len(rows)
    }


@app.get("/sync/fixtures")
async def sync_fixtures(
    league: int,
    season: int
):
    await sync_league(
        league=league,
        season=season
    )

    await sync_teams(
        league=league,
        season=season
    )

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

        fixture_id = fixture_info.get("id")

        if fixture_id is None:
            continue

        rows.append({
            "id": fixture_id,
            "league_id": league_info.get("id"),
            "season_id": league_info.get("season"),
            "home_team_id": home_team.get("id"),
            "away_team_id": away_team.get("id"),
            "starting_at": fixture_info.get("date"),
            "status": fixture_info.get("status", {}).get("short"),
            "home_goals": goals_info.get("home"),
            "away_goals": goals_info.get("away"),
            "home_ht_goals": halftime_info.get("home"),
            "away_ht_goals": halftime_info.get("away")
        })

    if not rows:
        raise HTTPException(
            status_code=502,
            detail="API-Football devolvió partidos sin identificadores válidos"
        )

    await supabase_upsert(
        "matches",
        rows,
        supabase_url,
        supabase_key
    )

    return {
        "ok": True,
        "message": "Liga, temporada, equipos y partidos sincronizados correctamente",
        "league": league,
        "season": season,
        "matches_received": len(fixtures_data),
        "matches_saved": len(rows)
    }


@app.get("/sync/statistics")
async def sync_statistics(
    fixture: int
):
    """
    Sincroniza las estadísticas de un partido concreto.

    Importante:
    API-Football cobra una solicitud por consulta.
    Por eso esta ruta trabaja con un solo fixture
    para poder controlar el límite gratuito.
    """

    data = await football_get(
        "/fixtures/statistics",
        {
            "fixture": fixture
        }
    )

    statistics_data = data.get("response", [])

    if not statistics_data:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron estadísticas para este partido"
        )

    supabase_url, supabase_key = get_supabase_config()

    rows = []

    for team_block in statistics_data:
        team_info = team_block.get("team", {})
        team_id = team_info.get("id")

        if team_id is None:
            continue

        statistics = team_block.get("statistics", [])

        for stat in statistics:
            stat_type = stat.get("type")
            value = stat.get("value")

            if not stat_type:
                continue

            value_numeric = None
            value_text = None

            if isinstance(value, (int, float)):
                value_numeric = float(value)
            elif isinstance(value, str):
                cleaned = value.replace("%", "").strip()

                try:
                    value_numeric = float(cleaned)
                except ValueError:
                    value_text = value
            elif value is not None:
                value_text = str(value)

            rows.append({
                "match_id": fixture,
                "team_id": team_id,
                "stat_type": stat_type,
                "value_numeric": value_numeric,
                "value_text": value_text
            })

    if not rows:
        raise HTTPException(
            status_code=502,
            detail="El partido no devolvió estadísticas utilizables"
        )

    await supabase_upsert(
        "match_statistics",
        rows,
        supabase_url,
        supabase_key
    )

    return {
        "ok": True,
        "message": "Estadísticas sincronizadas correctamente",
        "fixture": fixture,
        "statistics_saved": len(rows)
    }


@app.get("/sync/statistics/sample")
async def sync_statistics_sample(
    league: int,
    season: int
):
    """
    Busca un partido terminado de la competición y sincroniza
    sus estadísticas.

    Consume normalmente:
    1 solicitud para fixtures
    1 solicitud para statistics
    """

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
            detail="No se encontraron partidos"
        )

    selected_fixture = None

    for item in fixtures_data:
        fixture_info = item.get("fixture", {})
        status = fixture_info.get("status", {}).get("short")

        if status in {
            "FT",
            "AET",
            "PEN"
        }:
            selected_fixture = fixture_info.get("id")
            break

    if selected_fixture is None:
        raise HTTPException(
            status_code=404,
            detail="No se encontró un partido terminado"
        )

    result = await sync_statistics(
        fixture=selected_fixture
    )

    return {
        "ok": True,
        "message": "Partido de prueba sincronizado correctamente",
        "league": league,
        "season": season,
        "fixture": selected_fixture,
        "statistics_saved": result["statistics_saved"]
    }
