import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CONFIGURACION
# ============================================================

API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Futbol IA Backend",
    version="0.7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# VALIDACION DE CONFIGURACION
# ============================================================

def require_api_football_key() -> str:
    if not API_FOOTBALL_KEY:
        raise HTTPException(
            status_code=500,
            detail="Falta API_FOOTBALL_KEY"
        )

    return API_FOOTBALL_KEY


def get_supabase_config():
    if not SUPABASE_URL:
        raise HTTPException(
            status_code=500,
            detail="Falta SUPABASE_URL"
        )

    if not SUPABASE_SECRET_KEY:
        raise HTTPException(
            status_code=500,
            detail="Falta SUPABASE_SECRET_KEY"
        )

    return SUPABASE_URL.rstrip("/"), SUPABASE_SECRET_KEY


# ============================================================
# API-FOOTBALL
# ============================================================

async def football_get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    key = require_api_football_key()

    headers = {
        "x-apisports-key": key
    }

    url = f"{API_FOOTBALL_BASE_URL}{endpoint}"

    async with httpx.AsyncClient(timeout=60.0) as client:

        response = await client.get(
            url,
            headers=headers,
            params=params or {}
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail={
                "source": "API-Football",
                "status_code": response.status_code,
                "response": response.text
            }
        )

    try:
        data = response.json()
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="API-Football devolvió una respuesta no válida"
        )

    errors = data.get("errors")

    if errors:
        raise HTTPException(
            status_code=502,
            detail={
                "source": "API-Football",
                "errors": errors
            }
        )

    return data


# ============================================================
# SUPABASE
# ============================================================

def supabase_headers() -> Dict[str, str]:

    _, key = get_supabase_config()

    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }


async def supabase_get(
    table: str,
    params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:

    base_url, _ = get_supabase_config()

    url = f"{base_url}/rest/v1/{table}"

    async with httpx.AsyncClient(timeout=60.0) as client:

        response = await client.get(
            url,
            headers=supabase_headers(),
            params=params or {}
        )

    if response.status_code not in (200, 206):
        raise HTTPException(
            status_code=502,
            detail={
                "source": "Supabase",
                "operation": "GET",
                "table": table,
                "status_code": response.status_code,
                "response": response.text
            }
        )

    try:
        return response.json()
    except Exception:
        return []


async def supabase_upsert(
    table: str,
    rows: Any,
    on_conflict: Optional[str] = None
) -> List[Dict[str, Any]]:

    base_url, _ = get_supabase_config()

    url = f"{base_url}/rest/v1/{table}"

    headers = supabase_headers()

    headers["Prefer"] = "resolution=merge-duplicates,return=representation"

    params = {}

    if on_conflict:
        params["on_conflict"] = on_conflict

    async with httpx.AsyncClient(timeout=60.0) as client:

        response = await client.post(
            url,
            headers=headers,
            params=params,
            json=rows
        )

    if response.status_code not in (200, 201):
        raise HTTPException(
            status_code=502,
            detail={
                "source": "Supabase",
                "operation": "UPSERT",
                "table": table,
                "status_code": response.status_code,
                "response": response.text
            }
        )

    try:
        return response.json()
    except Exception:
        return []


# ============================================================
# UTILIDADES
# ============================================================

def clean_number(value: Any) -> Optional[float]:

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    if not text:
        return None

    text = text.replace("%", "")
    text = text.replace(",", ".")

    try:
        return float(text)
    except Exception:
        return None


def api_stat_to_values(value: Any):
    numeric = clean_number(value)

    if numeric is not None:
        return numeric, str(value)

    if value is None:
        return None, None

    return None, str(value)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "app": "Futbol IA Backend",
        "version": "0.7.0",
        "status": "online"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "ok": True
    }


# ============================================================
# API-FOOTBALL: COUNTRIES
# ============================================================

@app.get("/countries")
async def countries():

    return await football_get(
        "/countries"
    )


# ============================================================
# API-FOOTBALL: LEAGUES
# ============================================================

@app.get("/leagues")
async def leagues(
    country: Optional[str] = None,
    season: Optional[int] = None
):

    params = {}

    if country:
        params["country"] = country

    if season:
        params["season"] = season

    return await football_get(
        "/leagues",
        params
    )


# ============================================================
# API-FOOTBALL: FIXTURES
# ============================================================

@app.get("/fixtures")
async def fixtures(
    league: Optional[int] = None,
    season: Optional[int] = None,
    date: Optional[str] = None,
    team: Optional[int] = None,
    status: Optional[str] = None
):

    params = {}

    if league is not None:
        params["league"] = league

    if season is not None:
        params["season"] = season

    if date:
        params["date"] = date

    if team is not None:
        params["team"] = team

    if status:
        params["status"] = status

    return await football_get(
        "/fixtures",
        params
    )


# ============================================================
# API-FOOTBALL: FIXTURE DETAIL
# ============================================================

@app.get("/fixtures/{fixture_id}")
async def fixture_detail(
    fixture_id: int
):

    return await football_get(
        "/fixtures",
        {
            "id": fixture_id
        }
    )


# ============================================================
# API-FOOTBALL: TEAM
# ============================================================

@app.get("/teams/{team_id}")
async def team_detail(
    team_id: int
):

    return await football_get(
        "/teams",
        {
            "id": team_id
        }
    )


# ============================================================
# API-FOOTBALL: STANDINGS
# ============================================================

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


# ============================================================
# API-FOOTBALL: INJURIES
# ============================================================

@app.get("/injuries")
async def injuries(
    fixture: Optional[int] = None,
    league: Optional[int] = None,
    season: Optional[int] = None,
    team: Optional[int] = None
):

    params = {}

    if fixture is not None:
        params["fixture"] = fixture

    if league is not None:
        params["league"] = league

    if season is not None:
        params["season"] = season

    if team is not None:
        params["team"] = team

    return await football_get(
        "/injuries",
        params
    )


# ============================================================
# API-FOOTBALL: ODDS
# ============================================================

@app.get("/odds")
async def odds(
    fixture: Optional[int] = None,
    league: Optional[int] = None,
    season: Optional[int] = None
):

    params = {}

    if fixture is not None:
        params["fixture"] = fixture

    if league is not None:
        params["league"] = league

    if season is not None:
        params["season"] = season

    return await football_get(
        "/odds",
        params
    )


# ============================================================
# SINCRONIZAR LIGA
# ============================================================

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

    response = data.get("response", [])

    if not response:
        raise HTTPException(
            status_code=404,
            detail="Liga no encontrada"
        )

    league_data = response[0]

    league_info = league_data.get("league", {})
    country_info = league_data.get("country", {})

    league_row = {
        "id": league_info.get("id"),
        "name": league_info.get("name"),
        "country": country_info.get("name"),
        "type": league_info.get("type"),
        "active": True
    }

    await supabase_upsert(
        "leagues",
        league_row,
        "id"
    )

    season_row = {
        "id": season,
        "league_id": league,
        "name": str(season)
    }

    await supabase_upsert(
        "seasons",
        season_row,
        "id"
    )

    return {
        "ok": True,
        "league": league_row,
        "season": season_row
    }


# ============================================================
# SINCRONIZAR EQUIPOS
# ============================================================

@app.get("/sync/teams")
async def sync_teams(
    league: int,
    season: int
):

    await sync_league(
        league=league,
        season=season
    )

    data = await football_get(
        "/teams",
        {
            "league": league,
            "season": season
        }
    )

    response = data.get("response", [])

    rows = []

    for item in response:

        team = item.get("team", {})
        venue = item.get("venue", {})

        team_id = team.get("id")

        if team_id is None:
            continue

        rows.append({
            "id": team_id,
            "name": team.get("name") or "Sin nombre",
            "short_code": team.get("code"),
            "country": team.get("country"),
            "venue_name": venue.get("name"),
            "league_id": league
        })

    if rows:

        await supabase_upsert(
            "teams",
            rows,
            "id"
        )

    return {
        "ok": True,
        "league": league,
        "season": season,
        "teams": len(rows)
    }


# ============================================================
# SINCRONIZAR FIXTURES
# ============================================================

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

    response = data.get("response", [])

    rows = []

    for item in response:

        fixture = item.get("fixture", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        score = item.get("score", {})

        fixture_id = fixture.get("id")

        home = teams.get("home", {})
        away = teams.get("away", {})

        home_id = home.get("id")
        away_id = away.get("id")

        if fixture_id is None:
            continue

        if home_id is None or away_id is None:
            continue

        halftime = score.get("halftime", {})

        rows.append({
            "id": fixture_id,
            "league_id": league,
            "season_id": season,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "starting_at": fixture.get("date"),
            "status": fixture.get("status", {}).get("short"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "home_ht_goals": halftime.get("home"),
            "away_ht_goals": halftime.get("away")
        })

    if rows:

        await supabase_upsert(
            "matches",
            rows,
            "id"
        )

    return {
        "ok": True,
        "league": league,
        "season": season,
        "fixtures": len(rows)
    }


# ============================================================
# SINCRONIZAR ESTADISTICAS DE UN PARTIDO
# ============================================================

@app.get("/sync/statistics")
async def sync_statistics(
    fixture: int
):

    # --------------------------------------------------------
    # Obtener información del partido
    # --------------------------------------------------------

    fixture_data = await football_get(
        "/fixtures",
        {
            "id": fixture
        }
    )

    fixture_response = fixture_data.get("response", [])

    if not fixture_response:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró el fixture {fixture}"
        )

    fixture_item = fixture_response[0]

    fixture_info = fixture_item.get("fixture", {})
    league_info = fixture_item.get("league", {})
    teams_info = fixture_item.get("teams", {})
    goals_info = fixture_item.get("goals", {})
    score_info = fixture_item.get("score", {})

    league_id = league_info.get("id")
    season = league_info.get("season")

    home_team = teams_info.get("home", {})
    away_team = teams_info.get("away", {})

    home_team_id = home_team.get("id")
    away_team_id = away_team.get("id")

    if not league_id or not season:
        raise HTTPException(
            status_code=502,
            detail="El fixture no contiene liga o temporada"
        )

    if not home_team_id or not away_team_id:
        raise HTTPException(
            status_code=502,
            detail="El fixture no contiene los equipos"
        )

    # --------------------------------------------------------
    # Asegurar liga
    # --------------------------------------------------------

    league_row = {
        "id": league_id,
        "name": league_info.get("name"),
        "country": None,
        "type": league_info.get("type"),
        "active": True
    }

    await supabase_upsert(
        "leagues",
        league_row,
        "id"
    )

    # --------------------------------------------------------
    # Asegurar temporada
    # --------------------------------------------------------

    season_row = {
        "id": season,
        "league_id": league_id,
        "name": str(season)
    }

    await supabase_upsert(
        "seasons",
        season_row,
        "id"
    )

    # --------------------------------------------------------
    # Asegurar equipos
    # --------------------------------------------------------

    teams_rows = [
        {
            "id": home_team_id,
            "name": home_team.get("name") or "Local",
            "short_code": home_team.get("code"),
            "country": None,
            "venue_name": None,
            "league_id": league_id
        },
        {
            "id": away_team_id,
            "name": away_team.get("name") or "Visitante",
            "short_code": away_team.get("code"),
            "country": None,
            "venue_name": None,
            "league_id": league_id
        }
    ]

    await supabase_upsert(
        "teams",
        teams_rows,
        "id"
    )

    # --------------------------------------------------------
    # Asegurar partido
    # --------------------------------------------------------

    halftime = score_info.get("halftime", {})

    match_row = {
        "id": fixture,
        "league_id": league_id,
        "season_id": season,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "starting_at": fixture_info.get("date"),
        "status": fixture_info.get("status", {}).get("short"),
        "home_goals": goals_info.get("home"),
        "away_goals": goals_info.get("away"),
        "home_ht_goals": halftime.get("home"),
        "away_ht_goals": halftime.get("away")
    }

    await supabase_upsert(
        "matches",
        match_row,
        "id"
    )

    # --------------------------------------------------------
    # Obtener estadísticas
    # --------------------------------------------------------

    statistics_data = await football_get(
        "/fixtures/statistics",
        {
            "fixture": fixture
        }
    )

    statistics_response = statistics_data.get("response", [])

    if not statistics_response:
        return {
            "ok": True,
            "fixture": fixture,
            "statistics": 0,
            "message": "API-Football no devolvió estadísticas"
        }

    rows = []

    for team_block in statistics_response:

        team = team_block.get("team", {})
        team_id = team.get("id")

        if not team_id:
            continue

        statistics = team_block.get("statistics", [])

        for stat in statistics:

            stat_type = stat.get("type")
            value = stat.get("value")

            if not stat_type:
                continue

            numeric_value, text_value = api_stat_to_values(
                value
            )

            rows.append({
                "match_id": fixture,
                "team_id": team_id,
                "stat_type": stat_type,
                "value_numeric": numeric_value,
                "value_text": text_value
            })

    if rows:

        await supabase_upsert(
            "match_statistics",
            rows,
            "match_id,team_id,stat_type"
        )

    return {
        "ok": True,
        "fixture": fixture,
        "statistics": len(rows)
    }


# ============================================================
# ESTADISTICAS DE MUESTRA
# ============================================================

@app.get("/sync/statistics/sample")
async def sync_statistics_sample(
    league: int,
    season: int,
    limit: int = 1
):

    if limit < 1 or limit > 5:
        raise HTTPException(
            status_code=400,
            detail="El límite debe estar entre 1 y 5"
        )

    data = await football_get(
        "/fixtures",
        {
            "league": league,
            "season": season
        }
    )

    fixtures_data = data.get("response", [])

    finished_statuses = {
        "FT",
        "AET",
        "PEN"
    }

    selected = []

    for item in fixtures_data:

        fixture_info = item.get("fixture", {})
        status = fixture_info.get("status", {}).get("short")
        fixture_id = fixture_info.get("id")

        if (
            status in finished_statuses
            and fixture_id is not None
        ):
            selected.append(fixture_id)

        if len(selected) >= limit:
            break

    if not selected:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron partidos terminados"
        )

    processed = []
    failed = []

    for fixture_id in selected:

        try:

            result = await sync_statistics(
                fixture=fixture_id
            )

            processed.append({
                "fixture": fixture_id,
                "status": "ok",
                "result": result
            })

        except Exception as exc:

            failed.append({
                "fixture": fixture_id,
                "status": "error",
                "detail": str(exc)
            })

    return {
        "ok": True,
        "league": league,
        "season": season,
        "selected": selected,
        "processed": processed,
        "failed": failed
    }


# ============================================================
# OBTENER FIXTURES QUE YA TIENEN ESTADISTICAS
# ============================================================

async def get_saved_statistic_fixture_ids(
    supabase_url: str,
    supabase_key: str
) -> set:

    url = f"{supabase_url}/rest/v1/match_statistics"

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}"
    }

    params = {
        "select": "match_id",
        "limit": "10000"
    }

    async with httpx.AsyncClient(timeout=60.0) as client:

        response = await client.get(
            url,
            headers=headers,
            params=params
        )

    if response.status_code not in (200, 206):

        raise HTTPException(
            status_code=502,
            detail={
                "source": "Supabase",
                "operation": "GET match_statistics",
                "status_code": response.status_code,
                "response": response.text
            }
        )

    try:
        data = response.json()
    except Exception:
        data = []

    return {
        int(row["match_id"])
        for row in data
        if row.get("match_id") is not None
    }


# ============================================================
# SINCRONIZACION DE ESTADISTICAS POR LOTE
# ============================================================

@app.get("/sync/statistics/batch")
async def sync_statistics_batch(
    league: int,
    season: int,
    limit: int = 5
):

    # --------------------------------------------------------
    # Validar limite
    # --------------------------------------------------------

    if limit < 1 or limit > 10:

        raise HTTPException(
            status_code=400,
            detail="El límite debe estar entre 1 y 10"
        )

    # --------------------------------------------------------
    # Obtener partidos de la liga y temporada
    # --------------------------------------------------------

    data = await football_get(
        "/fixtures",
        {
            "league": league,
            "season": season
        }
    )

    fixtures_data = data.get(
        "response",
        []
    )

    if not fixtures_data:

        raise HTTPException(
            status_code=404,
            detail="No se encontraron partidos para esta liga y temporada"
        )

    # --------------------------------------------------------
    # Filtrar partidos terminados
    # --------------------------------------------------------

    finished_statuses = {
        "FT",
        "AET",
        "PEN"
    }

    finished_fixtures = []

    for item in fixtures_data:

        fixture_info = item.get(
            "fixture",
            {}
        )

        status_info = fixture_info.get(
            "status",
            {}
        )

        status_short = status_info.get(
            "short"
        )

        fixture_id = fixture_info.get(
            "id"
        )

        if (
            status_short in finished_statuses
            and fixture_id is not None
        ):

            finished_fixtures.append({
                "id": fixture_id,
                "status": status_short,
                "date": fixture_info.get("date")
            })

    if not finished_fixtures:

        raise HTTPException(
            status_code=404,
            detail="No se encontraron partidos terminados"
        )

    # --------------------------------------------------------
    # Consultar partidos que ya tienen estadísticas
    # --------------------------------------------------------

    supabase_url, supabase_key = get_supabase_config()

    saved_fixture_ids = await get_saved_statistic_fixture_ids(
        supabase_url,
        supabase_key
    )

    # --------------------------------------------------------
    # Dejar solamente partidos pendientes
    # --------------------------------------------------------

    pending_fixtures = [
        item
        for item in finished_fixtures
        if item["id"] not in saved_fixture_ids
    ]

    # --------------------------------------------------------
    # Aplicar limite
    # --------------------------------------------------------

    selected_fixtures = pending_fixtures[:limit]

    processed = []
    failed = []

    # --------------------------------------------------------
    # Sincronizar uno por uno
    # --------------------------------------------------------

    for item in selected_fixtures:

        fixture_id = item["id"]

        try:

            result = await sync_statistics(
                fixture=fixture_id
            )

            processed.append({
                "fixture": fixture_id,
                "status": "ok",
                "result": result
            })

        except HTTPException as exc:

            failed.append({
                "fixture": fixture_id,
                "status": "error",
                "http_status": exc.status_code,
                "detail": exc.detail
            })

        except Exception as exc:

            failed.append({
                "fixture": fixture_id,
                "status": "error",
                "detail": str(exc)
            })

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    return {
        "ok": True,
        "league": league,
        "season": season,
        "limit": limit,
        "existing_statistics_fixtures": len(
            saved_fixture_ids
        ),
        "finished_fixtures": len(
            finished_fixtures
        ),
        "pending_fixtures": len(
            pending_fixtures
        ),
        "selected": len(
            selected_fixtures
        ),
        "processed": len(
            processed
        ),
        "failed": len(
            failed
        ),
        "details": {
            "processed": processed,
            "failed": failed
        }
    }
