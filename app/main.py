import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException

from app.ai_predict import predict_match, load_active_model
from app.ai_evaluate import router as ai_evaluate_router


app = FastAPI(
    title="Fútbol IA 2.0 API",
    version="0.7.1"
)

app.include_router(ai_evaluate_router)

BASE_URL = "https://v3.football.api-sports.io"


# ============================================================
# CONFIGURACIÓN
# ============================================================

def get_api_football_key() -> str:
    key = os.getenv("API_FOOTBALL_KEY", "").strip()

    if not key:
        raise RuntimeError(
            "Falta la variable API_FOOTBALL_KEY"
        )

    return key


def get_supabase_config() -> tuple[str, str]:
    url = os.getenv(
        "SUPABASE_URL",
        ""
    ).strip().rstrip("/")

    key = os.getenv(
        "SUPABASE_SECRET_KEY",
        ""
    ).strip()

    if not url:
        raise RuntimeError(
            "Falta la variable SUPABASE_URL"
        )

    if not key:
        raise RuntimeError(
            "Falta la variable SUPABASE_SECRET_KEY"
        )

    return url, key


def supabase_headers(
    key: str,
    prefer: Optional[str] = None
) -> dict[str, str]:

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    if prefer:
        headers["Prefer"] = prefer

    return headers


# ============================================================
# API-FOOTBALL
# ============================================================

async def football_get(
    endpoint: str,
    params: Optional[dict[str, Any]] = None
) -> dict:

    api_key = get_api_football_key()

    url = (
        endpoint
        if endpoint.startswith("http")
        else f"{BASE_URL}{endpoint}"
    )

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(
            url,
            headers={
                "x-apisports-key": api_key
            },
            params=params or {}
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=(
                "API-Football respondió con "
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )
        )

    try:
        data = response.json()
    except Exception:
        raise HTTPException(
            status_code=502,
            detail=(
                "API-Football devolvió "
                "una respuesta no válida"
            )
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

async def supabase_request(
    method: str,
    table: str,
    params: Optional[dict[str, Any]] = None,
    payload: Optional[Any] = None,
    prefer: Optional[str] = None
) -> Any:

    supabase_url, supabase_key = (
        get_supabase_config()
    )

    url = f"{supabase_url}/rest/v1/{table}"

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.request(
            method,
            url,
            headers=supabase_headers(
                supabase_key,
                prefer
            ),
            params=params or {},
            json=payload
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=(
                f"Supabase {method} {table}: "
                f"{response.text}"
            )
        )

    if not response.text:
        return None

    try:
        return response.json()
    except Exception:
        return response.text


async def supabase_get(
    table: str,
    params: Optional[dict[str, Any]] = None
) -> list[dict]:

    result = await supabase_request(
        "GET",
        table,
        params=params
    )

    if isinstance(result, list):
        return result

    return []


async def supabase_upsert(
    table: str,
    payload: dict[str, Any],
    on_conflict: str
) -> Any:

    return await supabase_request(
        "POST",
        table,
        params={
            "on_conflict": on_conflict
        },
        payload=payload,
        prefer=(
            "resolution=merge-duplicates,"
            "return=representation"
        )
    )


# ============================================================
# UTILIDADES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_int(
    value: Any,
    default: Optional[int] = None
) -> Optional[int]:

    try:
        if value is None:
            return default

        return int(value)

    except Exception:
        return default


def clean_text(
    value: Any
) -> Optional[str]:

    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


# ============================================================
# PRINCIPAL
# ============================================================

@app.get("/")
async def root():

    return {
        "ok": True,
        "app": "Fútbol IA 2.0",
        "version": "0.7.1",
        "message": "Backend funcionando"
    }


@app.get("/health")
async def health():

    return {
        "ok": True,
        "status": "healthy",
        "app": "Fútbol IA 2.0",
        "version": "0.7.1",
        "time": utc_now()
    }


# ============================================================
# API-FOOTBALL
# ============================================================

@app.get("/countries")
async def countries():

    return await football_get(
        "/countries"
    )


@app.get("/leagues")
async def leagues(
    country: Optional[str] = None,
    season: Optional[int] = None
):

    params: dict[str, Any] = {}

    if country:
        params["country"] = country

    if season is not None:
        params["season"] = season

    return await football_get(
        "/leagues",
        params
    )


@app.get("/fixtures")
async def fixtures(
    league: Optional[int] = None,
    season: Optional[int] = None,
    team: Optional[int] = None,
    date: Optional[str] = None,
    next: Optional[int] = None,
    last: Optional[int] = None,
    status: Optional[str] = None
):

    params: dict[str, Any] = {}

    if league is not None:
        params["league"] = league

    if season is not None:
        params["season"] = season

    if team is not None:
        params["team"] = team

    if date:
        params["date"] = date

    if next is not None:
        params["next"] = next

    if last is not None:
        params["last"] = last

    if status:
        params["status"] = status

    return await football_get(
        "/fixtures",
        params
    )


@app.get("/fixtures/{fixture_id}")
async def fixture(
    fixture_id: int
):

    return await football_get(
        "/fixtures",
        {
            "id": fixture_id
        }
    )


@app.get("/teams/{team_id}")
async def team(
    team_id: int
):

    return await football_get(
        "/teams",
        {
            "id": team_id
        }
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

    params: dict[str, Any] = {}

    if league is not None:
        params["league"] = league

    if season is not None:
        params["season"] = season

    if fixture is not None:
        params["fixture"] = fixture

    if team is not None:
        params["team"] = team

    return await football_get(
        "/injuries",
        params
    )


@app.get("/odds")
async def odds(
    fixture: Optional[int] = None,
    league: Optional[int] = None,
    season: Optional[int] = None
):

    params: dict[str, Any] = {}

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

    response = data.get(
        "response",
        []
    )

    if not response:
        raise HTTPException(
            status_code=404,
            detail=(
                "No se encontró la liga "
                "o la temporada"
            )
        )

    item = response[0]

    league_info = item.get(
        "league",
        {}
    )

    country_info = item.get(
        "country",
        {}
    )

    seasons = item.get(
        "seasons",
        []
    )

    league_id = league_info.get(
        "id"
    )

    if league_id is None:
        raise HTTPException(
            status_code=502,
            detail=(
                "API-Football no devolvió "
                "el ID de la liga"
            )
        )

    league_payload = {
        "id": league_id,
        "name": league_info.get(
            "name"
        ),
        "country": country_info.get(
            "name"
        ),
        "type": league_info.get(
            "type"
        ),
        "active": True
    }

    await supabase_upsert(
        "leagues",
        league_payload,
        "id"
    )

    selected_season = None

    for season_info in seasons:

        if season_info.get(
            "year"
        ) == season:

            selected_season = (
                season_info
            )

            break

    if selected_season is None:
        selected_season = {
            "year": season
        }

    season_id = selected_season.get(
        "id"
    )

    if season_id is None:
        season_id = (
            league * 10000
            + season
        )

    season_payload = {
        "id": season_id,
        "league_id": league_id,
        "name": str(season),
        "starting_at": (
            selected_season.get(
                "start"
            )
        ),
        "ending_at": (
            selected_season.get(
                "end"
            )
        )
    }

    await supabase_upsert(
        "seasons",
        season_payload,
        "id"
    )

    return {
        "ok": True,
        "league": league_payload,
        "season": season_payload
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

    response = data.get(
        "response",
        []
    )

    saved = 0

    for item in response:

        team_info = item.get(
            "team",
            {}
        )

        venue_info = item.get(
            "venue",
            {}
        )

        team_id = team_info.get(
            "id"
        )

        if team_id is None:
            continue

        payload = {
            "id": team_id,
            "name": team_info.get(
                "name"
            ),
            "short_code": team_info.get(
                "code"
            ),
            "country": team_info.get(
                "country"
            ),
            "venue_name": venue_info.get(
                "name"
            ),
            "league_id": league
        }

        await supabase_upsert(
            "teams",
            payload,
            "id"
        )

        saved += 1

    return {
        "ok": True,
        "league": league,
        "season": season,
        "teams_received": len(
            response
        ),
        "teams_saved": saved
    }


# ============================================================
# SINCRONIZAR FIXTURES
# ============================================================

@app.get("/sync/fixtures")
async def sync_fixtures(
    league: int,
    season: int
):

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

    response = data.get(
        "response",
        []
    )

    if not response:
        raise HTTPException(
            status_code=404,
            detail=(
                "No se encontraron fixtures"
            )
        )

    saved = 0
    skipped = 0

    for item in response:

        fixture_info = item.get(
            "fixture",
            {}
        )

        league_info = item.get(
            "league",
            {}
        )

        teams_info = item.get(
            "teams",
            {}
        )

        goals_info = item.get(
            "goals",
            {}
        )

        score_info = item.get(
            "score",
            {}
        )

        fixture_id = fixture_info.get(
            "id"
        )

        home_team = teams_info.get(
            "home",
            {}
        )

        away_team = teams_info.get(
            "away",
            {}
        )

        home_team_id = home_team.get(
            "id"
        )

        away_team_id = away_team.get(
            "id"
        )

        if (
            fixture_id is None
            or home_team_id is None
            or away_team_id is None
        ):
            skipped += 1
            continue

        season_id = league_info.get(
            "season"
        )

        if season_id is None:
            season_id = (
                league * 10000
                + season
            )

        status_info = fixture_info.get(
            "status",
            {}
        )

        halftime = score_info.get(
            "halftime",
            {}
        )

        payload = {
            "id": fixture_id,
            "league_id": league,
            "season_id": season_id,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "referee_id": None,
            "starting_at": fixture_info.get(
                "date"
            ),
            "status": status_info.get(
                "short"
            ),
            "home_goals": goals_info.get(
                "home"
            ),
            "away_goals": goals_info.get(
                "away"
            ),
            "home_ht_goals": halftime.get(
                "home"
            ),
            "away_ht_goals": halftime.get(
                "away"
            ),
            "updated_at": utc_now()
        }

        await supabase_upsert(
            "matches",
            payload,
            "id"
        )

        saved += 1

    return {
        "ok": True,
        "league": league,
        "season": season,
        "fixtures_received": len(
            response
        ),
        "fixtures_saved": saved,
        "fixtures_skipped": skipped
    }


# ============================================================
# ESTADÍSTICAS DE UN PARTIDO
# ============================================================

@app.get("/sync/statistics")
async def sync_statistics(
    fixture: int
):

    data = await football_get(
        "/fixtures/statistics",
        {
            "fixture": fixture
        }
    )

    response = data.get(
        "response",
        []
    )

    if not response:
        raise HTTPException(
            status_code=404,
            detail=(
                "No se encontraron estadísticas "
                f"para el fixture {fixture}"
            )
        )

    saved = 0

    for team_block in response:

        team_info = team_block.get(
            "team",
            {}
        )

        team_id = team_info.get(
            "id"
        )

        if team_id is None:
            continue

        statistics = team_block.get(
            "statistics",
            []
        )

        for stat in statistics:

            stat_type = clean_text(
                stat.get("type")
            )

            value = stat.get(
                "value"
            )

            if not stat_type:
                continue

            numeric_value = None
            text_value = None

            if isinstance(
                value,
                (int, float)
            ):

                numeric_value = float(
                    value
                )

            elif isinstance(
                value,
                str
            ):

                cleaned = (
                    value
                    .replace(
                        "%",
                        ""
                    )
                    .replace(
                        ",",
                        "."
                    )
                    .strip()
                )

                try:
                    numeric_value = float(
                        cleaned
                    )

                except Exception:
                    text_value = value

            payload = {
                "match_id": fixture,
                "team_id": team_id,
                "stat_type": stat_type,
                "value_numeric": numeric_value,
                "value_text": text_value
            }

            await supabase_upsert(
                "match_statistics",
                payload,
                "match_id,team_id,stat_type"
            )

            saved += 1

    return {
        "ok": True,
        "fixture": fixture,
        "teams": len(
            response
        ),
        "statistics_saved": saved
    }


# ============================================================
# FIXTURES CON ESTADÍSTICAS
# ============================================================

async def get_saved_statistic_fixture_ids() -> set[int]:

    rows = await supabase_get(
        "match_statistics",
        {
            "select": "match_id",
            "limit": "10000"
        }
    )

    result: set[int] = set()

    for row in rows:

        match_id = safe_int(
            row.get(
                "match_id"
            )
        )

        if match_id is not None:
            result.add(
                match_id
            )

    return result


# ============================================================
# ESTADÍSTICAS POR LOTES
# ============================================================

@app.get("/sync/statistics/batch")
async def sync_statistics_batch(
    league: int,
    season: int,
    limit: int = 5
):

    if limit < 1 or limit > 10:
        raise HTTPException(
            status_code=400,
            detail=(
                "El límite debe estar "
                "entre 1 y 10"
            )
        )

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
            detail=(
                "No se encontraron partidos "
                "para esta liga y temporada"
            )
        )

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

        if status_short in finished_statuses:

            fixture_id = fixture_info.get(
                "id"
            )

            if fixture_id is not None:

                finished_fixtures.append({
                    "id": fixture_id,
                    "status": status_short,
                    "date": fixture_info.get(
                        "date"
                    )
                })

    if not finished_fixtures:
        raise HTTPException(
            status_code=404,
            detail=(
                "No se encontraron partidos "
                "terminados"
            )
        )

    saved_fixture_ids = (
        await get_saved_statistic_fixture_ids()
    )

    pending_fixtures = [
        item
        for item in finished_fixtures
        if item["id"]
        not in saved_fixture_ids
    ]

    selected_fixtures = (
        pending_fixtures[:limit]
    )

    processed = []
    failed = []

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


# ============================================================
# IA — ESTADO DEL MODELO
# ============================================================

@app.get("/ai/status")
async def ai_status():

    try:

        model = await load_active_model()

        metrics = model.get(
            "metrics",
            {}
        )

        return {
            "ok": True,
            "active": True,
            "model_version": model.get(
                "version"
            ),
            "model_name": model.get(
                "model_name"
            ),
            "trained_at": model.get(
                "trained_at"
            ),
            "training_matches": model.get(
                "training_matches"
            ),
            "validation_accuracy": metrics.get(
                "validation_accuracy"
            ),
            "validation_log_loss": metrics.get(
                "validation_log_loss"
            )
        }

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )


# ============================================================
# IA — PREDICCIÓN 1X2
# ============================================================

@app.get("/ai/predict/{match_id}")
async def ai_predict(
    match_id: int
):

    try:

        result = await predict_match(
            match_id
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )
