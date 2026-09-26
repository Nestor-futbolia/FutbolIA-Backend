import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx


BASE_URL = "https://v3.football.api-sports.io"

# ============================================================
# CONFIGURACIÓN
# ============================================================

UPCOMING_MATCH_LIMIT = 10
LAST_MATCHES_PER_TEAM = 10

# API-Football Free permite 10 solicitudes/minuto.
# 7 segundos entre llamadas deja margen de seguridad.
SECONDS_BETWEEN_API_CALLS = 7


API_FOOTBALL_KEY = os.getenv(
    "API_FOOTBALL_KEY",
    ""
).strip()

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    ""
).strip().rstrip("/")

SUPABASE_SECRET_KEY = os.getenv(
    "SUPABASE_SECRET_KEY",
    ""
).strip()


if not API_FOOTBALL_KEY:
    raise RuntimeError(
        "Falta API_FOOTBALL_KEY"
    )

if not SUPABASE_URL:
    raise RuntimeError(
        "Falta SUPABASE_URL"
    )

if not SUPABASE_SECRET_KEY:
    raise RuntimeError(
        "Falta SUPABASE_SECRET_KEY"
    )


SUPABASE_HEADERS = {
    "apikey": SUPABASE_SECRET_KEY,
    "Authorization": (
        f"Bearer {SUPABASE_SECRET_KEY}"
    ),
    "Content-Type": "application/json",
}


# ============================================================
# SUPABASE
# ============================================================

def supabase_url(table: str) -> str:
    return (
        f"{SUPABASE_URL}/rest/v1/{table}"
    )


def supabase_get(
    table: str,
    params: dict[str, Any]
):
    with httpx.Client(timeout=60.0) as client:
        response = client.get(
            supabase_url(table),
            headers=SUPABASE_HEADERS,
            params=params,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase GET {table} "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    if not response.text:
        return []

    return response.json()


def supabase_upsert(
    table: str,
    payload: dict[str, Any],
    on_conflict: str
):
    headers = dict(
        SUPABASE_HEADERS
    )

    headers["Prefer"] = (
        "resolution=merge-duplicates,"
        "return=representation"
    )

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            supabase_url(table),
            headers=headers,
            params={
                "on_conflict": on_conflict
            },
            json=payload,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase UPSERT {table} "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    if not response.text:
        return []

    return response.json()


# ============================================================
# API-FOOTBALL
# ============================================================

def football_get(
    endpoint: str,
    params: dict[str, Any]
):
    url = (
        endpoint
        if endpoint.startswith("http")
        else f"{BASE_URL}{endpoint}"
    )

    with httpx.Client(timeout=60.0) as client:
        response = client.get(
            url,
            headers={
                "x-apisports-key":
                    API_FOOTBALL_KEY
            },
            params=params,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            "API-Football HTTP "
            f"{response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    errors = data.get("errors")

    if errors:
        raise RuntimeError(
            "API-Football errors: "
            f"{errors}"
        )

    return data


# ============================================================
# UTILIDADES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_int(
    value,
    default=None
):
    try:
        if value is None:
            return default

        return int(value)

    except Exception:
        return default


def finished_status(
    status: str
) -> bool:
    return status in {
        "FT",
        "AET",
        "PEN",
    }


# ============================================================
# PRÓXIMOS PARTIDOS
# ============================================================

def load_upcoming_matches():
    now = datetime.now(
        timezone.utc
    ).isoformat()

    rows = supabase_get(
        "matches",
        {
            "select": (
                "id,"
                "starting_at,"
                "status,"
                "home_team_id,"
                "away_team_id"
            ),
            "starting_at": (
                f"gt.{now}"
            ),
            "order": "starting_at.asc",
            "limit": str(
                UPCOMING_MATCH_LIMIT
            ),
        },
    )

    return rows


# ============================================================
# GUARDAR FIXTURE
# ============================================================

def save_fixture(
    item: dict
):
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

    fixture_id = safe_int(
        fixture_info.get("id")
    )

    league_id = safe_int(
        league_info.get("id")
    )

    season_year = safe_int(
        league_info.get("season")
    )

    home = teams_info.get(
        "home",
        {}
    )

    away = teams_info.get(
        "away",
        {}
    )

    home_team_id = safe_int(
        home.get("id")
    )

    away_team_id = safe_int(
        away.get("id")
    )

    if (
        fixture_id is None
        or league_id is None
        or season_year is None
        or home_team_id is None
        or away_team_id is None
    ):
        return False

    # --------------------------------------------------------
    # LIGA
    # --------------------------------------------------------

    supabase_upsert(
        "leagues",
        {
            "id": league_id,
            "name": league_info.get(
                "name"
            ),
            "country": league_info.get(
                "country"
            ),
            "type": league_info.get(
                "type"
            ),
            "active": True,
        },
        "id",
    )

    # --------------------------------------------------------
    # TEMPORADA
    # --------------------------------------------------------

    season_id = (
        league_id * 10000
        + season_year
    )

    supabase_upsert(
        "seasons",
        {
            "id": season_id,
            "league_id": league_id,
            "name": str(
                season_year
            ),
        },
        "id",
    )

    # --------------------------------------------------------
    # EQUIPO LOCAL
    # --------------------------------------------------------

    supabase_upsert(
        "teams",
        {
            "id": home_team_id,
            "name": home.get(
                "name"
            ),
            "short_code": home.get(
                "code"
            ),
            "country": home.get(
                "country"
            ),
            "venue_name": None,
            "league_id": league_id,
        },
        "id",
    )

    # --------------------------------------------------------
    # EQUIPO VISITANTE
    # --------------------------------------------------------

    supabase_upsert(
        "teams",
        {
            "id": away_team_id,
            "name": away.get(
                "name"
            ),
            "short_code": away.get(
                "code"
            ),
            "country": away.get(
                "country"
            ),
            "venue_name": None,
            "league_id": league_id,
        },
        "id",
    )

    status_info = fixture_info.get(
        "status",
        {}
    )

    halftime = score_info.get(
        "halftime",
        {}
    )

    # --------------------------------------------------------
    # PARTIDO
    # --------------------------------------------------------

    supabase_upsert(
        "matches",
        {
            "id": fixture_id,
            "league_id": league_id,
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
            "updated_at": utc_now(),
        },
        "id",
    )

    return True


# ============================================================
# HISTORIAL DE UN EQUIPO
# ============================================================

def sync_team_history(
    team_id: int
):
    print("")
    print(
        "----------------------------------------"
    )
    print(
        f"Equipo {team_id}: "
        f"últimos {LAST_MATCHES_PER_TEAM} partidos"
    )

    data = football_get(
        "/fixtures",
        {
            "team": team_id,
            "last": LAST_MATCHES_PER_TEAM,
        },
    )

    response = data.get(
        "response",
        []
    )

    saved = 0
    finished = 0

    for item in response:
        fixture_info = item.get(
            "fixture",
            {}
        )

        status = (
            fixture_info
            .get("status", {})
            .get("short")
        )

        if not finished_status(
            status
        ):
            continue

        finished += 1

        try:
            if save_fixture(
                item
            ):
                saved += 1

        except Exception as exc:
            print(
                "Error guardando fixture "
                f"{fixture_info.get('id')}: "
                f"{exc}"
            )

    print(
        f"Partidos recibidos: "
        f"{len(response)}"
    )

    print(
        f"Terminados: {finished}"
    )

    print(
        f"Guardados/actualizados: "
        f"{saved}"
    )

    return {
        "team_id": team_id,
        "received": len(response),
        "finished": finished,
        "saved": saved,
    }


# ============================================================
# PRINCIPAL
# ============================================================

def main():
    print("")
    print(
        "========================================"
    )
    print(
        "FUTBOL IA - CARGA DE HISTORIAL"
    )
    print(
        "========================================"
    )

    upcoming = (
        load_upcoming_matches()
    )

    print(
        f"Próximos partidos encontrados: "
        f"{len(upcoming)}"
    )

    if not upcoming:
        print(
            "No hay próximos partidos "
            "guardados en Supabase."
        )
        return

    team_ids = []

    for match in upcoming:

        home_id = safe_int(
            match.get(
                "home_team_id"
            )
        )

        away_id = safe_int(
            match.get(
                "away_team_id"
            )
        )

        if (
            home_id is not None
            and home_id not in team_ids
        ):
            team_ids.append(
                home_id
            )

        if (
            away_id is not None
            and away_id not in team_ids
        ):
            team_ids.append(
                away_id
            )

    print(
        f"Equipos únicos: "
        f"{len(team_ids)}"
    )

    print(
        f"Solicitudes máximas a "
        f"API-Football: {len(team_ids)}"
    )

    results = []

    for index, team_id in enumerate(
        team_ids
    ):

        if index > 0:
            print(
                ""
            )
            print(
                f"Esperando "
                f"{SECONDS_BETWEEN_API_CALLS} "
                "segundos para respetar "
                "el límite de API-Football..."
            )

            time.sleep(
                SECONDS_BETWEEN_API_CALLS
            )

        try:
            result = (
                sync_team_history(
                    team_id
                )
            )

            results.append(
                result
            )

        except Exception as exc:

            print(
                f"ERROR equipo {team_id}: "
                f"{exc}"
            )

            results.append({
                "team_id": team_id,
                "error": str(exc),
            })

    print("")
    print(
        "========================================"
    )
    print(
        "HISTORIAL FINALIZADO"
    )
    print(
        "========================================"
    )

    total_saved = sum(
        item.get(
            "saved",
            0
        )
        for item in results
    )

    total_finished = sum(
        item.get(
            "finished",
            0
        )
        for item in results
    )

    print(
        f"Equipos procesados: "
        f"{len(results)}"
    )

    print(
        f"Partidos terminados encontrados: "
        f"{total_finished}"
    )

    print(
        f"Partidos guardados/actualizados: "
        f"{total_saved}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
