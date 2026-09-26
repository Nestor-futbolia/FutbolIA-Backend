import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_URL = "https://v3.football.api-sports.io"

# API-Football Free permite trabajar con esta temporada.
HISTORICAL_SEASON = 2024

# Número máximo de partidos históricos que queremos conservar
# por cada equipo para esta primera carga.
LAST_MATCHES_PER_TEAM = 10

# API-Football Free tiene límite de 10 solicitudes por minuto.
# 7 segundos entre solicitudes mantiene un margen de seguridad.
SECONDS_BETWEEN_API_CALLS = 7

# Estados que consideramos partidos terminados.
FINISHED_STATUSES = {
    "FT",
    "AET",
    "PEN",
}


# ============================================================
# CONFIGURACIÓN DE VARIABLES DE ENTORNO
# ============================================================

def get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Falta la variable de entorno obligatoria: {name}"
        )

    return value


def get_api_football_key() -> str:
    return get_required_env("API_FOOTBALL_KEY")


def get_supabase_config() -> tuple[str, str]:
    url = get_required_env("SUPABASE_URL").rstrip("/")
    key = get_required_env("SUPABASE_SECRET_KEY")

    return url, key


# ============================================================
# UTILIDADES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def is_finished_status(status: Any) -> bool:
    value = clean_text(status)

    if not value:
        return False

    return value.upper() in FINISHED_STATUSES


# ============================================================
# API-FOOTBALL
# ============================================================

def football_get(
    endpoint: str,
    params: Optional[dict[str, Any]] = None
) -> dict:

    api_key = get_api_football_key()

    url = (
        endpoint
        if endpoint.startswith("http")
        else f"{BASE_URL}{endpoint}"
    )

    print("")
    print("API-Football:")
    print(f"  Endpoint: {endpoint}")
    print(f"  Parámetros: {params}")

    try:
        with httpx.Client(timeout=60) as client:
            response = client.get(
                url,
                headers={
                    "x-apisports-key": api_key
                },
                params=params or {}
            )

    except Exception as exc:
        raise RuntimeError(
            f"Error de conexión con API-Football: {exc}"
        ) from exc

    print(f"  HTTP: {response.status_code}")

    if response.status_code >= 400:
        raise RuntimeError(
            "API-Football respondió con "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    try:
        data = response.json()

    except Exception as exc:
        raise RuntimeError(
            "API-Football devolvió una respuesta "
            "que no es JSON válido."
        ) from exc

    errors = data.get("errors")

    if errors:
        raise RuntimeError(
            f"API-Football errors: {errors}"
        )

    return data


# ============================================================
# SUPABASE
# ============================================================

def supabase_headers(
    key: str,
    prefer: Optional[str] = None
) -> dict[str, str]:

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    if prefer:
        headers["Prefer"] = prefer

    return headers


def supabase_request(
    method: str,
    table: str,
    params: Optional[dict[str, Any]] = None,
    payload: Optional[Any] = None,
    prefer: Optional[str] = None
) -> Any:

    supabase_url, supabase_key = get_supabase_config()

    url = f"{supabase_url}/rest/v1/{table}"

    try:
        with httpx.Client(timeout=60) as client:
            response = client.request(
                method,
                url,
                headers=supabase_headers(
                    supabase_key,
                    prefer
                ),
                params=params or {},
                json=payload,
            )

    except Exception as exc:
        raise RuntimeError(
            f"Error de conexión con Supabase "
            f"en {table}: {exc}"
        ) from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase {method} {table}: "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    if not response.text:
        return None

    try:
        return response.json()

    except Exception:
        return response.text


def supabase_get(
    table: str,
    params: Optional[dict[str, Any]] = None
) -> list[dict]:

    result = supabase_request(
        "GET",
        table,
        params=params,
    )

    if isinstance(result, list):
        return result

    return []


def supabase_upsert(
    table: str,
    payload: dict[str, Any],
    on_conflict: str,
) -> Any:

    return supabase_request(
        "POST",
        table,
        params={
            "on_conflict": on_conflict
        },
        payload=payload,
        prefer=(
            "resolution=merge-duplicates,"
            "return=representation"
        ),
    )


# ============================================================
# GUARDAR LIGA
# ============================================================

def save_league(
    league: dict[str, Any]
) -> Optional[int]:

    league_id = safe_int(
        league.get("id")
    )

    if league_id is None:
        return None

    payload = {
        "id": league_id,
        "name": clean_text(
            league.get("name")
        ) or f"League {league_id}",
        "country": clean_text(
            league.get("country")
        ),
        "type": clean_text(
            league.get("type")
        ),
        "active": True,
    }

    supabase_upsert(
        "leagues",
        payload,
        "id",
    )

    return league_id


# ============================================================
# GUARDAR TEMPORADA
# ============================================================

def build_season_id(
    league_id: int,
    season_year: int
) -> int:

    # ID determinista para relacionar correctamente
    # la temporada con la liga.
    return (
        league_id * 10000
        + season_year
    )


def save_season(
    league_id: int,
    season_year: int
) -> int:

    season_id = build_season_id(
        league_id,
        season_year
    )

    payload = {
        "id": season_id,
        "league_id": league_id,
        "name": str(season_year),
        "starting_at": None,
        "ending_at": None,
    }

    supabase_upsert(
        "seasons",
        payload,
        "id",
    )

    return season_id


# ============================================================
# GUARDAR EQUIPO
# ============================================================

def save_team(
    team: dict[str, Any],
    league_id: Optional[int]
) -> Optional[int]:

    team_id = safe_int(
        team.get("id")
    )

    if team_id is None:
        return None

    venue = team.get("venue") or {}

    payload = {
        "id": team_id,
        "name": clean_text(
            team.get("name")
        ) or f"Team {team_id}",
        "short_code": clean_text(
            team.get("code")
        ),
        "country": clean_text(
            team.get("country")
        ),
        "venue_name": clean_text(
            venue.get("name")
        ),
        "league_id": league_id,
    }

    supabase_upsert(
        "teams",
        payload,
        "id",
    )

    return team_id


# ============================================================
# GUARDAR PARTIDO
# ============================================================

def save_match(
    fixture: dict[str, Any]
) -> Optional[int]:

    fixture_id = safe_int(
        fixture.get("id")
    )

    if fixture_id is None:
        return None

    league = fixture.get("league") or {}
    teams = fixture.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    goals = fixture.get("goals") or {}
    score = fixture.get("score") or {}
    halftime = score.get("halftime") or {}

    league_id = safe_int(
        league.get("id")
    )

    season_year = safe_int(
        league.get("season"),
        HISTORICAL_SEASON
    )

    home_team_id = safe_int(
        home.get("id")
    )

    away_team_id = safe_int(
        away.get("id")
    )

    if league_id is None:
        print(
            f"  ⚠️ Partido {fixture_id}: "
            "sin league_id."
        )
        return None

    if season_year is None:
        season_year = HISTORICAL_SEASON

    season_id = build_season_id(
        league_id,
        season_year
    )

    starting_at = clean_text(
        fixture.get("date")
    )

    status_data = fixture.get("status") or {}

    status = clean_text(
        status_data.get("short")
    )

    home_goals = safe_int(
        goals.get("home")
    )

    away_goals = safe_int(
        goals.get("away")
    )

    home_ht_goals = safe_int(
        halftime.get("home")
    )

    away_ht_goals = safe_int(
        halftime.get("away")
    )

    payload = {
        "id": fixture_id,
        "league_id": league_id,
        "season_id": season_id,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "referee_id": None,
        "starting_at": starting_at,
        "status": status,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_ht_goals": home_ht_goals,
        "away_ht_goals": away_ht_goals,
    }

    supabase_upsert(
        "matches",
        payload,
        "id",
    )

    return fixture_id


# ============================================================
# CARGAR PARTIDOS PRÓXIMOS DESDE SUPABASE
# ============================================================

def load_upcoming_matches(
    limit: int = 10
) -> list[dict]:

    print("")
    print("========================================")
    print("BUSCANDO PRÓXIMOS PARTIDOS")
    print("========================================")

    # Primero intentamos con estado NS, que es el estado
    # utilizado por API-Football para partidos programados.
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
            "status": "eq.NS",
            "order": "starting_at.asc",
            "limit": str(limit),
        },
    )

    # Si no hay NS, buscamos cualquier partido futuro.
    if not rows:
        now = utc_now()

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
                "starting_at": f"gte.{now}",
                "order": "starting_at.asc",
                "limit": str(limit),
            },
        )

    print(
        f"Próximos partidos encontrados: "
        f"{len(rows)}"
    )

    for row in rows:
        print(
            f"  - Match {row.get('id')} | "
            f"Home {row.get('home_team_id')} | "
            f"Away {row.get('away_team_id')} | "
            f"{row.get('starting_at')}"
        )

    return rows[:limit]


# ============================================================
# OBTENER HISTORIAL DE UN EQUIPO
# ============================================================

def get_team_historical_matches(
    team_id: int
) -> list[dict]:

    print("")
    print("----------------------------------------")
    print(
        f"HISTORIAL DEL EQUIPO {team_id}"
    )
    print("----------------------------------------")

    # IMPORTANTE:
    #
    # NO usamos:
    #
    #     last=10
    #
    # porque API-Football Free bloquea el parámetro
    # "last".
    #
    # En su lugar pedimos la temporada completa y
    # seleccionamos nosotros los últimos 10 partidos.
    data = football_get(
        "/fixtures",
        {
            "team": team_id,
            "season": HISTORICAL_SEASON,
        },
    )

    response = data.get(
        "response",
        []
    )

    if not isinstance(response, list):
        response = []

    print(
        f"Partidos recibidos para equipo "
        f"{team_id}: {len(response)}"
    )

    finished = []

    for fixture in response:

        fixture_status = (
            (fixture.get("fixture") or {})
            .get("status") or {}
        )

        status = fixture_status.get(
            "short"
        )

        if not is_finished_status(status):
            continue

        fixture_id = safe_int(
            (fixture.get("fixture") or {}).get(
                "id"
            )
        )

        if fixture_id is None:
            continue

        fixture["_fixture_id"] = fixture_id

        finished.append(fixture)

    # Orden cronológico descendente:
    # el partido más reciente queda primero.
    finished.sort(
        key=lambda item: (
            clean_text(
                (item.get("fixture") or {}).get(
                    "date"
                )
            ) or ""
        ),
        reverse=True,
    )

    selected = finished[
        :LAST_MATCHES_PER_TEAM
    ]

    print(
        f"Partidos terminados: "
        f"{len(finished)}"
    )

    print(
        f"Partidos seleccionados: "
        f"{len(selected)}"
    )

    return selected


# ============================================================
# GUARDAR HISTORIAL DE UN EQUIPO
# ============================================================

def sync_team_history(
    team_id: int
) -> dict[str, int]:

    fixtures = get_team_historical_matches(
        team_id
    )

    saved = 0
    failed = 0

    for fixture in fixtures:

        fixture_id = safe_int(
            fixture.get("_fixture_id")
        )

        try:
            league = fixture.get(
                "league"
            ) or {}

            teams = fixture.get(
                "teams"
            ) or {}

            home = teams.get(
                "home"
            ) or {}

            away = teams.get(
                "away"
            ) or {}

            league_id = safe_int(
                league.get("id")
            )

            season_year = safe_int(
                league.get("season"),
                HISTORICAL_SEASON
            )

            if league_id is None:
                raise RuntimeError(
                    "El partido no tiene "
                    "league.id"
                )

            if season_year is None:
                season_year = HISTORICAL_SEASON

            # ------------------------------------------------
            # Liga
            # ------------------------------------------------

            save_league(
                league
            )

            # ------------------------------------------------
            # Temporada
            # ------------------------------------------------

            save_season(
                league_id,
                season_year
            )

            # ------------------------------------------------
            # Equipos
            # ------------------------------------------------

            save_team(
                home,
                league_id
            )

            save_team(
                away,
                league_id
            )

            # ------------------------------------------------
            # Partido
            # ------------------------------------------------

            saved_id = save_match(
                fixture
            )

            if saved_id is None:
                raise RuntimeError(
                    "No se pudo guardar "
                    "el partido."
                )

            saved += 1

            print(
                f"  ✅ Guardado partido "
                f"{saved_id}"
            )

        except Exception as exc:

            failed += 1

            print(
                f"  ❌ Error guardando "
                f"partido {fixture_id}: "
                f"{exc}"
            )

    return {
        "received": len(fixtures),
        "saved": saved,
        "failed": failed,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("========================================")
    print("FÚTBOL IA - CARGA DE HISTORIAL")
    print("========================================")
    print("")
    print(
        f"Hora UTC: {utc_now()}"
    )
    print(
        f"Temporada histórica: "
        f"{HISTORICAL_SEASON}"
    )
    print(
        f"Partidos por equipo: "
        f"{LAST_MATCHES_PER_TEAM}"
    )
    print(
        f"Espera entre llamadas API: "
        f"{SECONDS_BETWEEN_API_CALLS}s"
    )
    print("")
    print(
        "IMPORTANTE: se solicita la temporada "
        "completa porque API-Football Free "
        "NO permite el parámetro 'last'."
    )

    # ========================================================
    # VALIDAR CONFIGURACIÓN
    # ========================================================

    get_api_football_key()
    get_supabase_config()

    print("")
    print("Configuración validada correctamente.")

    # ========================================================
    # OBTENER PRÓXIMOS PARTIDOS
    # ========================================================

    upcoming = load_upcoming_matches(
        limit=10
    )

    if not upcoming:
        raise RuntimeError(
            "No se encontraron próximos partidos "
            "en Supabase. Ejecuta primero "
            "/sync/upcoming."
        )

    # ========================================================
    # OBTENER EQUIPOS ÚNICOS
    # ========================================================

    team_ids: list[int] = []

    for match in upcoming:

        home_team_id = safe_int(
            match.get("home_team_id")
        )

        away_team_id = safe_int(
            match.get("away_team_id")
        )

        if (
            home_team_id is not None
            and home_team_id not in team_ids
        ):
            team_ids.append(
                home_team_id
            )

        if (
            away_team_id is not None
            and away_team_id not in team_ids
        ):
            team_ids.append(
                away_team_id
            )

    if not team_ids:
        raise RuntimeError(
            "Los próximos partidos no tienen "
            "IDs de equipos válidos."
        )

    print("")
    print(
        f"Equipos únicos encontrados: "
        f"{len(team_ids)}"
    )

    for team_id in team_ids:
        print(
            f"  - Equipo {team_id}"
        )

    # ========================================================
    # CARGAR HISTORIAL
    # ========================================================

    total_received = 0
    total_saved = 0
    total_failed = 0

    team_errors: list[str] = []

    for index, team_id in enumerate(
        team_ids,
        start=1
    ):

        print("")
        print("========================================")
        print(
            f"EQUIPO {index}/{len(team_ids)}"
        )
        print(
            f"ID: {team_id}"
        )
        print("========================================")

        try:

            result = sync_team_history(
                team_id
            )

            total_received += result[
                "received"
            ]

            total_saved += result[
                "saved"
            ]

            total_failed += result[
                "failed"
            ]

            print("")
            print(
                f"Resultado equipo {team_id}: "
                f"recibidos={result['received']}, "
                f"guardados={result['saved']}, "
                f"errores={result['failed']}"
            )

            if result["failed"] > 0:
                team_errors.append(
                    f"Equipo {team_id}: "
                    f"{result['failed']} "
                    "partidos no pudieron guardarse."
                )

        except Exception as exc:

            message = (
                f"Equipo {team_id}: "
                f"{exc}"
            )

            team_errors.append(
                message
            )

            print("")
            print(
                f"❌ ERROR EN EQUIPO {team_id}"
            )
            print(
                str(exc)
            )

        # ----------------------------------------------------
        # Esperar antes de la siguiente llamada.
        #
        # Esto evita superar el límite de 10 requests/min.
        # ----------------------------------------------------

        if index < len(team_ids):

            print("")
            print(
                "Esperando "
                f"{SECONDS_BETWEEN_API_CALLS} "
                "segundos para respetar "
                "el límite de API-Football..."
            )

            time.sleep(
                SECONDS_BETWEEN_API_CALLS
            )

    # ========================================================
    # RESUMEN
    # ========================================================

    print("")
    print("")
    print("========================================")
    print("RESUMEN DE CARGA HISTÓRICA")
    print("========================================")
    print(
        f"Equipos procesados: "
        f"{len(team_ids)}"
    )
    print(
        f"Partidos recibidos: "
        f"{total_received}"
    )
    print(
        f"Partidos guardados: "
        f"{total_saved}"
    )
    print(
        f"Errores de guardado: "
        f"{total_failed}"
    )
    print(
        f"Equipos con errores: "
        f"{len(team_errors)}"
    )

    if team_errors:

        print("")
        print(
            "DETALLE DE ERRORES:"
        )

        for error in team_errors:
            print(
                f"- {error}"
            )

    # ========================================================
    # NO PERMITIR FALSO ÉXITO
    # ========================================================

    if total_saved == 0:

        raise RuntimeError(
            "La carga histórica terminó sin "
            "guardar ningún partido."
        )

    if team_errors:

        raise RuntimeError(
            "La carga histórica terminó con "
            f"{len(team_errors)} errores de equipo. "
            "Revisa los mensajes anteriores."
        )

    print("")
    print(
        "========================================"
    )
    print(
        "✅ HISTORIAL CARGADO CORRECTAMENTE"
    )
    print(
        "========================================"
    )
    print("")
    print(
        "Los partidos históricos ya están "
        "disponibles para el entrenamiento "
        "de Fútbol IA."
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:

        print("")
        print(
            "Proceso cancelado por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        print("")
        print("========================================")
        print("❌ ERROR FATAL")
        print("========================================")
        print(
            str(exc)
        )

        sys.exit(1)
