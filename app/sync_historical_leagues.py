import os
import sys
import time
from typing import Any, Optional

import httpx


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_URL = "https://v3.football.api-sports.io"

# El plan gratuito que estamos utilizando permite estas
# temporadas históricas en nuestra configuración actual.
SEASONS = [
    2022,
    2023,
    2024,
]

# Principales ligas europeas.
#
# 39  = Premier League
# 140 = La Liga
# 135 = Serie A
# 78  = Bundesliga
# 61  = Ligue 1
#
# API-Football utiliza estos IDs para las competiciones.
LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}

# Para respetar el límite de 10 solicitudes/minuto.
SECONDS_BETWEEN_API_CALLS = 7

# Cantidad de filas enviadas a Supabase por lote.
SUPABASE_BATCH_SIZE = 300


# ============================================================
# VARIABLES DE ENTORNO
# ============================================================

def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Falta la variable de entorno: {name}"
        )

    return value


def get_api_key() -> str:
    return required_env(
        "API_FOOTBALL_KEY"
    )


def get_supabase_config() -> tuple[str, str]:
    url = required_env(
        "SUPABASE_URL"
    ).rstrip("/")

    key = required_env(
        "SUPABASE_SECRET_KEY"
    )

    return url, key


# ============================================================
# UTILIDADES
# ============================================================

def safe_int(
    value: Any,
    default: Optional[int] = None,
) -> Optional[int]:

    try:

        if value is None:
            return default

        return int(value)

    except (
        TypeError,
        ValueError,
    ):

        return default


def clean_text(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def chunk_list(
    items: list[dict],
    size: int,
):

    for start in range(
        0,
        len(items),
        size,
    ):

        yield items[
            start:start + size
        ]


# ============================================================
# API-FOOTBALL
# ============================================================

def football_get(
    endpoint: str,
    params: dict[str, Any],
) -> dict[str, Any]:

    api_key = get_api_key()

    url = (
        f"{BASE_URL}{endpoint}"
    )

    print("")
    print("API-Football")
    print(
        f"  GET {endpoint}"
    )
    print(
        f"  Parámetros: {params}"
    )

    try:

        with httpx.Client(
            timeout=90
        ) as client:

            response = client.get(
                url,
                headers={
                    "x-apisports-key": api_key,
                },
                params=params,
            )

    except Exception as exc:

        raise RuntimeError(
            "No se pudo conectar con "
            f"API-Football: {exc}"
        ) from exc

    print(
        f"  HTTP: {response.status_code}"
    )

    # --------------------------------------------------------
    # Control de límite
    # --------------------------------------------------------

    daily_remaining = response.headers.get(
        "x-ratelimit-requests-remaining"
    )

    minute_remaining = response.headers.get(
        "X-RateLimit-Remaining"
    )

    if daily_remaining is not None:

        print(
            "  Solicitudes diarias restantes: "
            f"{daily_remaining}"
        )

    if minute_remaining is not None:

        print(
            "  Solicitudes por minuto restantes: "
            f"{minute_remaining}"
        )

    # --------------------------------------------------------
    # HTTP error
    # --------------------------------------------------------

    if response.status_code >= 400:

        raise RuntimeError(
            "API-Football respondió con "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:

        data = response.json()

    except Exception as exc:

        raise RuntimeError(
            "API-Football devolvió "
            "JSON inválido."
        ) from exc

    # --------------------------------------------------------
    # API errors
    # --------------------------------------------------------

    errors = data.get(
        "errors"
    )

    if errors:

        raise RuntimeError(
            f"API-Football errors: "
            f"{errors}"
        )

    return data


# ============================================================
# SUPABASE
# ============================================================

def supabase_headers() -> dict[str, str]:

    _, key = get_supabase_config()

    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def supabase_upsert_batch(
    table: str,
    rows: list[dict[str, Any]],
    on_conflict: str,
) -> int:

    if not rows:
        return 0

    supabase_url, _ = (
        get_supabase_config()
    )

    url = (
        f"{supabase_url}"
        f"/rest/v1/{table}"
    )

    headers = supabase_headers()

    headers["Prefer"] = (
        "resolution=merge-duplicates,"
        "return=minimal"
    )

    total_saved = 0

    for batch in chunk_list(
        rows,
        SUPABASE_BATCH_SIZE,
    ):

        try:

            with httpx.Client(
                timeout=90
            ) as client:

                response = client.post(
                    url,
                    params={
                        "on_conflict": on_conflict
                    },
                    headers=headers,
                    json=batch,
                )

        except Exception as exc:

            raise RuntimeError(
                f"Error conectando con "
                f"Supabase para {table}: "
                f"{exc}"
            ) from exc

        if response.status_code >= 400:

            raise RuntimeError(
                f"Supabase UPSERT {table}: "
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )

        total_saved += len(
            batch
        )

    return total_saved


# ============================================================
# ID DE TEMPORADA
# ============================================================

def build_season_id(
    league_id: int,
    season: int,
) -> int:

    # ID determinista.
    #
    # Ejemplo:
    # liga 140 + temporada 2024
    # => 1402024
    #
    # Esto mantiene las relaciones con matches
    # aunque el endpoint fixtures no devuelva
    # season.id como campo separado.
    return (
        league_id * 10000
        + season
    )


# ============================================================
# CONSTRUIR DATOS DE SUPABASE
# ============================================================

def build_league_row(
    league_id: int,
    league_name: str,
    fixture_rows: list[dict[str, Any]],
) -> dict[str, Any]:

    country = None
    league_type = "League"

    for item in fixture_rows:

        league_info = (
            item.get("league")
            or {}
        )

        country_value = clean_text(
            league_info.get("country")
        )

        type_value = clean_text(
            league_info.get("type")
        )

        if country_value:
            country = country_value

        if type_value:
            league_type = type_value

        if country or type_value:
            break

    return {
        "id": league_id,
        "name": league_name,
        "country": country,
        "type": league_type,
        "active": True,
    }


def build_season_row(
    league_id: int,
    season: int,
    fixture_rows: list[dict[str, Any]],
) -> dict[str, Any]:

    season_id = build_season_id(
        league_id,
        season,
    )

    starting_at = None
    ending_at = None

    # Buscamos fechas si el proveedor las devuelve.
    dates = []

    for item in fixture_rows:

        fixture_info = (
            item.get("fixture")
            or {}
        )

        date_text = clean_text(
            fixture_info.get("date")
        )

        if date_text:
            dates.append(
                date_text
            )

    if dates:

        dates.sort()

        starting_at = dates[0]

        ending_at = dates[-1]

    return {
        "id": season_id,
        "league_id": league_id,
        "name": str(season),
        "starting_at": starting_at,
        "ending_at": ending_at,
    }


def build_team_rows(
    league_id: int,
    fixture_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    teams: dict[
        int,
        dict[str, Any]
    ] = {}

    for item in fixture_rows:

        teams_info = (
            item.get("teams")
            or {}
        )

        for side in (
            "home",
            "away",
        ):

            team = (
                teams_info.get(side)
                or {}
            )

            team_id = safe_int(
                team.get("id")
            )

            if team_id is None:
                continue

            team_name = clean_text(
                team.get("name")
            )

            code = clean_text(
                team.get("code")
            )

            if team_id not in teams:

                teams[team_id] = {
                    "id": team_id,
                    "name": (
                        team_name
                        or f"Team {team_id}"
                    ),
                    "short_code": code,
                    "country": None,
                    "venue_name": None,
                    "league_id": league_id,
                }

            else:

                if (
                    team_name
                    and not teams[
                        team_id
                    ].get("name")
                ):

                    teams[
                        team_id
                    ]["name"] = (
                        team_name
                    )

                if (
                    code
                    and not teams[
                        team_id
                    ].get(
                        "short_code"
                    )
                ):

                    teams[
                        team_id
                    ]["short_code"] = (
                        code
                    )

    return list(
        teams.values()
    )


def build_match_rows(
    league_id: int,
    season: int,
    fixture_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    int,
]:

    season_id = build_season_id(
        league_id,
        season,
    )

    matches: list[
        dict[str, Any]
    ] = []

    skipped = 0

    for item in fixture_rows:

        fixture_info = (
            item.get("fixture")
            or {}
        )

        league_info = (
            item.get("league")
            or {}
        )

        teams_info = (
            item.get("teams")
            or {}
        )

        goals_info = (
            item.get("goals")
            or {}
        )

        score_info = (
            item.get("score")
            or {}
        )

        fixture_id = safe_int(
            fixture_info.get("id")
        )

        home = (
            teams_info.get("home")
            or {}
        )

        away = (
            teams_info.get("away")
            or {}
        )

        home_team_id = safe_int(
            home.get("id")
        )

        away_team_id = safe_int(
            away.get("id")
        )

        if (
            fixture_id is None
            or home_team_id is None
            or away_team_id is None
        ):

            skipped += 1
            continue

        # La temporada indicada por el usuario
        # tiene prioridad.
        fixture_season = safe_int(
            league_info.get("season"),
            season,
        )

        if fixture_season is None:
            fixture_season = season

        fixture_season_id = build_season_id(
            league_id,
            fixture_season,
        )

        status_info = (
            fixture_info.get("status")
            or {}
        )

        halftime = (
            score_info.get("halftime")
            or {}
        )

        row = {
            "id": fixture_id,

            "league_id": league_id,

            "season_id": (
                fixture_season_id
            ),

            "home_team_id": (
                home_team_id
            ),

            "away_team_id": (
                away_team_id
            ),

            "referee_id": None,

            "starting_at": clean_text(
                fixture_info.get("date")
            ),

            "status": clean_text(
                status_info.get("short")
            ),

            "home_goals": safe_int(
                goals_info.get("home")
            ),

            "away_goals": safe_int(
                goals_info.get("away")
            ),

            "home_ht_goals": safe_int(
                halftime.get("home")
            ),

            "away_ht_goals": safe_int(
                halftime.get("away")
            ),
        }

        matches.append(
            row
        )

    return matches, skipped


# ============================================================
# SINCRONIZAR UNA LIGA + TEMPORADA
# ============================================================

def sync_league_season(
    league_id: int,
    league_name: str,
    season: int,
) -> dict[str, Any]:

    print("")
    print("========================================")
    print(
        f"{league_name} — {season}"
    )
    print("========================================")

    # --------------------------------------------------------
    # UNA SOLA llamada a API-Football.
    # No usamos /teams ni /leagues por separado.
    # --------------------------------------------------------

    data = football_get(
        "/fixtures",
        {
            "league": league_id,
            "season": season,
        },
    )

    response = data.get(
        "response",
        []
    )

    if not isinstance(
        response,
        list,
    ):

        response = []

    print("")
    print(
        f"Fixtures recibidos: "
        f"{len(response)}"
    )

    # --------------------------------------------------------
    # Si no hay fixtures, se considera que esa
    # combinación no está disponible.
    # --------------------------------------------------------

    if not response:

        return {
            "league_id": league_id,
            "league": league_name,
            "season": season,
            "fixtures_received": 0,
            "teams_saved": 0,
            "matches_saved": 0,
            "skipped": 0,
            "available": False,
        }

    # --------------------------------------------------------
    # LIGA
    # --------------------------------------------------------

    league_row = build_league_row(
        league_id,
        league_name,
        response,
    )

    supabase_upsert_batch(
        "leagues",
        [league_row],
        "id",
    )

    # --------------------------------------------------------
    # TEMPORADA
    # --------------------------------------------------------

    season_row = build_season_row(
        league_id,
        season,
        response,
    )

    supabase_upsert_batch(
        "seasons",
        [season_row],
        "id",
    )

    # --------------------------------------------------------
    # EQUIPOS
    # --------------------------------------------------------

    team_rows = build_team_rows(
        league_id,
        response,
    )

    teams_saved = supabase_upsert_batch(
        "teams",
        team_rows,
        "id",
    )

    print(
        f"Equipos guardados: "
        f"{teams_saved}"
    )

    # --------------------------------------------------------
    # PARTIDOS
    # --------------------------------------------------------

    match_rows, skipped = (
        build_match_rows(
            league_id,
            season,
            response,
        )
    )

    matches_saved = supabase_upsert_batch(
        "matches",
        match_rows,
        "id",
    )

    print(
        f"Partidos guardados: "
        f"{matches_saved}"
    )

    print(
        f"Partidos omitidos: "
        f"{skipped}"
    )

    return {
        "league_id": league_id,
        "league": league_name,
        "season": season,
        "fixtures_received": len(response),
        "teams_saved": teams_saved,
        "matches_saved": matches_saved,
        "skipped": skipped,
        "available": True,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("")
    print("========================================")
    print("FÚTBOL IA - HISTORIAL COMPLETO")
    print("========================================")
    print("")

    print(
        "Competiciones:"
    )

    for league_id, name in LEAGUES.items():

        print(
            f"  - {league_id}: "
            f"{name}"
        )

    print("")

    print(
        "Temporadas:"
    )

    for season in SEASONS:

        print(
            f"  - {season}"
        )

    print("")

    print(
        "Total de consultas API previstas: "
        f"{len(LEAGUES) * len(SEASONS)}"
    )

    print(
        f"Espera entre consultas: "
        f"{SECONDS_BETWEEN_API_CALLS}s"
    )

    # --------------------------------------------------------
    # VALIDAR CONFIGURACIÓN
    # --------------------------------------------------------

    get_api_key()
    get_supabase_config()

    print("")
    print(
        "Configuración validada correctamente."
    )

    total_fixtures = 0
    total_teams = 0
    total_matches = 0
    total_skipped = 0

    available_count = 0
    unavailable_count = 0

    results = []

    # --------------------------------------------------------
    # PROCESAR
    # --------------------------------------------------------

    combinations = [
        (
            league_id,
            league_name,
            season,
        )
        for league_id, league_name
        in LEAGUES.items()
        for season in SEASONS
    ]

    for index, (
        league_id,
        league_name,
        season,
    ) in enumerate(
        combinations,
        start=1,
    ):

        print("")
        print(
            "########################################"
        )
        print(
            f"CONSULTA {index}/"
            f"{len(combinations)}"
        )
        print(
            f"{league_name} "
            f"({league_id}) — {season}"
        )
        print(
            "########################################"
        )

        try:

            result = sync_league_season(
                league_id=league_id,
                league_name=league_name,
                season=season,
            )

            results.append(
                result
            )

            total_fixtures += (
                result[
                    "fixtures_received"
                ]
            )

            total_teams += (
                result[
                    "teams_saved"
                ]
            )

            total_matches += (
                result[
                    "matches_saved"
                ]
            )

            total_skipped += (
                result[
                    "skipped"
                ]
            )

            if result["available"]:

                available_count += 1

            else:

                unavailable_count += 1

                print(
                    "⚠️ Sin fixtures para "
                    f"{league_name} {season}."
                )

        except Exception as exc:

            # Si API-Football nos informa de una
            # temporada que no está disponible,
            # la registramos como no disponible.
            #
            # Otros errores hacen fallar el workflow.
            message = str(exc)

            lower_message = (
                message.lower()
            )

            is_plan_season_error = (
                "season" in lower_message
                and (
                    "free plan"
                    in lower_message
                    or "not available"
                    in lower_message
                    or "do not have access"
                    in lower_message
                )
            )

            if is_plan_season_error:

                unavailable_count += 1

                print("")
                print(
                    "⚠️ Temporada no disponible "
                    "en el plan actual:"
                )

                print(
                    message
                )

                results.append({
                    "league_id": league_id,
                    "league": league_name,
                    "season": season,
                    "fixtures_received": 0,
                    "teams_saved": 0,
                    "matches_saved": 0,
                    "skipped": 0,
                    "available": False,
                    "plan_unavailable": True,
                })

            else:

                raise RuntimeError(
                    f"Falló {league_name} "
                    f"{season}: {message}"
                ) from exc

        # ----------------------------------------------------
        # ESPERA ENTRE CONSULTAS API
        # ----------------------------------------------------

        if index < len(combinations):

            print("")
            print(
                "Esperando "
                f"{SECONDS_BETWEEN_API_CALLS}s "
                "para respetar el límite de "
                "API-Football..."
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
    print("RESUMEN HISTORIAL COMPLETO")
    print("========================================")

    print(
        f"Combinaciones procesadas: "
        f"{len(combinations)}"
    )

    print(
        f"Combinaciones disponibles: "
        f"{available_count}"
    )

    print(
        f"Combinaciones no disponibles: "
        f"{unavailable_count}"
    )

    print(
        f"Fixtures recibidos: "
        f"{total_fixtures}"
    )

    print(
        f"Equipos guardados: "
        f"{total_teams}"
    )

    print(
        f"Partidos guardados: "
        f"{total_matches}"
    )

    print(
        f"Partidos omitidos: "
        f"{total_skipped}"
    )

    # --------------------------------------------------------
    # DETALLE
    # --------------------------------------------------------

    print("")
    print(
        "DETALLE:"
    )

    for result in results:

        status = (
            "OK"
            if result.get(
                "available"
            )
            else "NO DISPONIBLE"
        )

        print(
            f"- "
            f"{result['league']} "
            f"{result['season']}: "
            f"{status} | "
            f"fixtures="
            f"{result['fixtures_received']} | "
            f"matches="
            f"{result['matches_saved']}"
        )

    # --------------------------------------------------------
    # VALIDACIÓN FINAL
    # --------------------------------------------------------

    if total_matches == 0:

        raise RuntimeError(
            "No se guardó ningún partido "
            "histórico."
        )

    print("")
    print(
        "========================================"
    )

    print(
        "✅ HISTORIAL COMPLETO CARGADO"
    )

    print(
        "========================================"
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

        print(
            "========================================"
        )

        sys.exit(1)
