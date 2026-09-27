import os
import time
import requests

BASE_URL = "https://v3.football.api-sports.io"
HISTORICAL_SEASON = 2024

# API-Football Free:
# 100 solicitudes/día
# 10 solicitudes/minuto
#
# Dejamos un presupuesto pequeño por ejecución para no consumir
# toda la cuota de una sola vez.
MAX_API_CALLS_PER_RUN = 20
SECONDS_BETWEEN_API_CALLS = 7
MIN_DAILY_REMAINING_TO_CONTINUE = 6

# Ligas prioritarias.
# Una consulta por liga/temporada puede devolver muchos partidos.
PRIORITY_LEAGUES = [
    (39, "Premier League", "England"),
    (140, "La Liga", "Spain"),
    (135, "Serie A", "Italy"),
    (78, "Bundesliga", "Germany"),
    (61, "Ligue 1", "France"),
    (40, "Championship", "England"),
    (88, "Eredivisie", "Netherlands"),
    (94, "Primeira Liga", "Portugal"),
    (71, "Serie A", "Brazil"),
    (253, "MLS", "USA"),
    (2, "UEFA Champions League", "Europe"),
    (3, "UEFA Europa League", "Europe"),
    (13, "CONMEBOL Libertadores", "South America"),
    (242, "Liga Pro", "Ecuador"),
]

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not API_FOOTBALL_KEY:
    raise RuntimeError("Falta API_FOOTBALL_KEY")

if not SUPABASE_URL:
    raise RuntimeError("Falta SUPABASE_URL")

if not SUPABASE_SECRET_KEY:
    raise RuntimeError("Falta SUPABASE_SECRET_KEY")


SUPABASE_HEADERS = {
    "apikey": SUPABASE_SECRET_KEY,
    "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
    "Content-Type": "application/json",
}

API_HEADERS = {
    "x-apisports-key": API_FOOTBALL_KEY
}


api_calls = 0
daily_remaining = None
minute_remaining = None


def supabase_get(path, params=None):
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{path}"

    response = requests.get(
        url,
        headers=SUPABASE_HEADERS,
        params=params,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Supabase GET {path}: "
            f"HTTP {response.status_code}: "
            f"{response.text[:700]}"
        )

    return response.json()


def supabase_upsert(path, rows, on_conflict):
    if not rows:
        return 0

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{path}"

    headers = dict(SUPABASE_HEADERS)
    headers["Prefer"] = (
        "resolution=merge-duplicates,return=minimal"
    )

    response = requests.post(
        url,
        headers=headers,
        params={"on_conflict": on_conflict},
        json=rows,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"Supabase UPSERT {path}: "
            f"HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    return len(rows)


def football_get(endpoint, params):
    global api_calls
    global daily_remaining
    global minute_remaining

    if api_calls >= MAX_API_CALLS_PER_RUN:
        raise StopIteration(
            "Presupuesto de seguridad alcanzado: "
            f"{MAX_API_CALLS_PER_RUN} solicitudes."
        )

    response = requests.get(
        f"{BASE_URL}{endpoint}",
        headers=API_HEADERS,
        params=params,
        timeout=90,
    )

    api_calls += 1

    daily_raw = response.headers.get(
        "x-ratelimit-requests-remaining"
    )

    minute_raw = response.headers.get(
        "X-RateLimit-Remaining"
    )

    try:
        daily_remaining = (
            int(daily_raw)
            if daily_raw is not None
            else None
        )
    except ValueError:
        daily_remaining = None

    try:
        minute_remaining = (
            int(minute_raw)
            if minute_raw is not None
            else None
        )
    except ValueError:
        minute_remaining = None

    print(
        f"API {endpoint} "
        f"HTTP {response.status_code} | "
        f"llamada #{api_calls} | "
        f"diarias_restantes={daily_raw or '?'} | "
        f"minuto_restantes={minute_raw or '?'}"
    )

    if not response.ok:
        raise RuntimeError(
            f"API-Football HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    data = response.json()

    errors = data.get("errors") or {}

    if errors:
        raise RuntimeError(
            f"API-Football errors: {errors}"
        )

    return data


def choose_leagues():
    """
    Prioriza ligas que todavía no tienen partidos 2024
    almacenados en Supabase.

    Esta función NO consume solicitudes de API-Football.
    """

    states = {}

    for league_id, name, country in PRIORITY_LEAGUES:

        rows = supabase_get(
            "matches",
            {
                "select": "id",
                "league_id": f"eq.{league_id}",
                "season_id": f"eq.{HISTORICAL_SEASON}",
                "limit": 1,
            },
        )

        states[league_id] = bool(rows)

    ordered = sorted(
        PRIORITY_LEAGUES,
        key=lambda item: (
            states[item[0]],
            PRIORITY_LEAGUES.index(item),
        ),
    )

    print("Prioridad de ligas:")

    for league_id, name, country in ordered:
        state = (
            "YA TIENE DATOS"
            if states[league_id]
            else "SIN DATOS"
        )

        print(
            f"  {league_id} - {name} "
            f"({country}) -> {state}"
        )

    return ordered


def save_league_and_season(
    league_id,
    league_name,
    country,
):
    supabase_upsert(
        "leagues",
        [
            {
                "id": league_id,
                "name": league_name,
                "country": country,
                "type": "League",
                "active": True,
            }
        ],
        "id",
    )

    supabase_upsert(
        "seasons",
        [
            {
                "id": HISTORICAL_SEASON,
                "league_id": league_id,
                "name": str(HISTORICAL_SEASON),
            }
        ],
        "id",
    )


def save_fixtures(
    fixtures,
    league_id,
    league_name,
    country,
):
    teams = {}
    matches = {}

    for item in fixtures:

        fixture = item.get("fixture") or {}
        teams_obj = item.get("teams") or {}
        goals = item.get("goals") or {}

        score = item.get("score") or {}
        halftime = score.get("halftime") or {}

        fixture_id = fixture.get("id")

        home = teams_obj.get("home") or {}
        away = teams_obj.get("away") or {}

        if (
            fixture_id is None
            or home.get("id") is None
            or away.get("id") is None
        ):
            continue

        for team in (home, away):

            team_id = int(team["id"])

            teams[team_id] = {
                "id": team_id,
                "name": (
                    team.get("name")
                    or f"Team {team_id}"
                ),
                "short_code": team.get("code"),
                "country": country,
                "venue_name": None,
                "league_id": league_id,
            }

        matches[int(fixture_id)] = {
            "id": int(fixture_id),
            "league_id": league_id,
            "season_id": HISTORICAL_SEASON,
            "home_team_id": int(home["id"]),
            "away_team_id": int(away["id"]),
            "starting_at": fixture.get("date"),
            "status": (
                fixture.get("status") or {}
            ).get("short"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "home_ht_goals": halftime.get("home"),
            "away_ht_goals": halftime.get("away"),
        }

    save_league_and_season(
        league_id,
        league_name,
        country,
    )

    if teams:
        supabase_upsert(
            "teams",
            list(teams.values()),
            "id",
        )

    if matches:
        supabase_upsert(
            "matches",
            list(matches.values()),
            "id",
        )

    return len(matches)


def fetch_league(
    league_id,
    league_name,
    country,
):
    total_received = 0
    total_saved = 0

    page = 1

    while True:

        if (
            daily_remaining is not None
            and daily_remaining
            < MIN_DAILY_REMAINING_TO_CONTINUE
        ):
            raise StopIteration(
                "Solo quedan "
                f"{daily_remaining} solicitudes diarias. "
                "Se detiene antes de agotar la cuota."
            )

        data = football_get(
            "/fixtures",
            {
                "league": league_id,
                "season": HISTORICAL_SEASON,
                "status": "FT-AET-PEN",
                "page": page,
            },
        )

        fixtures = data.get("response") or []

        total_received += len(fixtures)

        paging = data.get("paging") or {}

        total_pages = int(
            paging.get("total") or 1
        )

        print(
            f"{league_name}: "
            f"página {page}/{total_pages} | "
            f"partidos={len(fixtures)}"
        )

        if fixtures:
            saved = save_fixtures(
                fixtures,
                league_id,
                league_name,
                country,
            )

            total_saved += saved

        if page >= total_pages:
            break

        page += 1

        time.sleep(
            SECONDS_BETWEEN_API_CALLS
        )

    return total_received, total_saved


def main():

    print(
        "=== FÚTBOL IA - "
        "RECOLECTOR HISTÓRICO V2 ==="
    )

    print(
        f"Temporada: {HISTORICAL_SEASON}"
    )

    print(
        "Máximo de solicitudes por ejecución: "
        f"{MAX_API_CALLS_PER_RUN}"
    )

    print(
        "Espera entre llamadas: "
        f"{SECONDS_BETWEEN_API_CALLS}s"
    )

    print(
        "Solo fixtures finalizados."
    )

    print(
        "No se descargan estadísticas "
        "en este paso."
    )

    print(
        "No se consulta API-Football "
        "para descubrir ligas."
    )

    print("")

    leagues = choose_leagues()

    total_received = 0
    total_saved = 0
    completed_leagues = 0

    stopped_by_budget = False

    errors = []

    for index, (
        league_id,
        league_name,
        country,
    ) in enumerate(
        leagues,
        start=1,
    ):

        if api_calls >= MAX_API_CALLS_PER_RUN:
            stopped_by_budget = True
            break

        if (
            daily_remaining is not None
            and daily_remaining
            < MIN_DAILY_REMAINING_TO_CONTINUE
        ):
            stopped_by_budget = True
            break

        print("")
        print(
            f"--- LIGA {index}/"
            f"{len(leagues)}: "
            f"{league_name} "
            f"({league_id}) ---"
        )

        try:

            received, saved = fetch_league(
                league_id,
                league_name,
                country,
            )

            total_received += received
            total_saved += saved

            completed_leagues += 1

            print(
                f"Resultado {league_name}: "
                f"recibidos={received}, "
                f"guardados/actualizados={saved}"
            )

        except StopIteration as exc:

            print(
                f"PAUSA CONTROLADA: {exc}"
            )

            stopped_by_budget = True
            break

        except Exception as exc:

            print(
                f"ERROR en {league_name}: "
                f"{exc}"
            )

            errors.append(
                (
                    league_id,
                    league_name,
                    str(exc),
                )
            )

        if index < len(leagues):

            time.sleep(
                SECONDS_BETWEEN_API_CALLS
            )

    print("")
    print("=== RESUMEN ===")

    print(
        f"Ligas completadas: "
        f"{completed_leagues}"
    )

    print(
        f"Solicitudes API utilizadas: "
        f"{api_calls}"
    )

    print(
        f"Partidos recibidos: "
        f"{total_received}"
    )

    print(
        f"Partidos guardados/actualizados: "
        f"{total_saved}"
    )

    print(
        f"Daily remaining: "
        f"{daily_remaining}"
    )

    print(
        f"Minute remaining: "
        f"{minute_remaining}"
    )

    print(
        f"Errores: "
        f"{len(errors)}"
    )

    print(
        f"Detención por presupuesto/cuota: "
        f"{stopped_by_budget}"
    )

    for (
        league_id,
        league_name,
        error,
    ) in errors:

        print(
            f"  - {league_name} "
            f"({league_id}): "
            f"{error}"
        )

    if errors:
        raise RuntimeError(
            "La ejecución tuvo errores. "
            "Los datos guardados antes del "
            "error permanecen en Supabase."
        )

    if total_received == 0:
        raise RuntimeError(
            "No se recibieron partidos históricos."
        )

    print("")
    print(
        "HISTORIAL OPTIMIZADO "
        "CARGADO CORRECTAMENTE"
    )


if __name__ == "__main__":
    main()
