import os
import time
import requests

BASE_URL = "https://v3.football.api-sports.io"
HISTORICAL_SEASON = 2024

# Free API-Football: 100 requests/day and 10 requests/minute.
# We deliberately stay below the daily ceiling and space calls.
MAX_LEAGUES_PER_RUN = 20
MAX_API_CALLS_PER_RUN = 80
SECONDS_BETWEEN_API_CALLS = 7

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

API_HEADERS = {"x-apisports-key": API_FOOTBALL_KEY}


def supabase_get(path, params=None):
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{path}"
    r = requests.get(url, headers=SUPABASE_HEADERS, params=params, timeout=60)
    if not r.ok:
        raise RuntimeError(
            f"Supabase GET {path}: HTTP {r.status_code}: {r.text[:700]}"
        )
    return r.json()


def supabase_upsert(path, rows, on_conflict):
    if not rows:
        return 0
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{path}"
    headers = dict(SUPABASE_HEADERS)
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    r = requests.post(
        url,
        headers=headers,
        params={"on_conflict": on_conflict},
        json=rows,
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(
            f"Supabase UPSERT {path}: HTTP {r.status_code}: {r.text[:1000]}"
        )
    return len(rows)


api_calls = 0
daily_remaining = None


def football_get(endpoint, params):
    global api_calls, daily_remaining

    if api_calls >= MAX_API_CALLS_PER_RUN:
        raise RuntimeError(
            f"Se alcanzó el límite de seguridad de {MAX_API_CALLS_PER_RUN} "
            "solicitudes en esta ejecución."
        )

    r = requests.get(
        f"{BASE_URL}{endpoint}",
        headers=API_HEADERS,
        params=params,
        timeout=90,
    )
    api_calls += 1

    daily = r.headers.get("x-ratelimit-requests-remaining")
    minute = r.headers.get("X-RateLimit-Remaining")
    try:
        daily_remaining = int(daily) if daily is not None else None
    except ValueError:
        daily_remaining = None

    print(
        f"API {endpoint} HTTP {r.status_code} | "
        f"solicitud #{api_calls} | daily_remaining={daily or '?'} | "
        f"minute_remaining={minute or '?'}"
    )

    if not r.ok:
        raise RuntimeError(
            f"API-Football HTTP {r.status_code}: {r.text[:1000]}"
        )

    data = r.json()
    errors = data.get("errors") or {}
    if errors:
        raise RuntimeError(f"API-Football errors: {errors}")

    return data


def get_target_leagues():
    # First use leagues already represented by future matches in our DB.
    # This avoids spending an API call just to discover leagues.
    rows = supabase_get(
        "matches",
        {
            "select": "league_id",
            "status": "eq.NS",
            "league_id": "not.is.null",
            "starting_at": "not.is.null",
            "order": "starting_at.asc",
            "limit": 1000,
        },
    )

    ids = []
    seen = set()
    for row in rows:
        lid = row.get("league_id")
        if lid is not None and int(lid) not in seen:
            seen.add(int(lid))
            ids.append(int(lid))

    # If there are no upcoming matches, use active leagues already stored.
    if not ids:
        rows = supabase_get(
            "leagues",
            {
                "select": "id",
                "active": "eq.true",
                "order": "id.asc",
                "limit": 1000,
            },
        )
        for row in rows:
            lid = row.get("id")
            if lid is not None and int(lid) not in seen:
                seen.add(int(lid))
                ids.append(int(lid))

    return ids[:MAX_LEAGUES_PER_RUN]


def save_league_and_season(league_obj, season_obj):
    league_id = league_obj.get("id")
    season_year = season_obj.get("year") or HISTORICAL_SEASON

    if league_id is None:
        return

    supabase_upsert(
        "leagues",
        [
            {
                "id": int(league_id),
                "name": league_obj.get("name") or f"League {league_id}",
                "country": league_obj.get("country") or "",
                "type": league_obj.get("type") or "",
                "active": True,
            }
        ],
        "id",
    )

    # In this schema the season primary key is numeric. API-Football season
    # identifiers are the season year for these competitions.
    supabase_upsert(
        "seasons",
        [
            {
                "id": int(season_year),
                "league_id": int(league_id),
                "name": str(season_year),
            }
        ],
        "id",
    )


def save_fixtures(fixtures, league_id):
    teams = {}
    matches = {}

    for item in fixtures:
        fixture = item.get("fixture") or {}
        league = item.get("league") or {}
        teams_obj = item.get("teams") or {}
        goals = item.get("goals") or {}
        score = item.get("score") or {}
        halftime = score.get("halftime") or {}

        fid = fixture.get("id")
        home = teams_obj.get("home") or {}
        away = teams_obj.get("away") or {}

        if fid is None or home.get("id") is None or away.get("id") is None:
            continue

        for team in (home, away):
            tid = int(team["id"])
            teams[tid] = {
                "id": tid,
                "name": team.get("name") or f"Team {tid}",
                "short_code": team.get("code"),
                "country": None,
                "venue_name": None,
                "league_id": int(league_id),
            }

        matches[int(fid)] = {
            "id": int(fid),
            "league_id": int(league_id),
            "season_id": int((league.get("season") or {}).get("year") or HISTORICAL_SEASON),
            "home_team_id": int(home["id"]),
            "away_team_id": int(away["id"]),
            "starting_at": fixture.get("date"),
            "status": (fixture.get("status") or {}).get("short"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "home_ht_goals": halftime.get("home"),
            "away_ht_goals": halftime.get("away"),
        }

    if teams:
        supabase_upsert("teams", list(teams.values()), "id")
    if matches:
        supabase_upsert("matches", list(matches.values()), "id")

    return len(matches)


def fetch_all_league_fixtures(league_id):
    all_fixtures = []
    page = 1

    while True:
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
        all_fixtures.extend(fixtures)

        paging = data.get("paging") or {}
        total_pages = int(paging.get("total") or 1)

        print(
            f"Liga {league_id}: página {page}/{total_pages}, "
            f"partidos en página={len(fixtures)}"
        )

        if page >= total_pages:
            break

        page += 1
        time.sleep(SECONDS_BETWEEN_API_CALLS)

    return all_fixtures


def main():
    print("=== FÚTBOL IA - RECOLECTOR HISTÓRICO OPTIMIZADO ===")
    print(f"Temporada: {HISTORICAL_SEASON}")
    print("Estrategia: 1 llamada / liga-temporada en vez de 1 llamada / equipo.")
    print(f"Máximo de ligas por ejecución: {MAX_LEAGUES_PER_RUN}")
    print(f"Máximo de llamadas API por ejecución: {MAX_API_CALLS_PER_RUN}")
    print("Las estadísticas por partido NO se descargan en este paso.")
    print("")

    leagues = get_target_leagues()
    if not leagues:
        raise RuntimeError("No hay ligas objetivo en Supabase.")

    print(f"Ligas seleccionadas: {len(leagues)}")
    print(f"IDs: {leagues}")

    total_fixtures = 0
    total_saved = 0
    errors = []

    for index, league_id in enumerate(leagues, start=1):
        print("")
        print(f"--- LIGA {index}/{len(leagues)}: {league_id} ---")

        try:
            fixtures = fetch_all_league_fixtures(league_id)
            print(f"Total recibido para liga {league_id}: {len(fixtures)}")

            if fixtures:
                first_league = fixtures[0].get("league") or {}
                first_season = first_league.get("season") or {}
                save_league_and_season(first_league, first_season)

            saved = save_fixtures(fixtures, league_id)
            total_fixtures += len(fixtures)
            total_saved += saved

            print(f"Partidos guardados/actualizados: {saved}")

        except Exception as exc:
            print(f"ERROR en liga {league_id}: {exc}")
            errors.append((league_id, str(exc)))

        if daily_remaining is not None and daily_remaining <= 5:
            print(
                "Quedan 5 o menos solicitudes diarias. "
                "Se detiene para proteger la cuota."
            )
            break

        if index < len(leagues):
            print(f"Esperando {SECONDS_BETWEEN_API_CALLS}s...")
            time.sleep(SECONDS_BETWEEN_API_CALLS)

    print("")
    print("=== RESUMEN ===")
    print(f"Ligas intentadas: {len(leagues)}")
    print(f"Solicitudes API usadas: {api_calls}")
    print(f"Partidos recibidos: {total_fixtures}")
    print(f"Partidos guardados/actualizados: {total_saved}")
    print(f"Errores: {len(errors)}")

    for league_id, error in errors:
        print(f"  - Liga {league_id}: {error}")

    if errors:
        raise RuntimeError("La carga terminó con errores.")

    if total_fixtures == 0:
        raise RuntimeError("No se recibieron partidos históricos.")

    print("HISTORIAL OPTIMIZADO CARGADO CORRECTAMENTE")


if __name__ == "__main__":
    main()
