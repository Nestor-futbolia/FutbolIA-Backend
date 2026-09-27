import os
import time
import requests


BASE_URL = "https://v3.football.api-sports.io"

HISTORICAL_SEASON = int(
    os.getenv("HISTORICAL_SEASON", "2024")
)

BATCH_INDEX = int(
    os.getenv("LEAGUE_BATCH", "0")
)

# Máximo de llamadas API-Football por ejecución.
MAX_API_CALLS_PER_RUN = 20

# Free = 10/min.
# Usamos 8 segundos para mantenernos claramente por debajo.
SECONDS_BETWEEN_API_CALLS = 8

# Si el encabezado indica pocas solicitudes restantes,
# dejamos margen de seguridad.
MIN_DAILY_REMAINING = 6


# ============================================================
# LIGAS PRIORITARIAS
# ============================================================

PRIORITY_LEAGUES = [
    (39, "Premier League", "England"),
    (140, "La Liga", "Spain"),
    (135, "Serie A", "Italy"),
    (78, "Bundesliga", "Germany"),

    (61, "Ligue 1", "France"),
    (40, "Championship", "England"),
    (88, "Eredivisie", "Netherlands"),
    (94, "Primeira Liga", "Portugal"),

    (71, "Serie A Brazil", "Brazil"),
    (253, "MLS", "USA"),
    (2, "UEFA Champions League", "Europe"),
    (3, "UEFA Europa League", "Europe"),

    (13, "CONMEBOL Libertadores", "South America"),
    (242, "Liga Pro", "Ecuador"),
]


# ============================================================
# VARIABLES
# ============================================================

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


# ============================================================
# EXCEPCIONES CONTROLADAS
# ============================================================

class DailyQuotaReached(Exception):
    pass


class ApiCallBudgetReached(Exception):
    pass


# ============================================================
# SUPABASE
# ============================================================

def supabase_get(path, params=None):

    url = (
        f"{SUPABASE_URL.rstrip('/')}"
        f"/rest/v1/{path}"
    )

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
            f"{response.text[:1000]}"
        )

    return response.json()


def supabase_upsert(
    path,
    rows,
    on_conflict="id",
):

    if not rows:
        return

    url = (
        f"{SUPABASE_URL.rstrip('/')}"
        f"/rest/v1/{path}"
    )

    headers = dict(SUPABASE_HEADERS)

    headers["Prefer"] = (
        "resolution=merge-duplicates,"
        "return=minimal"
    )

    response = requests.post(
        url,
        headers=headers,
        params={
            "on_conflict": on_conflict
        },
        json=rows,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"Supabase UPSERT {path}: "
            f"HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )


# ============================================================
# ESPERA ENTRE LLAMADAS
# ============================================================

_last_api_call_time = 0.0


def wait_before_api_call():

    global _last_api_call_time

    now = time.time()

    elapsed = (
        now - _last_api_call_time
    )

    if elapsed < SECONDS_BETWEEN_API_CALLS:

        wait_time = (
            SECONDS_BETWEEN_API_CALLS
            - elapsed
        )

        print(
            f"Esperando {wait_time:.1f}s "
            "antes de la siguiente llamada..."
        )

        time.sleep(wait_time)

    _last_api_call_time = time.time()


# ============================================================
# API-FOOTBALL
# ============================================================

def football_get(
    endpoint,
    params,
):

    global api_calls
    global daily_remaining
    global minute_remaining

    if api_calls >= MAX_API_CALLS_PER_RUN:

        raise ApiCallBudgetReached(
            f"Se alcanzó el máximo de "
            f"{MAX_API_CALLS_PER_RUN} "
            "llamadas de esta ejecución."
        )

    if (
        daily_remaining is not None
        and daily_remaining
        <= MIN_DAILY_REMAINING
    ):

        raise DailyQuotaReached(
            f"Quedan solamente "
            f"{daily_remaining} "
            "solicitudes diarias."
        )

    wait_before_api_call()

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
        f"daily={daily_raw or '?'} | "
        f"minute={minute_raw or '?'}"
    )

    # --------------------------------------------------------
    # HTTP 429
    # --------------------------------------------------------

    if response.status_code == 429:

        text = response.text.lower()

        if (
            "day" in text
            or "daily" in text
        ):

            raise DailyQuotaReached(
                "API-Football indicó "
                "límite diario agotado."
            )

        raise RuntimeError(
            "API-Football HTTP 429: "
            f"{response.text[:1000]}"
        )

    # --------------------------------------------------------
    # OTROS ERRORES HTTP
    # --------------------------------------------------------

    if not response.ok:

        raise RuntimeError(
            f"API-Football HTTP "
            f"{response.status_code}: "
            f"{response.text[:1000]}"
        )

    data = response.json()

    # --------------------------------------------------------
    # MUY IMPORTANTE:
    # API-Football puede devolver HTTP 200
    # pero contener un error dentro de "errors".
    # --------------------------------------------------------

    errors = data.get("errors") or {}

    if errors:

        error_text = str(errors)

        print(
            "API-Football devolvió errors: "
            f"{error_text}"
        )

        lower_error = error_text.lower()

        # Si el cuerpo dice que la cuota diaria está
        # agotada, NOS DETENEMOS INMEDIATAMENTE.
        if (
            "request limit for the day"
            in lower_error
            or "limit for the day"
            in lower_error
            or "daily" in lower_error
            or "day" in lower_error
            and "limit" in lower_error
        ):

            raise DailyQuotaReached(
                "API-Football confirmó "
                "que la cuota diaria está agotada."
            )

        raise RuntimeError(
            f"API-Football errors: {errors}"
        )

    return data


# ============================================================
# GUARDAR LIGA Y TEMPORADA
# ============================================================

def ensure_league_and_season(
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

    # La tabla seasons utiliza el año como ID
    # en la estructura actual del proyecto.
    existing = supabase_get(
        "seasons",
        {
            "id": f"eq.{HISTORICAL_SEASON}",
            "select": "id",
            "limit": 1,
        },
    )

    if not existing:

        supabase_upsert(
            "seasons",
            [
                {
                    "id": HISTORICAL_SEASON,
                    "league_id": league_id,
                    "name": str(
                        HISTORICAL_SEASON
                    ),
                }
            ],
            "id",
        )


# ============================================================
# GUARDAR FIXTURES
# ============================================================

def save_fixtures(
    fixtures,
    league_id,
    league_name,
    country,
):

    teams = {}
    matches = {}

    finished_statuses = {
        "FT",
        "AET",
        "PEN",
    }

    for item in fixtures:

        fixture = (
            item.get("fixture") or {}
        )

        status = (
            fixture.get("status") or {}
        ).get("short")

        # Solo partidos terminados.
        if status not in finished_statuses:
            continue

        fixture_id = fixture.get("id")

        teams_obj = (
            item.get("teams") or {}
        )

        home = (
            teams_obj.get("home") or {}
        )

        away = (
            teams_obj.get("away") or {}
        )

        if not fixture_id:
            continue

        if not home.get("id"):
            continue

        if not away.get("id"):
            continue

        goals = (
            item.get("goals") or {}
        )

        score = (
            item.get("score") or {}
        )

        halftime = (
            score.get("halftime") or {}
        )

        home_id = int(home["id"])
        away_id = int(away["id"])

        teams[home_id] = {
            "id": home_id,
            "name": (
                home.get("name")
                or f"Team {home_id}"
            ),
            "short_code": home.get("code"),
            "country": country,
            "league_id": league_id,
        }

        teams[away_id] = {
            "id": away_id,
            "name": (
                away.get("name")
                or f"Team {away_id}"
            ),
            "short_code": away.get("code"),
            "country": country,
            "league_id": league_id,
        }

        matches[int(fixture_id)] = {
            "id": int(fixture_id),
            "league_id": league_id,
            "season_id": HISTORICAL_SEASON,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "starting_at": fixture.get(
                "date"
            ),
            "status": status,
            "home_goals": goals.get(
                "home"
            ),
            "away_goals": goals.get(
                "away"
            ),
            "home_ht_goals": halftime.get(
                "home"
            ),
            "away_ht_goals": halftime.get(
                "away"
            ),
        }

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


# ============================================================
# SINCRONIZAR UNA LIGA
# ============================================================

def sync_league(
    league_id,
    league_name,
    country,
):

    print("")
    print(
        f"=== {league_name} "
        f"({league_id}) ==="
    )

    ensure_league_and_season(
        league_id,
        league_name,
        country,
    )

    page = 1

    total_received = 0
    total_saved = 0

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

        fixtures = (
            data.get("response") or []
        )

        total_received += len(fixtures)

        paging = (
            data.get("paging") or {}
        )

        current_page = int(
            paging.get(
                "current",
                page,
            )
        )

        total_pages = int(
            paging.get(
                "total",
                1,
            )
        )

        print(
            f"{league_name}: "
            f"página {current_page}/"
            f"{total_pages} | "
            f"recibidos={len(fixtures)}"
        )

        saved = save_fixtures(
            fixtures,
            league_id,
            league_name,
            country,
        )

        total_saved += saved

        print(
            f"{league_name}: "
            f"guardados={saved}"
        )

        if current_page >= total_pages:
            break

        page = current_page + 1

    return (
        total_received,
        total_saved,
    )


# ============================================================
# SELECCIONAR SOLO UN LOTE
# ============================================================

def get_batch():

    batch_size = 4

    start = (
        BATCH_INDEX * batch_size
    )

    end = start + batch_size

    batch = PRIORITY_LEAGUES[
        start:end
    ]

    if not batch:

        raise RuntimeError(
            f"El lote {BATCH_INDEX} "
            "no existe."
        )

    return batch


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================"
    )

    print(
        "FÚTBOL IA - "
        "RECOLECTOR HISTÓRICO V3"
    )

    print(
        "======================================"
    )

    print(
        f"Temporada: "
        f"{HISTORICAL_SEASON}"
    )

    print(
        f"Lote: "
        f"{BATCH_INDEX}"
    )

    print(
        f"Máximo llamadas: "
        f"{MAX_API_CALLS_PER_RUN}"
    )

    print(
        f"Espera: "
        f"{SECONDS_BETWEEN_API_CALLS}s"
    )

    print("")

    batch = get_batch()

    print(
        "Ligas de este lote:"
    )

    for league_id, name, country in batch:

        print(
            f"  {league_id} - "
            f"{name} - "
            f"{country}"
        )

    total_received = 0
    total_saved = 0
    completed = 0

    try:

        for (
            league_id,
            league_name,
            country,
        ) in batch:

            try:

                received, saved = (
                    sync_league(
                        league_id,
                        league_name,
                        country,
                    )
                )

                total_received += received
                total_saved += saved

                completed += 1

                print(
                    f"COMPLETADA: "
                    f"{league_name} | "
                    f"recibidos={received} | "
                    f"guardados={saved}"
                )

            except DailyQuotaReached as exc:

                print("")
                print(
                    "CUOTA DIARIA ALCANZADA."
                )

                print(
                    str(exc)
                )

                print(
                    "La ejecución se detiene "
                    "sin intentar más ligas."
                )

                break

            except ApiCallBudgetReached as exc:

                print("")
                print(
                    "PRESUPUESTO DE LLAMADAS "
                    "ALCANZADO."
                )

                print(
                    str(exc)
                )

                break

            # Si todavía quedan ligas del lote,
            # respetamos la velocidad de API.
            if completed < len(batch):

                time.sleep(
                    SECONDS_BETWEEN_API_CALLS
                )

    except DailyQuotaReached as exc:

        print(
            "CUOTA DIARIA: "
            f"{exc}"
        )

    print("")
    print(
        "======================================"
    )

    print("RESUMEN")

    print(
        f"Ligas completadas: "
        f"{completed}/{len(batch)}"
    )

    print(
        f"Llamadas API utilizadas: "
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
        f"Daily remaining header: "
        f"{daily_remaining}"
    )

    print(
        f"Minute remaining header: "
        f"{minute_remaining}"
    )

    print(
        "======================================"
    )

    if completed == 0:

        print(
            "No se completó ninguna liga."
        )

        print(
            "Si API-Football informó "
            "cuota agotada, simplemente "
            "esperaremos al siguiente "
            "reinicio de cuota."
        )

        # No provocamos un fallo adicional.
        return

    print("")
    print(
        "SINCRONIZACIÓN TERMINADA "
        "CORRECTAMENTE."
    )


if __name__ == "__main__":
    main()
