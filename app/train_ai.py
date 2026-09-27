import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_NAME = "FutbolIA-1X2-LogisticRegression"

LOOKBACK_MATCHES = 5

# Porcentaje temporal para validación.
# Nunca mezclamos partidos futuros con partidos pasados.
TRAIN_RATIO = 0.80

FINISHED_STATUSES = {
    "FT",
    "AET",
    "PEN",
}

FEATURE_NAMES = [
    "home_goals_for_5",
    "home_goals_against_5",
    "home_points_5",
    "away_goals_for_5",
    "away_goals_against_5",
    "away_points_5",
    "goals_form_difference",
    "points_form_difference",
    "home_advantage",
]


# ============================================================
# VARIABLES DE ENTORNO
# ============================================================

def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"No existe la variable de entorno requerida: {name}"
        )

    return value.strip()


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
    default: Optional[int] = None,
) -> Optional[int]:
    try:
        if value is None:
            return default

        return int(value)

    except (TypeError, ValueError):
        return default


def safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    try:
        if value is None:
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def is_finished_status(value: Any) -> bool:
    status = clean_text(value)

    if not status:
        return False

    return status.upper() in FINISHED_STATUSES


# ============================================================
# SUPABASE
# ============================================================

def supabase_headers() -> dict[str, str]:
    _, key = get_supabase_config()

    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def supabase_get(
    table: str,
    params: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:

    supabase_url, _ = get_supabase_config()

    url = f"{supabase_url}/rest/v1/{table}"

    request_params = params or {}

    try:
        with httpx.Client(timeout=60) as client:
            response = client.get(
                url,
                headers=supabase_headers(),
                params=request_params,
            )

    except Exception as exc:
        raise RuntimeError(
            f"Error de conexión con Supabase en "
            f"{table}: {exc}"
        ) from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase GET {table}: "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    try:
        data = response.json()

    except Exception as exc:
        raise RuntimeError(
            f"Supabase devolvió JSON inválido "
            f"para {table}."
        ) from exc

    if not isinstance(data, list):
        raise RuntimeError(
            f"Supabase devolvió un formato inesperado "
            f"para {table}."
        )

    return data


def supabase_upsert(
    table: str,
    payload: dict[str, Any],
    on_conflict: Optional[str] = None,
) -> dict[str, Any]:

    supabase_url, _ = get_supabase_config()

    url = f"{supabase_url}/rest/v1/{table}"

    headers = supabase_headers()

    headers["Prefer"] = (
        "resolution=merge-duplicates,return=representation"
    )

    params = {}

    if on_conflict:
        params["on_conflict"] = on_conflict

    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                url,
                headers=headers,
                params=params,
                json=payload,
            )

    except Exception as exc:
        raise RuntimeError(
            f"Error de conexión con Supabase "
            f"al guardar en {table}: {exc}"
        ) from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase UPSERT {table}: "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    try:
        data = response.json()

    except Exception:
        data = {}

    if isinstance(data, list) and data:
        return data[0]

    if isinstance(data, dict):
        return data

    return {}


def supabase_patch(
    table: str,
    filters: dict[str, str],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:

    supabase_url, _ = get_supabase_config()

    url = f"{supabase_url}/rest/v1/{table}"

    headers = supabase_headers()

    headers["Prefer"] = "return=representation"

    params = dict(filters)

    try:
        with httpx.Client(timeout=60) as client:
            response = client.patch(
                url,
                headers=headers,
                params=params,
                json=payload,
            )

    except Exception as exc:
        raise RuntimeError(
            f"Error de conexión con Supabase "
            f"al actualizar {table}: {exc}"
        ) from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase PATCH {table}: "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    try:
        data = response.json()

    except Exception:
        return []

    if isinstance(data, list):
        return data

    return []


# ============================================================
# CARGAR TODOS LOS PARTIDOS
# ============================================================

def load_all_finished_matches() -> list[dict[str, Any]]:

    print("")
    print("=== CARGANDO PARTIDOS ===")

    all_matches: list[dict[str, Any]] = []

    page_size = 1000
    offset = 0

    while True:

        print(
            f"  Cargando partidos "
            f"{offset + 1}-{offset + page_size}..."
        )

        rows = supabase_get(
            "matches",
            {
                "select": (
                    "id,"
                    "starting_at,"
                    "status,"
                    "home_team_id,"
                    "away_team_id,"
                    "home_goals,"
                    "away_goals"
                ),
                "status": "in.(FT,AET,PEN)",
                "order": "starting_at.asc,id.asc",
                "offset": offset,
                "limit": page_size,
            },
        )

        if not rows:
            break

        all_matches.extend(rows)

        print(
            f"  Página recibida: {len(rows)}"
        )

        if len(rows) < page_size:
            break

        offset += page_size

    print(
        f"Partidos recibidos: "
        f"{len(all_matches)}"
    )

    return all_matches


# ============================================================
# NORMALIZAR PARTIDOS
# ============================================================

def normalize_matches(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    normalized: list[dict[str, Any]] = []

    for row in rows:

        match_id = safe_int(row.get("id"))

        home_team_id = safe_int(
            row.get("home_team_id")
        )

        away_team_id = safe_int(
            row.get("away_team_id")
        )

        home_goals = safe_int(
            row.get("home_goals")
        )

        away_goals = safe_int(
            row.get("away_goals")
        )

        starting_at = clean_text(
            row.get("starting_at")
        )

        status = clean_text(
            row.get("status")
        )

        if match_id is None:
            continue

        if home_team_id is None:
            continue

        if away_team_id is None:
            continue

        if home_goals is None:
            continue

        if away_goals is None:
            continue

        if not starting_at:
            continue

        if not is_finished_status(status):
            continue

        if home_goals > away_goals:
            result = "H"

        elif home_goals < away_goals:
            result = "A"

        else:
            result = "D"

        normalized.append(
            {
                "id": match_id,
                "starting_at": starting_at,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
            }
        )

    normalized.sort(
        key=lambda item: (
            item["starting_at"],
            item["id"],
        )
    )

    return normalized


# ============================================================
# HISTORIAL DE EQUIPOS
# ============================================================

def empty_team_history() -> list[dict[str, Any]]:
    return []


def get_team_history(
    histories: dict[int, list[dict[str, Any]]],
    team_id: int,
) -> list[dict[str, Any]]:

    if team_id not in histories:
        histories[team_id] = empty_team_history()

    return histories[team_id]


def calculate_team_points(
    team_match: dict[str, Any],
    team_id: int,
) -> int:

    home_team_id = team_match["home_team_id"]
    away_team_id = team_match["away_team_id"]

    home_goals = team_match["home_goals"]
    away_goals = team_match["away_goals"]

    if home_team_id == team_id:

        if home_goals > away_goals:
            return 3

        if home_goals == away_goals:
            return 1

        return 0

    if away_team_id == team_id:

        if away_goals > home_goals:
            return 3

        if away_goals == home_goals:
            return 1

        return 0

    return 0


def build_form_features(
    history: list[dict[str, Any]],
) -> Optional[dict[str, float]]:

    if len(history) < LOOKBACK_MATCHES:
        return None

    recent = history[-LOOKBACK_MATCHES:]

    goals_for = 0
    goals_against = 0
    points = 0

    for match in recent:

        goals_for += match["goals_for"]
        goals_against += match["goals_against"]
        points += match["points"]

    return {
        "goals_for": float(goals_for),
        "goals_against": float(goals_against),
        "points": float(points),
    }


# ============================================================
# CONSTRUIR DATASET SIN DATA LEAKAGE
# ============================================================

def build_dataset(
    matches: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, int]:

    print("")
    print("=== CONSTRUYENDO DATASET ===")

    histories: dict[
        int,
        list[dict[str, Any]]
    ] = {}

    X: list[list[float]] = []
    y: list[str] = []

    discarded = 0

    for match in matches:

        home_team_id = match["home_team_id"]
        away_team_id = match["away_team_id"]

        home_history = get_team_history(
            histories,
            home_team_id,
        )

        away_history = get_team_history(
            histories,
            away_team_id,
        )

        # ----------------------------------------------------
        # MUY IMPORTANTE:
        #
        # Las características se calculan ANTES de agregar
        # el partido actual al historial.
        #
        # Así evitamos DATA LEAKAGE.
        # ----------------------------------------------------

        home_form = build_form_features(
            home_history
        )

        away_form = build_form_features(
            away_history
        )

        if (
            home_form is None
            or away_form is None
        ):
            discarded += 1

        else:

            features = [
                home_form["goals_for"],
                home_form["goals_against"],
                home_form["points"],
                away_form["goals_for"],
                away_form["goals_against"],
                away_form["points"],
                (
                    home_form["goals_for"]
                    - home_form["goals_against"]
                )
                -
                (
                    away_form["goals_for"]
                    - away_form["goals_against"]
                ),
                (
                    home_form["points"]
                    - away_form["points"]
                ),
                1.0,
            ]

            X.append(features)
            y.append(match["result"])

        # ----------------------------------------------------
        # AHORA SÍ agregamos el partido al historial.
        # ----------------------------------------------------

        home_history.append(
            {
                "goals_for": match["home_goals"],
                "goals_against": match["away_goals"],
                "points": calculate_team_points(
                    match,
                    home_team_id,
                ),
            }
        )

        away_history.append(
            {
                "goals_for": match["away_goals"],
                "goals_against": match["home_goals"],
                "points": calculate_team_points(
                    match,
                    away_team_id,
                ),
            }
        )

    print(
        f"Ejemplos utilizables: {len(X)}"
    )

    print(
        f"Partidos descartados por falta "
        f"de historial: {discarded}"
    )

    if not X:
        raise RuntimeError(
            "No se pudieron construir ejemplos "
            "utilizables para entrenar."
        )

    return (
        np.asarray(X, dtype=float),
        np.asarray(y),
        discarded,
    )


# ============================================================
# VALIDAR DATASET
# ============================================================

def validate_dataset(
    y: np.ndarray,
) -> None:

    classes, counts = np.unique(
        y,
        return_counts=True,
    )

    print("")
    print("=== DISTRIBUCIÓN DE RESULTADOS ===")

    for class_name, count in zip(
        classes,
        counts,
    ):
        print(
            f"  {class_name}: {count}"
        )

    required = {"H", "D", "A"}

    available = set(
        classes.tolist()
    )

    missing = required - available

    if missing:
        raise RuntimeError(
            "El dataset no contiene las tres "
            f"clases H/D/A. Faltan: {sorted(missing)}"
        )


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train_model(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[
    LogisticRegression,
    StandardScaler,
    float,
    float,
    int,
    int,
]:

    total = len(X)

    if total < 100:
        raise RuntimeError(
            "Hay muy pocos ejemplos para entrenar "
            f"de forma fiable: {total}"
        )

    split_index = int(
        total * TRAIN_RATIO
    )

    if split_index <= 0:
        raise RuntimeError(
            "El conjunto de entrenamiento quedó vacío."
        )

    if split_index >= total:
        split_index = total - 1

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_validation = X[split_index:]
    y_validation = y[split_index:]

    print("")
    print("=== VALIDACION DEL MODELO ===")

    print(
        f"Entrenamiento: {len(X_train)}"
    )

    print(
        f"Validacion: {len(X_validation)}"
    )

    train_classes = set(
        y_train.tolist()
    )

    if not {"H", "D", "A"}.issubset(
        train_classes
    ):
        raise RuntimeError(
            "El conjunto de entrenamiento "
            "no contiene H, D y A."
        )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        C=1.0,
        random_state=42,
        multi_class="auto",
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    probabilities = model.predict_proba(
        X_validation_scaled
    )

    predictions = model.predict(
        X_validation_scaled
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    validation_log_loss = log_loss(
        y_validation,
        probabilities,
        labels=model.classes_,
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Log loss: {validation_log_loss:.4f}"
    )

    return (
        model,
        scaler,
        float(accuracy),
        float(validation_log_loss),
        len(X_train),
        len(X_validation),
    )


# ============================================================
# MODELO ACTIVO ANTERIOR
# ============================================================

def load_active_model() -> Optional[dict[str, Any]]:

    rows = supabase_get(
        "model_versions",
        {
            "select": (
                "version,"
                "model_name,"
                "trained_at,"
                "training_matches,"
                "metrics,"
                "active"
            ),
            "active": "eq.true",
            "limit": 1,
        },
    )

    if not rows:
        return None

    return rows[0]


def get_previous_log_loss(
    active_model: Optional[dict[str, Any]],
) -> Optional[float]:

    if not active_model:
        return None

    metrics = active_model.get(
        "metrics"
    )

    if not isinstance(metrics, dict):
        return None

    value = metrics.get(
        "validation_log_loss"
    )

    if value is None:
        value = metrics.get(
            "log_loss"
        )

    return safe_float(value)


# ============================================================
# SERIALIZAR MODELO
# ============================================================

def serialize_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    accuracy: float,
    validation_log_loss: float,
    training_matches: int,
    validation_matches: int,
    total_dataset_matches: int,
    discarded_matches: int,
) -> dict[str, Any]:

    coefficients = (
        model.coef_.tolist()
    )

    intercept = (
        model.intercept_.tolist()
    )

    classes = [
        str(value)
        for value in model.classes_.tolist()
    ]

    scaler_mean = [
        float(value)
        for value in scaler.mean_.tolist()
    ]

    scaler_scale = [
        float(value)
        for value in scaler.scale_.tolist()
    ]

    return {
        "model_type": (
            "multinomial_logistic_regression"
        ),
        "model_name": MODEL_NAME,

        "feature_names": FEATURE_NAMES,

        "classes": classes,

        "coefficients": coefficients,

        "intercept": intercept,

        "scaler_mean": scaler_mean,

        "scaler_scale": scaler_scale,

        "validation_accuracy": accuracy,

        "validation_log_loss": (
            validation_log_loss
        ),

        "training_matches": (
            training_matches
        ),

        "validation_matches": (
            validation_matches
        ),

        "dataset_matches": (
            total_dataset_matches
        ),

        "discarded_matches": (
            discarded_matches
        ),

        "trained_at": utc_now(),
    }


# ============================================================
# GUARDAR MODELO
# ============================================================

def save_model_version(
    version: str,
    metrics: dict[str, Any],
    active: bool,
) -> None:

    payload = {
        "version": version,
        "model_name": MODEL_NAME,
        "trained_at": utc_now(),
        "training_matches": (
            metrics["training_matches"]
        ),
        "metrics": metrics,
        "active": active,
    }

    supabase_upsert(
        "model_versions",
        payload,
        on_conflict="version",
    )


# ============================================================
# DESACTIVAR MODELOS ANTERIORES
# ============================================================

def deactivate_existing_models() -> None:

    print(
        "Desactivando modelo anterior..."
    )

    supabase_patch(
        "model_versions",
        {
            "active": "eq.true",
        },
        {
            "active": False,
        },
    )


# ============================================================
# ENTRENAMIENTO PRINCIPAL
# ============================================================

def main() -> None:

    print("")
    print("========================================")
    print("FUTBOL IA - ENTRENAMIENTO REAL")
    print("========================================")

    print(
        f"Modelo: {MODEL_NAME}"
    )

    print(
        f"Historial utilizado: "
        f"últimos {LOOKBACK_MATCHES} partidos"
    )

    print(
        "División temporal: "
        f"{int(TRAIN_RATIO * 100)}% / "
        f"{int((1 - TRAIN_RATIO) * 100)}%"
    )

    # --------------------------------------------------------
    # VALIDAR CONFIGURACIÓN
    # --------------------------------------------------------

    get_supabase_config()

    # --------------------------------------------------------
    # CARGAR DATOS
    # --------------------------------------------------------

    raw_matches = (
        load_all_finished_matches()
    )

    if not raw_matches:
        raise RuntimeError(
            "No existen partidos terminados "
            "en Supabase."
        )

    matches = normalize_matches(
        raw_matches
    )

    print(
        f"Partidos válidos: {len(matches)}"
    )

    if len(matches) < 100:
        raise RuntimeError(
            "Hay menos de 100 partidos válidos. "
            "No se realizará el entrenamiento."
        )

    # --------------------------------------------------------
    # CONSTRUIR FEATURES
    # --------------------------------------------------------

    X, y, discarded = (
        build_dataset(matches)
    )

    # --------------------------------------------------------
    # VALIDAR CLASES
    # --------------------------------------------------------

    validate_dataset(y)

    # --------------------------------------------------------
    # ENTRENAR
    # --------------------------------------------------------

    (
        model,
        scaler,
        accuracy,
        validation_log_loss,
        training_matches,
        validation_matches,
    ) = train_model(
        X,
        y,
    )

    # --------------------------------------------------------
    # MODELO ACTUAL
    # --------------------------------------------------------

    active_model = (
        load_active_model()
    )

    previous_log_loss = (
        get_previous_log_loss(
            active_model
        )
    )

    # --------------------------------------------------------
    # DECIDIR ACTIVACIÓN
    # --------------------------------------------------------

    if previous_log_loss is None:
        should_activate = True

    else:
        should_activate = (
            validation_log_loss
            <= previous_log_loss
        )

    # --------------------------------------------------------
    # CREAR VERSIÓN
    # --------------------------------------------------------

    version = (
        "1X2-"
        + datetime.now(
            timezone.utc
        ).strftime("%Y%m%d-%H%M%S")
    )

    metrics = serialize_model(
        model=model,
        scaler=scaler,
        accuracy=accuracy,
        validation_log_loss=(
            validation_log_loss
        ),
        training_matches=(
            training_matches
        ),
        validation_matches=(
            validation_matches
        ),
        total_dataset_matches=len(
            matches
        ),
        discarded_matches=discarded,
    )

    # --------------------------------------------------------
    # GUARDAR NUEVO MODELO
    # --------------------------------------------------------

    save_model_version(
        version=version,
        metrics=metrics,
        active=False,
    )

    # --------------------------------------------------------
    # ACTIVAR SI MEJORA
    # --------------------------------------------------------

    if should_activate:

        deactivate_existing_models()

        supabase_patch(
            "model_versions",
            {
                "version": f"eq.{version}",
            },
            {
                "active": True,
            },
        )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    print("")
    print("=== MODELO GUARDADO ===")

    print(
        f"Version: {version}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Log loss: {validation_log_loss:.4f}"
    )

    if previous_log_loss is None:
        print(
            "Modelo anterior: ninguno"
        )
    else:
        print(
            "Modelo anterior: "
            f"{previous_log_loss}"
        )

    print(
        f"Activado: {should_activate}"
    )

    print(
        f"Partidos del dataset: "
        f"{len(matches)}"
    )

    print(
        f"Ejemplos utilizables: "
        f"{len(X)}"
    )

    print(
        f"Entrenamiento: "
        f"{training_matches}"
    )

    print(
        f"Validacion: "
        f"{validation_matches}"
    )

    print("")
    print("========================================")
    print("ENTRENAMIENTO FINALIZADO")
    print("========================================")

    result = {
        "version": version,
        "active": should_activate,
        "accuracy": accuracy,
        "log_loss": validation_log_loss,
        "previous_log_loss": (
            previous_log_loss
        ),
        "dataset_matches": len(matches),
        "usable_examples": len(X),
        "discarded_matches": discarded,
        "training_matches": training_matches,
        "validation_matches": validation_matches,
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
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
