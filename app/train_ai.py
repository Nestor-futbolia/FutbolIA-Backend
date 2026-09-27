import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler


APP_NAME = "Fútbol IA"
MODEL_NAME = "FutbolIA-1X2-LogisticRegression"

# IMPORTANTE:
# Este script NO llama a API-Football.
# Entrena exclusivamente con el historial que ya existe en Supabase.
# Así no consume la cuota diaria de la API de fútbol.

PAGE_SIZE = 1000
REQUEST_TIMEOUT = 30

VALID_FINISHED_STATUSES = {"FT", "AET", "PEN"}

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


class SupabaseError(RuntimeError):
    pass


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta la variable de entorno: {name}")
    return value


SUPABASE_URL = require_env("SUPABASE_URL").rstrip("/")
SUPABASE_SECRET_KEY = require_env("SUPABASE_SECRET_KEY")


session = requests.Session()
session.headers.update(
    {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Content-Type": "application/json",
    }
)


def supabase_request(
    method: str,
    path: str,
    *,
    params: Dict[str, Any] | None = None,
    payload: Any | None = None,
    headers: Dict[str, str] | None = None,
) -> requests.Response:
    url = f"{SUPABASE_URL}/rest/v1/{path.lstrip('/')}"
    merged_headers = dict(session.headers)
    if headers:
        merged_headers.update(headers)

    try:
        response = session.request(
            method,
            url,
            params=params,
            json=payload,
            headers=merged_headers,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise SupabaseError(f"Error de conexión con Supabase: {exc}") from exc

    if not response.ok:
        body = response.text[:2000]
        raise SupabaseError(
            f"Supabase {method} {path}: HTTP {response.status_code}: {body}"
        )

    return response


def fetch_all_finished_matches() -> List[Dict[str, Any]]:
    """Lee TODOS los partidos finalizados disponibles en Supabase por páginas."""
    all_rows: List[Dict[str, Any]] = []
    offset = 0

    print("=== CARGANDO HISTORIAL DESDE SUPABASE ===")
    print("Fuente: public.matches")
    print("API-Football: NO SE USA EN ESTE PASO")

    while True:
        params = {
            "select": (
                "id,league_id,season_id,home_team_id,away_team_id,"
                "starting_at,status,home_goals,away_goals"
            ),
            "status": "in.(FT,AET,PEN)",
            "order": "starting_at.asc,id.asc",
            "limit": PAGE_SIZE,
            "offset": offset,
        }

        response = supabase_request("GET", "matches", params=params)
        try:
            rows = response.json()
        except ValueError as exc:
            raise SupabaseError("Supabase devolvió JSON inválido al leer matches") from exc

        if not isinstance(rows, list):
            raise SupabaseError("La respuesta de Supabase para matches no es una lista")

        if not rows:
            break

        all_rows.extend(rows)
        print(f"Página: offset={offset}, filas={len(rows)}, acumuladas={len(all_rows)}")

        if len(rows) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return all_rows


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_match(row: Dict[str, Any]) -> Dict[str, Any] | None:
    home_team_id = parse_int(row.get("home_team_id"))
    away_team_id = parse_int(row.get("away_team_id"))
    home_goals = parse_int(row.get("home_goals"))
    away_goals = parse_int(row.get("away_goals"))
    starting_at = str(row.get("starting_at") or "").strip()
    status = str(row.get("status") or "").strip().upper()

    if not home_team_id or not away_team_id:
        return None
    if home_team_id == away_team_id:
        return None
    if home_goals is None or away_goals is None:
        return None
    if status not in VALID_FINISHED_STATUSES:
        return None
    if not starting_at:
        return None

    return {
        "id": parse_int(row.get("id"), 0),
        "league_id": parse_int(row.get("league_id")),
        "season_id": parse_int(row.get("season_id")),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "starting_at": starting_at,
        "status": status,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


def build_features_and_labels(
    rows: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Construye features usando solo partidos PREVIOS al partido objetivo.
    Después de crear cada ejemplo, actualiza el historial con ese resultado.
    Esto evita data leakage.
    """
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        match = normalize_match(row)
        if match is not None:
            normalized.append(match)

    # Supabase ya entrega por fecha, pero ordenamos otra vez por seguridad.
    normalized.sort(key=lambda m: (m["starting_at"], m["id"]))

    # team_id -> últimos 5 partidos como lista de (gf, ga, puntos)
    history: Dict[int, List[Tuple[float, float, float]]] = {}

    X: List[List[float]] = []
    y: List[str] = []
    meta: List[Dict[str, Any]] = []

    discarded_no_history = 0

    for match in normalized:
        home_id = match["home_team_id"]
        away_id = match["away_team_id"]

        home_history = history.get(home_id, [])[-5:]
        away_history = history.get(away_id, [])[-5:]

        # Para producir un ejemplo comparable, exigimos al menos 5 partidos
        # previos de cada equipo.
        if len(home_history) < 5 or len(away_history) < 5:
            discarded_no_history += 1
        else:
            home_gf = sum(item[0] for item in home_history)
            home_ga = sum(item[1] for item in home_history)
            home_points = sum(item[2] for item in home_history)

            away_gf = sum(item[0] for item in away_history)
            away_ga = sum(item[1] for item in away_history)
            away_points = sum(item[2] for item in away_history)

            features = [
                home_gf,
                home_ga,
                home_points,
                away_gf,
                away_ga,
                away_points,
                (home_gf - home_ga) - (away_gf - away_ga),
                home_points - away_points,
                1.0,
            ]

            if match["home_goals"] > match["away_goals"]:
                label = "H"
            elif match["home_goals"] < match["away_goals"]:
                label = "A"
            else:
                label = "D"

            X.append(features)
            y.append(label)
            meta.append(
                {
                    "match_id": match["id"],
                    "starting_at": match["starting_at"],
                    "league_id": match["league_id"],
                    "season_id": match["season_id"],
                }
            )

        # IMPORTANTE: actualizar DESPUÉS de crear las features.
        if match["home_goals"] > match["away_goals"]:
            home_points_value = 3.0
            away_points_value = 0.0
        elif match["home_goals"] < match["away_goals"]:
            home_points_value = 0.0
            away_points_value = 3.0
        else:
            home_points_value = 1.0
            away_points_value = 1.0

        home_history = history.setdefault(home_id, [])
        home_history.append(
            (
                float(match["home_goals"]),
                float(match["away_goals"]),
                home_points_value,
            )
        )
        if len(home_history) > 5:
            del home_history[:-5]

        away_history = history.setdefault(away_id, [])
        away_history.append(
            (
                float(match["away_goals"]),
                float(match["home_goals"]),
                away_points_value,
            )
        )
        if len(away_history) > 5:
            del away_history[:-5]

    print(f"Partidos válidos: {len(normalized)}")
    print(f"Ejemplos utilizables: {len(X)}")
    print(f"Partidos descartados por falta de historial: {discarded_no_history}")

    if not X:
        raise RuntimeError(
            "No hay ejemplos utilizables. Necesitamos al menos 5 partidos previos "
            "para cada equipo en una parte de la historia."
        )

    return np.asarray(X, dtype=float), np.asarray(y), meta


def train_model(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    if len(X) < 50:
        raise RuntimeError(
            f"Solo hay {len(X)} ejemplos utilizables. Se requieren al menos 50 para entrenar."
        )

    # División temporal: primeros 80% para entrenamiento, últimos 20% para validación.
    split_index = int(len(X) * 0.80)
    if split_index <= 0 or split_index >= len(X):
        raise RuntimeError("No se pudo construir una división temporal 80/20 válida.")

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_valid = X[split_index:]
    y_valid = y[split_index:]

    unique_train = set(y_train.tolist())
    if unique_train != {"H", "D", "A"}:
        raise RuntimeError(
            f"El conjunto de entrenamiento no contiene las 3 clases H/D/A. Clases: {sorted(unique_train)}"
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=1.0,
        random_state=42,
        multi_class="auto",
    )
    model.fit(X_train_scaled, y_train)

    valid_probabilities = model.predict_proba(X_valid_scaled)
    valid_predictions = model.predict(X_valid_scaled)

    accuracy = float(accuracy_score(y_valid, valid_predictions))
    multiclass_log_loss = float(
        log_loss(y_valid, valid_probabilities, labels=model.classes_)
    )

    print("=== VALIDACIÓN DEL MODELO ===")
    print(f"Entrenamiento: {len(X_train)}")
    print(f"Validación: {len(X_valid)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log loss: {multiclass_log_loss:.4f}")

    # Copia de todos los elementos necesarios para reproducir la inferencia en Android/API.
    serialized = {
        "model_type": "multinomial_logistic_regression",
        "feature_names": FEATURE_NAMES,
        "features": FEATURE_NAMES,
        "classes": [str(value) for value in model.classes_.tolist()],
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "validation_accuracy": accuracy,
        "validation_log_loss": multiclass_log_loss,
        "training_examples": int(len(X_train)),
        "validation_examples": int(len(X_valid)),
    }

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy,
        "log_loss": multiclass_log_loss,
        "serialized": serialized,
        "training_examples": len(X_train),
        "validation_examples": len(X_valid),
    }


def get_active_model_metric() -> float | None:
    params = {
        "select": "version,metrics,active",
        "active": "eq.true",
        "limit": 1,
    }
    response = supabase_request("GET", "model_versions", params=params)
    try:
        rows = response.json()
    except ValueError as exc:
        raise SupabaseError("JSON inválido al leer model_versions") from exc

    if not isinstance(rows, list) or not rows:
        return None

    metrics = rows[0].get("metrics") or {}
    if not isinstance(metrics, dict):
        return None

    value = metrics.get("validation_log_loss")
    return parse_float(value)


def save_model(result: Dict[str, Any]) -> Tuple[str, bool, float | None]:
    now = datetime.now(timezone.utc)
    version = f"1X2-{now.strftime('%Y%m%d-%H%M%S')}"

    previous_metric = get_active_model_metric()
    new_metric = float(result["log_loss"])

    # Activar automáticamente el primer modelo.
    should_activate = previous_metric is None or new_metric <= previous_metric

    metrics = dict(result["serialized"])
    metrics.update(
        {
            "validation_accuracy": float(result["accuracy"]),
            "validation_log_loss": new_metric,
            "trained_at": now.isoformat(),
            "training_matches": int(result["training_examples"]),
            "validation_examples": int(result["validation_examples"]),
            "source": "supabase_existing_history_only",
            "api_football_calls": 0,
        }
    )

    payload = {
        "version": version,
        "model_name": MODEL_NAME,
        "trained_at": now.isoformat(),
        "training_matches": int(result["training_examples"]),
        "metrics": metrics,
        "active": bool(should_activate),
    }

    # Si será el nuevo modelo activo, desactivamos los anteriores primero.
    if should_activate:
        supabase_request(
            "PATCH",
            "model_versions",
            params={"active": "eq.true"},
            payload={"active": False},
        )

    response = supabase_request(
        "POST",
        "model_versions",
        params={"on_conflict": "version"},
        payload=payload,
        headers={"Prefer": "resolution=merge-duplicates,return=representation"},
    )

    if not response.ok:
        raise SupabaseError(f"No se pudo guardar el modelo: HTTP {response.status_code}")

    print("=== MODELO GUARDADO ===")
    print(f"Version: {version}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Log loss: {new_metric:.4f}")
    if previous_metric is None:
        print("Modelo anterior: ninguno")
    else:
        print(f"Modelo anterior (log loss): {previous_metric:.10f}")
    print(f"Activado: {should_activate}")

    return version, should_activate, previous_metric


def main() -> int:
    print(f"{APP_NAME.upper()} - ENTRENAMIENTO REAL")
    print("====================================")

    rows = fetch_all_finished_matches()
    print(f"Filas recibidas desde Supabase: {len(rows)}")

    if not rows:
        raise RuntimeError(
            "Supabase no tiene partidos finalizados. Primero hay que cargar historial en public.matches."
        )

    X, y, meta = build_features_and_labels(rows)
    print(f"Features: {X.shape[1]}")
    print(f"Ejemplos totales: {len(X)}")
    print(f"Primera fecha utilizable: {meta[0]['starting_at']}")
    print(f"Última fecha utilizable: {meta[-1]['starting_at']}")

    result = train_model(X, y)
    save_model(result)

    print("====================================")
    print("ENTRENAMIENTO TERMINADO CORRECTAMENTE")
    print("Sin llamadas a API-Football durante el entrenamiento")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR DE ENTRENAMIENTO:", exc, file=sys.stderr)
        raise
