import json
import math
import os
import sys
from collections import defaultdict, deque
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

SUPABASE_PAGE_SIZE = 1000
FORM_MATCHES = 5

VALID_STATUSES = {
    "FT",
    "AET",
    "PEN",
}

MODEL_NAME = "FutbolIA-1X2-LogisticRegression"

RANDOM_STATE = 42


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

def required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Falta la variable de entorno {name}."
        )

    return value.strip()


def get_supabase_config() -> tuple[str, str]:
    url = required_env(
        "SUPABASE_URL"
    ).rstrip("/")

    key = required_env(
        "SUPABASE_SECRET_KEY"
    )

    return url, key


def supabase_headers(
    key: str,
    extra: Optional[dict[str, str]] = None,
) -> dict[str, str]:

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    if extra:
        headers.update(extra)

    return headers


# ============================================================
# UTILIDADES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def as_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        if value is None:
            return default

        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def as_int(
    value: Any,
) -> Optional[int]:

    try:
        if value is None:
            return None

        return int(value)

    except (TypeError, ValueError):
        return None


def parse_date(
    value: Any,
) -> Optional[datetime]:

    if not value:
        return None

    text = str(value).strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(
            timezone.utc
        )

    except ValueError:
        return None


# ============================================================
# RESULTADO DEL PARTIDO
# ============================================================

def match_result(
    row: dict[str, Any],
) -> Optional[str]:

    home_goals = as_int(
        row.get("home_goals")
    )

    away_goals = as_int(
        row.get("away_goals")
    )

    if (
        home_goals is None
        or away_goals is None
    ):
        return None

    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


# ============================================================
# CARGAR TODO EL HISTORIAL DE SUPABASE
# ============================================================

def supabase_get_all_matches() -> list[dict[str, Any]]:

    supabase_url, supabase_key = (
        get_supabase_config()
    )

    url = (
        f"{supabase_url}"
        "/rest/v1/matches"
    )

    all_rows: list[
        dict[str, Any]
    ] = []

    offset = 0

    print("")
    print("========================================")
    print("CARGANDO TODO EL HISTORIAL")
    print("========================================")

    while True:

        start = offset

        end = (
            offset
            + SUPABASE_PAGE_SIZE
            - 1
        )

        params = {
            "select": "*",
            "status": "in.(FT,AET,PEN)",
            "starting_at": "not.is.null",
            "order": "starting_at.asc,id.asc",
        }

        headers = supabase_headers(
            supabase_key,
            {
                "Range": (
                    f"{start}-{end}"
                ),
                "Prefer": "count=exact",
            },
        )

        print("")
        print(
            f"Solicitando historial "
            f"{start}-{end}..."
        )

        try:

            with httpx.Client(
                timeout=60
            ) as client:

                response = client.get(
                    url,
                    params=params,
                    headers=headers,
                )

        except Exception as exc:

            raise RuntimeError(
                "Error de conexión con "
                f"Supabase: {exc}"
            ) from exc

        if response.status_code >= 400:

            raise RuntimeError(
                "Supabase GET matches: "
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )

        try:

            rows = response.json()

        except Exception as exc:

            raise RuntimeError(
                "Supabase devolvió "
                "JSON inválido."
            ) from exc

        if not isinstance(
            rows,
            list,
        ):

            raise RuntimeError(
                "La respuesta de Supabase "
                "no es una lista."
            )

        received = len(rows)

        all_rows.extend(rows)

        print(
            f"Partidos recibidos en esta página: "
            f"{received}"
        )

        print(
            f"Total acumulado: "
            f"{len(all_rows)}"
        )

        # Si recibimos menos de 1000,
        # ya llegamos al final.
        if received < SUPABASE_PAGE_SIZE:
            break

        offset += SUPABASE_PAGE_SIZE

    print("")
    print(
        "TOTAL DE PARTIDOS HISTÓRICOS: "
        f"{len(all_rows)}"
    )

    return all_rows


# ============================================================
# ESTADO HISTÓRICO DE CADA EQUIPO
# ============================================================

def create_team_state() -> dict[str, deque]:

    return {
        "gf": deque(
            maxlen=FORM_MATCHES
        ),
        "ga": deque(
            maxlen=FORM_MATCHES
        ),
        "points": deque(
            maxlen=FORM_MATCHES
        ),
    }


# ============================================================
# CONSTRUIR DATASET
# ============================================================

def build_dataset(
    rows: list[dict[str, Any]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
]:

    prepared = []

    for row in rows:

        match_id = as_int(
            row.get("id")
        )

        home_team_id = as_int(
            row.get("home_team_id")
        )

        away_team_id = as_int(
            row.get("away_team_id")
        )

        starting_at = parse_date(
            row.get("starting_at")
        )

        result = match_result(
            row
        )

        if (
            match_id is None
            or home_team_id is None
            or away_team_id is None
            or starting_at is None
            or result is None
        ):
            continue

        prepared.append(
            (
                starting_at,
                match_id,
                row,
                home_team_id,
                away_team_id,
                result,
            )
        )

    # Orden cronológico.
    #
    # Esto es fundamental para evitar
    # utilizar información del futuro.
    prepared.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    team_states: dict[
        int,
        dict[str, deque],
    ] = defaultdict(
        create_team_state
    )

    X: list[list[float]] = []

    y: list[str] = []

    skipped_no_history = 0

    for (
        starting_at,
        match_id,
        row,
        home_team_id,
        away_team_id,
        result,
    ) in prepared:

        home_state = team_states[
            home_team_id
        ]

        away_state = team_states[
            away_team_id
        ]

        # ----------------------------------------------------
        # IMPORTANTE:
        # Solo utilizamos información ANTERIOR al partido.
        # ----------------------------------------------------

        home_has_history = (
            len(home_state["gf"])
            >= FORM_MATCHES
        )

        away_has_history = (
            len(away_state["gf"])
            >= FORM_MATCHES
        )

        if (
            not home_has_history
            or not away_has_history
        ):

            skipped_no_history += 1

        else:

            home_goals_for = float(
                sum(
                    home_state["gf"]
                )
            )

            home_goals_against = float(
                sum(
                    home_state["ga"]
                )
            )

            home_points = float(
                sum(
                    home_state["points"]
                )
            )

            away_goals_for = float(
                sum(
                    away_state["gf"]
                )
            )

            away_goals_against = float(
                sum(
                    away_state["ga"]
                )
            )

            away_points = float(
                sum(
                    away_state["points"]
                )
            )

            goals_form_difference = (
                (
                    home_goals_for
                    - home_goals_against
                )
                -
                (
                    away_goals_for
                    - away_goals_against
                )
            )

            points_form_difference = (
                home_points
                - away_points
            )

            features = [
                home_goals_for,
                home_goals_against,
                home_points,
                away_goals_for,
                away_goals_against,
                away_points,
                goals_form_difference,
                points_form_difference,
                1.0,
            ]

            X.append(
                features
            )

            y.append(
                result
            )

        # ----------------------------------------------------
        # AHORA actualizamos el historial.
        #
        # Esto ocurre DESPUÉS de crear las features,
        # evitando fuga de información.
        # ----------------------------------------------------

        home_goals = as_int(
            row.get("home_goals")
        )

        away_goals = as_int(
            row.get("away_goals")
        )

        if (
            home_goals is None
            or away_goals is None
        ):
            continue

        home_state["gf"].append(
            float(home_goals)
        )

        home_state["ga"].append(
            float(away_goals)
        )

        away_state["gf"].append(
            float(away_goals)
        )

        away_state["ga"].append(
            float(home_goals)
        )

        if result == "H":

            home_state["points"].append(
                3.0
            )

            away_state["points"].append(
                0.0
            )

        elif result == "D":

            home_state["points"].append(
                1.0
            )

            away_state["points"].append(
                1.0
            )

        else:

            home_state["points"].append(
                0.0
            )

            away_state["points"].append(
                3.0
            )

    if not X:

        raise RuntimeError(
            "No hay suficientes partidos "
            "con historial previo para "
            "crear el dataset."
        )

    return (
        np.asarray(
            X,
            dtype=np.float64,
        ),
        np.asarray(
            y,
        ),
        skipped_no_history,
    )


# ============================================================
# MODELO ACTIVO ACTUAL
# ============================================================

def get_active_model() -> Optional[
    dict[str, Any]
]:

    supabase_url, supabase_key = (
        get_supabase_config()
    )

    url = (
        f"{supabase_url}"
        "/rest/v1/model_versions"
    )

    params = {
        "select": (
            "version,"
            "model_name,"
            "trained_at,"
            "training_matches,"
            "metrics,"
            "active"
        ),
        "active": "eq.true",
        "limit": "1",
    }

    try:

        with httpx.Client(
            timeout=60
        ) as client:

            response = client.get(
                url,
                params=params,
                headers=supabase_headers(
                    supabase_key
                ),
            )

    except Exception as exc:

        raise RuntimeError(
            "Error consultando el "
            f"modelo activo: {exc}"
        ) from exc

    if response.status_code >= 400:

        raise RuntimeError(
            "Supabase GET "
            "model_versions: "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    rows = response.json()

    if not rows:
        return None

    return rows[0]


# ============================================================
# GUARDAR MODELO
# ============================================================

def save_model(
    payload: dict[str, Any],
) -> None:

    supabase_url, supabase_key = (
        get_supabase_config()
    )

    url = (
        f"{supabase_url}"
        "/rest/v1/model_versions"
    )

    headers = supabase_headers(
        supabase_key,
        {
            "Prefer": (
                "resolution=merge-duplicates,"
                "return=minimal"
            ),
        },
    )

    try:

        with httpx.Client(
            timeout=60
        ) as client:

            response = client.post(
                url,
                headers=headers,
                json=payload,
            )

    except Exception as exc:

        raise RuntimeError(
            "Error guardando el modelo "
            f"en Supabase: {exc}"
        ) from exc

    if response.status_code >= 400:

        raise RuntimeError(
            "Supabase POST "
            "model_versions: "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )


# ============================================================
# DESACTIVAR MODELOS ANTERIORES
# ============================================================

def deactivate_all_models() -> None:

    supabase_url, supabase_key = (
        get_supabase_config()
    )

    url = (
        f"{supabase_url}"
        "/rest/v1/model_versions"
    )

    headers = supabase_headers(
        supabase_key,
        {
            "Prefer": "return=minimal"
        },
    )

    params = {
        "active": "eq.true"
    }

    try:

        with httpx.Client(
            timeout=60
        ) as client:

            response = client.patch(
                url,
                params=params,
                headers=headers,
                json={
                    "active": False
                },
            )

    except Exception as exc:

        raise RuntimeError(
            "Error desactivando "
            f"modelos anteriores: {exc}"
        ) from exc

    if response.status_code >= 400:

        raise RuntimeError(
            "Supabase PATCH "
            "model_versions: "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )


# ============================================================
# ACTIVAR UNA VERSIÓN CONCRETA
# ============================================================

def activate_model(
    version: str,
) -> None:

    supabase_url, supabase_key = (
        get_supabase_config()
    )

    url = (
        f"{supabase_url}"
        "/rest/v1/model_versions"
    )

    headers = supabase_headers(
        supabase_key,
        {
            "Prefer": "return=minimal"
        },
    )

    params = {
        "version": f"eq.{version}"
    }

    try:

        with httpx.Client(
            timeout=60
        ) as client:

            response = client.patch(
                url,
                params=params,
                headers=headers,
                json={
                    "active": True
                },
            )

    except Exception as exc:

        raise RuntimeError(
            "Error activando el "
            f"modelo {version}: {exc}"
        ) from exc

    if response.status_code >= 400:

        raise RuntimeError(
            "Supabase PATCH modelo "
            f"{version}: "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )


# ============================================================
# ENTRENAMIENTO
# ============================================================

def main() -> None:

    print("========================================")
    print("FUTBOL IA - ENTRENAMIENTO REAL")
    print("========================================")

    # --------------------------------------------------------
    # CARGAR TODO EL HISTORIAL
    # --------------------------------------------------------

    rows = (
        supabase_get_all_matches()
    )

    if len(rows) < 100:

        raise RuntimeError(
            f"Solo hay {len(rows)} "
            "partidos históricos. "
            "Se necesitan al menos 100."
        )

    # --------------------------------------------------------
    # CONSTRUIR DATASET
    # --------------------------------------------------------

    print("")
    print("========================================")
    print("CONSTRUYENDO DATASET")
    print("========================================")

    (
        X,
        y,
        skipped_no_history,
    ) = build_dataset(
        rows
    )

    print("")
    print(
        f"Partidos recibidos: "
        f"{len(rows)}"
    )

    print(
        f"Ejemplos utilizables: "
        f"{len(X)}"
    )

    print(
        f"Partidos descartados por "
        f"falta de historial: "
        f"{skipped_no_history}"
    )

    # --------------------------------------------------------
    # DISTRIBUCIÓN DE RESULTADOS
    # --------------------------------------------------------

    classes_found, counts = (
        np.unique(
            y,
            return_counts=True,
        )
    )

    distribution = {
        str(
            class_name
        ): int(
            count
        )
        for (
            class_name,
            count,
        ) in zip(
            classes_found,
            counts,
        )
    }

    print(
        f"Distribución H/D/A: "
        f"{distribution}"
    )

    if len(
        classes_found
    ) < 3:

        raise RuntimeError(
            "El dataset no contiene "
            "las tres clases H/D/A."
        )

    if len(X) < 150:

        raise RuntimeError(
            f"Solo hay {len(X)} "
            "ejemplos utilizables. "
            "Se necesitan al menos 150."
        )

    # --------------------------------------------------------
    # DIVISIÓN CRONOLÓGICA
    # --------------------------------------------------------

    split_index = int(
        len(X) * 0.80
    )

    if (
        split_index <= 0
        or split_index >= len(X)
    ):

        raise RuntimeError(
            "No se pudo crear la "
            "división cronológica 80/20."
        )

    X_train = X[
        :split_index
    ]

    y_train = y[
        :split_index
    ]

    X_valid = X[
        split_index:
    ]

    y_valid = y[
        split_index:
    ]

    print("")
    print("========================================")
    print("VALIDACIÓN DEL MODELO")
    print("========================================")

    print(
        f"Entrenamiento: "
        f"{len(X_train)}"
    )

    print(
        f"Validación: "
        f"{len(X_valid)}"
    )

    train_classes = np.unique(
        y_train
    )

    if len(
        train_classes
    ) < 3:

        raise RuntimeError(
            "El conjunto de entrenamiento "
            "no contiene H/D/A."
        )

    # --------------------------------------------------------
    # ESCALADO
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_valid_scaled = (
        scaler.transform(
            X_valid
        )
    )

    # --------------------------------------------------------
    # REGRESIÓN LOGÍSTICA MULTICLASE
    # --------------------------------------------------------

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        C=1.0,
        multi_class="auto",
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    # --------------------------------------------------------
    # VALIDACIÓN
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_valid_scaled
        )
    )

    predictions = (
        model.predict(
            X_valid_scaled
        )
    )

    accuracy = float(
        accuracy_score(
            y_valid,
            predictions,
        )
    )

    validation_log_loss = float(
        log_loss(
            y_valid,
            probabilities,
            labels=model.classes_,
        )
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Log loss: "
        f"{validation_log_loss:.4f}"
    )

    # --------------------------------------------------------
    # OBTENER MODELO ACTUAL
    # --------------------------------------------------------

    active_model = (
        get_active_model()
    )

    previous_log_loss = None

    if active_model:

        old_metrics = (
            active_model.get(
                "metrics"
            )
            or {}
        )

        try:

            previous_log_loss = float(
                old_metrics.get(
                    "validation_log_loss"
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            previous_log_loss = None

    # --------------------------------------------------------
    # DECIDIR SI ACTIVAR
    # --------------------------------------------------------

    should_activate = (
        active_model is None
        or previous_log_loss is None
        or validation_log_loss
        < previous_log_loss
    )

    # --------------------------------------------------------
    # CREAR VERSIÓN
    # --------------------------------------------------------

    trained_at = utc_now()

    version = (
        "1X2-"
        + datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d-%H%M%S"
        )
    )

    # --------------------------------------------------------
    # SERIALIZAR MODELO
    #
    # IMPORTANTE:
    # Estos nombres deben coincidir con
    # app/ai_predict.py
    # --------------------------------------------------------

    metrics = {

        "accuracy": accuracy,

        "validation_accuracy": (
            accuracy
        ),

        "log_loss": (
            validation_log_loss
        ),

        "validation_log_loss": (
            validation_log_loss
        ),

        "previous_log_loss": (
            previous_log_loss
        ),

        # Datos del dataset completo
        "total_historical_matches": (
            len(rows)
        ),

        "usable_examples": (
            len(X)
        ),

        "training_examples": (
            len(X_train)
        ),

        "validation_examples": (
            len(X_valid)
        ),

        "skipped_no_history": (
            skipped_no_history
        ),

        # Features
        "feature_names": (
            FEATURE_NAMES
        ),

        "features": (
            FEATURE_NAMES
        ),

        # Clases
        "classes": [
            str(value)
            for value
            in model.classes_
        ],

        # Modelo
        "coefficients": (
            model.coef_.tolist()
        ),

        "intercept": (
            model.intercept_.tolist()
        ),

        # StandardScaler
        "scaler_mean": (
            scaler.mean_.tolist()
        ),

        "scaler_scale": (
            scaler.scale_.tolist()
        ),

        "model_type": (
            "multinomial_logistic_regression"
        ),

        "model_name": (
            MODEL_NAME
        ),

        "trained_at": (
            trained_at
        ),
    }

    # --------------------------------------------------------
    # GUARDAR COMO INACTIVO PRIMERO
    # --------------------------------------------------------

    payload = {

        "version": version,

        "model_name": (
            MODEL_NAME
        ),

        "trained_at": (
            trained_at
        ),

        "training_matches": (
            len(X_train)
        ),

        "metrics": metrics,

        # Primero lo guardamos inactivo.
        "active": False,
    }

    print("")
    print("========================================")
    print("MODELO NUEVO")
    print("========================================")

    print(
        f"Versión: {version}"
    )

    print(
        f"Historial total: "
        f"{len(rows)}"
    )

    print(
        f"Ejemplos utilizables: "
        f"{len(X)}"
    )

    print(
        f"Entrenamiento: "
        f"{len(X_train)}"
    )

    print(
        f"Validación: "
        f"{len(X_valid)}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Log loss: "
        f"{validation_log_loss:.4f}"
    )

    print(
        f"Modelo anterior: "
        f"{previous_log_loss}"
    )

    print(
        f"¿Debe activarse?: "
        f"{should_activate}"
    )

    # Guardamos el nuevo modelo.
    save_model(
        payload
    )

    # --------------------------------------------------------
    # ACTIVACIÓN SEGURA
    # --------------------------------------------------------

    if should_activate:

        print("")
        print(
            "El nuevo modelo mejora "
            "al modelo activo."
        )

        print(
            "Desactivando modelo anterior..."
        )

        deactivate_all_models()

        print(
            "Activando nuevo modelo..."
        )

        activate_model(
            version
        )

        print(
            "✅ Nuevo modelo activado."
        )

    else:

        print("")
        print(
            "ℹ️ El nuevo modelo NO se activa "
            "porque no mejora el log loss "
            "del modelo actualmente activo."
        )

    # --------------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------------

    print("")
    print("========================================")
    print("ENTRENAMIENTO FINALIZADO")
    print("========================================")

    result = {

        "version": version,

        "active": (
            should_activate
        ),

        "accuracy": accuracy,

        "log_loss": (
            validation_log_loss
        ),

        "previous_log_loss": (
            previous_log_loss
        ),

        "total_historical_matches": (
            len(rows)
        ),

        "usable_examples": (
            len(X)
        ),

        "training_matches": (
            len(X_train)
        ),

        "validation_matches": (
            len(X_valid)
        ),

        "skipped_no_history": (
            skipped_no_history
        ),
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("========================================")


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("")
        print(
            "Proceso cancelado."
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
