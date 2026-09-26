import os
import math
from datetime import datetime, timezone
from typing import Any, Optional

import httpx


# ============================================================
# CONFIGURACIÓN
# ============================================================

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

WINDOW = 5

FINISHED_STATUSES = {
    "FT",
    "AET",
    "PEN",
}


# ============================================================
# SUPABASE
# ============================================================

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
            "Falta SUPABASE_URL"
        )

    if not key:
        raise RuntimeError(
            "Falta SUPABASE_SECRET_KEY"
        )

    return url, key


def supabase_headers() -> dict[str, str]:

    _, key = get_supabase_config()

    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


async def supabase_get(
    table: str,
    params: Optional[dict[str, Any]] = None,
) -> list[dict]:

    url, _ = get_supabase_config()

    endpoint = (
        f"{url}/rest/v1/{table}"
    )

    async with httpx.AsyncClient(
        timeout=60
    ) as client:

        response = await client.get(
            endpoint,
            headers=supabase_headers(),
            params=params or {},
        )

    if response.status_code >= 400:

        raise RuntimeError(
            f"Supabase GET {table}: "
            f"{response.status_code} "
            f"{response.text}"
        )

    try:

        data = response.json()

    except Exception:

        raise RuntimeError(
            f"Supabase devolvió una respuesta "
            f"no válida para {table}"
        )

    if isinstance(data, list):
        return data

    return []


async def supabase_post(
    table: str,
    payload: Any,
    prefer: str = (
        "resolution=merge-duplicates,"
        "return=representation"
    ),
) -> Any:

    url, _ = get_supabase_config()

    endpoint = (
        f"{url}/rest/v1/{table}"
    )

    headers = supabase_headers()

    headers["Prefer"] = prefer

    async with httpx.AsyncClient(
        timeout=60
    ) as client:

        response = await client.post(
            endpoint,
            headers=headers,
            json=payload,
        )

    if response.status_code >= 400:

        raise RuntimeError(
            f"Supabase POST {table}: "
            f"{response.status_code} "
            f"{response.text}"
        )

    if not response.text:
        return None

    try:

        return response.json()

    except Exception:

        return response.text


# ============================================================
# UTILIDADES
# ============================================================

def now_iso() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        if value is None:
            return default

        return float(value)

    except Exception:

        return default


def safe_int(
    value: Any,
    default: Optional[int] = None,
) -> Optional[int]:

    try:

        if value is None:
            return default

        return int(value)

    except Exception:

        return default


def normalize_status(
    value: Any,
) -> str:

    if value is None:
        return ""

    return str(value).strip().upper()


def normalize_class(
    value: Any,
) -> str:

    text = str(value).strip().upper()

    if text in {
        "H",
        "HOME",
        "LOCAL",
        "1",
    }:
        return "H"

    if text in {
        "D",
        "DRAW",
        "EMPATE",
        "X",
    }:
        return "D"

    if text in {
        "A",
        "AWAY",
        "VISITOR",
        "VISITANTE",
        "2",
    }:
        return "A"

    return text


# ============================================================
# MODELO
# ============================================================

async def load_active_model() -> dict:

    rows = await supabase_get(
        "model_versions",
        {
            "select": "*",
            "active": "eq.true",
            "order": "trained_at.desc",
            "limit": "1",
        },
    )

    if not rows:

        raise RuntimeError(
            "No existe un modelo activo "
            "en model_versions"
        )

    row = rows[0]

    metrics = row.get(
        "metrics"
    )

    if not isinstance(
        metrics,
        dict,
    ):

        raise RuntimeError(
            "El modelo activo no contiene "
            "metrics válidos"
        )

    model = {
        "version": row.get(
            "version"
        ),

        "model_name": row.get(
            "model_name"
        ),

        "trained_at": row.get(
            "trained_at"
        ),

        "training_matches": row.get(
            "training_matches"
        ),

        "metrics": metrics,
    }

    model_type = (
        metrics.get(
            "model_type"
        )
    )

    if model_type is None:

        model_type = (
            "multinomial_logistic_regression"
        )

    if model_type not in {
        "multinomial_logistic_regression",
        "logistic_regression",
    }:

        raise RuntimeError(
            "Tipo de modelo no compatible: "
            f"{model_type}"
        )

    model["model_type"] = model_type

    feature_names = metrics.get(
        "feature_names"
    )

    if not feature_names:

        feature_names = metrics.get(
            "features"
        )

    if not feature_names:

        feature_names = FEATURE_NAMES

    model["feature_names"] = list(
        feature_names
    )

    classes = metrics.get(
        "classes",
        ["H", "D", "A"],
    )

    model["classes"] = [
        normalize_class(value)
        for value in classes
    ]

    coefficients = metrics.get(
        "coefficients"
    )

    intercept = metrics.get(
        "intercept"
    )

    if coefficients is None:

        coefficients = metrics.get(
            "coef"
        )

    if intercept is None:

        intercept = metrics.get(
            "intercepts"
        )

    if coefficients is None:

        raise RuntimeError(
            "El modelo activo no contiene "
            "coefficients"
        )

    if intercept is None:

        raise RuntimeError(
            "El modelo activo no contiene "
            "intercept"
        )

    model["coefficients"] = coefficients
    model["intercept"] = intercept

    scaler_mean = metrics.get(
        "scaler_mean"
    )

    scaler_scale = metrics.get(
        "scaler_scale"
    )

    if scaler_mean is None:

        scaler_mean = [
            0.0
            for _ in model[
                "feature_names"
            ]
        ]

    if scaler_scale is None:

        scaler_scale = [
            1.0
            for _ in model[
                "feature_names"
            ]
        ]

    model["scaler_mean"] = scaler_mean
    model["scaler_scale"] = scaler_scale

    return model


# ============================================================
# PARTIDO
# ============================================================

async def get_match(
    match_id: int,
) -> dict:

    rows = await supabase_get(
        "matches",
        {
            "select": "*",
            "id": f"eq.{match_id}",
            "limit": "1",
        },
    )

    if not rows:

        raise RuntimeError(
            f"No existe el partido {match_id} "
            f"en la tabla matches de Supabase"
        )

    return rows[0]


# ============================================================
# HISTORIAL
# ============================================================

async def get_previous_matches(
    before_date: Optional[str],
    limit: int = 1000,
) -> list[dict]:

    params = {
        "select": (
            "id,"
            "starting_at,"
            "status,"
            "home_team_id,"
            "away_team_id,"
            "home_goals,"
            "away_goals"
        ),
        "status": (
            "in.(FT,AET,PEN)"
        ),
        "order": (
            "starting_at.asc"
        ),
        "limit": str(limit),
    }

    if before_date:

        params[
            "starting_at"
        ] = f"lt.{before_date}"

    return await supabase_get(
        "matches",
        params,
    )


def team_form(
    matches: list[dict],
    team_id: int,
) -> dict:

    history = []

    for match in matches:

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
            home_id != team_id
            and away_id != team_id
        ):
            continue

        home_goals = safe_int(
            match.get(
                "home_goals"
            ),
            0,
        )

        away_goals = safe_int(
            match.get(
                "away_goals"
            ),
            0,
        )

        if home_goals is None:
            home_goals = 0

        if away_goals is None:
            away_goals = 0

        if home_id == team_id:

            goals_for = home_goals
            goals_against = away_goals

        else:

            goals_for = away_goals
            goals_against = home_goals

        if goals_for > goals_against:

            points = 3

        elif goals_for == goals_against:

            points = 1

        else:

            points = 0

        history.append({
            "goals_for": goals_for,
            "goals_against": goals_against,
            "points": points,
        })

    recent = history[-WINDOW:]

    if not recent:

        return {
            "goals_for": 0.0,
            "goals_against": 0.0,
            "points": 0.0,
            "matches": 0,
        }

    return {
        "goals_for": sum(
            item["goals_for"]
            for item in recent
        ) / len(recent),

        "goals_against": sum(
            item["goals_against"]
            for item in recent
        ) / len(recent),

        "points": sum(
            item["points"]
            for item in recent
        ),

        "matches": len(recent),
    }


# ============================================================
# CARACTERÍSTICAS
# ============================================================

async def build_features(
    match: dict,
) -> list[float]:

    match_id = safe_int(
        match.get(
            "id"
        )
    )

    home_team_id = safe_int(
        match.get(
            "home_team_id"
        )
    )

    away_team_id = safe_int(
        match.get(
            "away_team_id"
        )
    )

    if home_team_id is None:

        raise RuntimeError(
            f"El partido {match_id} "
            f"no tiene home_team_id"
        )

    if away_team_id is None:

        raise RuntimeError(
            f"El partido {match_id} "
            f"no tiene away_team_id"
        )

    starting_at = match.get(
        "starting_at"
    )

    previous_matches = (
        await get_previous_matches(
            before_date=starting_at
        )
    )

    home_form = team_form(
        previous_matches,
        home_team_id,
    )

    away_form = team_form(
        previous_matches,
        away_team_id,
    )

    goals_form_difference = (
        (
            home_form["goals_for"]
            - home_form["goals_against"]
        )
        -
        (
            away_form["goals_for"]
            - away_form["goals_against"]
        )
    )

    points_form_difference = (
        home_form["points"]
        - away_form["points"]
    )

    features = {

        "home_goals_for_5":
            home_form["goals_for"],

        "home_goals_against_5":
            home_form["goals_against"],

        "home_points_5":
            home_form["points"],

        "away_goals_for_5":
            away_form["goals_for"],

        "away_goals_against_5":
            away_form["goals_against"],

        "away_points_5":
            away_form["points"],

        "goals_form_difference":
            goals_form_difference,

        "points_form_difference":
            points_form_difference,

        "home_advantage":
            1.0,
    }

    return [
        safe_float(
            features.get(
                name,
                0.0
            )
        )
        for name in FEATURE_NAMES
    ]


# ============================================================
# ESCALADO
# ============================================================

def scale_features(
    values: list[float],
    model: dict,
) -> list[float]:

    means = model.get(
        "scaler_mean",
        [],
    )

    scales = model.get(
        "scaler_scale",
        [],
    )

    result = []

    for index, value in enumerate(
        values
    ):

        mean = 0.0

        scale = 1.0

        if index < len(means):

            mean = safe_float(
                means[index],
                0.0,
            )

        if index < len(scales):

            scale = safe_float(
                scales[index],
                1.0,
            )

        if abs(scale) < 1e-12:

            scale = 1.0

        result.append(
            (
                safe_float(
                    value
                )
                - mean
            )
            / scale
        )

    return result


# ============================================================
# SIGMOIDE / SOFTMAX
# ============================================================

def sigmoid(
    value: float,
) -> float:

    value = max(
        -60.0,
        min(
            60.0,
            value
        )
    )

    return 1.0 / (
        1.0 + math.exp(-value)
    )


def softmax(
    values: list[float],
) -> list[float]:

    if not values:

        return []

    maximum = max(
        values
    )

    exponentials = [
        math.exp(
            max(
                -60.0,
                min(
                    60.0,
                    value - maximum
                )
            )
        )
        for value in values
    ]

    total = sum(
        exponentials
    )

    if total <= 0:

        equal = 1.0 / len(
            exponentials
        )

        return [
            equal
            for _ in exponentials
        ]

    return [
        value / total
        for value in exponentials
    ]


# ============================================================
# PREDICCIÓN DEL MODELO
# ============================================================

def calculate_probabilities(
    values: list[float],
    model: dict,
) -> dict[str, float]:

    scaled = scale_features(
        values,
        model,
    )

    coefficients = model.get(
        "coefficients"
    )

    intercept = model.get(
        "intercept"
    )

    classes = model.get(
        "classes",
        ["H", "D", "A"],
    )

    if not isinstance(
        coefficients,
        list,
    ):

        raise RuntimeError(
            "coefficients no es una lista"
        )

    if not isinstance(
        intercept,
        list,
    ):

        intercept = [
            intercept
        ]

    # --------------------------------------------------------
    # Caso multinomial
    # --------------------------------------------------------

    if len(coefficients) > 1:

        scores = []

        for class_index, class_coef in enumerate(
            coefficients
        ):

            bias = 0.0

            if class_index < len(
                intercept
            ):

                bias = safe_float(
                    intercept[class_index]
                )

            score = bias

            if isinstance(
                class_coef,
                list,
            ):

                for index, value in enumerate(
                    scaled
                ):

                    if index < len(
                        class_coef
                    ):

                        score += (
                            safe_float(
                                class_coef[index]
                            )
                            * value
                        )

            scores.append(
                score
            )

        probabilities = softmax(
            scores
        )

        result = {
            "H": 0.0,
            "D": 0.0,
            "A": 0.0,
        }

        for index, probability in enumerate(
            probabilities
        ):

            if index >= len(
                classes
            ):
                continue

            class_name = normalize_class(
                classes[index]
            )

            if class_name in result:

                result[
                    class_name
                ] = probability

        return result

    # --------------------------------------------------------
    # Caso binario / fallback
    # --------------------------------------------------------

    class_coef = (
        coefficients[0]
        if coefficients
        else []
    )

    bias = safe_float(
        intercept[0]
        if intercept
        else 0.0
    )

    score = bias

    if isinstance(
        class_coef,
        list,
    ):

        for index, value in enumerate(
            scaled
        ):

            if index < len(
                class_coef
            ):

                score += (
                    safe_float(
                        class_coef[index]
                    )
                    * value
                )

    probability = sigmoid(
        score
    )

    # Si el modelo binario no permite
    # representar 3 clases, usamos un
    # fallback neutro para no inventar
    # una tercera clase.

    return {
        "H": probability,
        "D": 0.0,
        "A": 1.0 - probability,
    }


# ============================================================
# PREDICCIÓN PRINCIPAL
# ============================================================

async def predict_match(
    match_id: int,
) -> dict:

    # --------------------------------------------------------
    # 1. Buscar partido
    # --------------------------------------------------------

    match = await get_match(
        match_id
    )

    # --------------------------------------------------------
    # 2. Cargar modelo
    # --------------------------------------------------------

    model = await load_active_model()

    # --------------------------------------------------------
    # 3. Estado del partido
    # --------------------------------------------------------

    status = normalize_status(
        match.get(
            "status"
        )
    )

    # No permitimos predecir como
    # futuro un partido que ya terminó.
    #
    # Para pruebas técnicas sí podemos
    # calcular la predicción histórica,
    # pero la marcamos como histórica.

    historical = (
        status in FINISHED_STATUSES
    )

    # --------------------------------------------------------
    # 4. Crear características
    # --------------------------------------------------------

    feature_values = (
        await build_features(
            match
        )
    )

    # --------------------------------------------------------
    # 5. Calcular probabilidades
    # --------------------------------------------------------

    probabilities = (
        calculate_probabilities(
            feature_values,
            model,
        )
    )

    home_probability = (
        probabilities["H"]
    )

    draw_probability = (
        probabilities["D"]
    )

    away_probability = (
        probabilities["A"]
    )

    # --------------------------------------------------------
    # 6. Normalizar por seguridad
    # --------------------------------------------------------

    total = (
        home_probability
        + draw_probability
        + away_probability
    )

    if total <= 0:

        home_probability = 1.0 / 3.0
        draw_probability = 1.0 / 3.0
        away_probability = 1.0 / 3.0

    else:

        home_probability /= total
        draw_probability /= total
        away_probability /= total

    # --------------------------------------------------------
    # 7. Selección principal
    # --------------------------------------------------------

    probability_map = {
        "HOME": home_probability,
        "DRAW": draw_probability,
        "AWAY": away_probability,
    }

    predicted_selection = max(
        probability_map,
        key=probability_map.get,
    )

    predicted_probability = (
        probability_map[
            predicted_selection
        ]
    )

    # --------------------------------------------------------
    # 8. Guardar predicciones
    # --------------------------------------------------------

    model_version = model.get(
        "version"
    )

    if not model_version:

        raise RuntimeError(
            "El modelo activo no tiene versión"
        )

    features_snapshot = {

        "feature_names":
            FEATURE_NAMES,

        "feature_values":
            feature_values,

        "historical":
            historical,

        "match_status":
            status,
    }

    saved_predictions = []

    selections = [
        (
            "HOME",
            home_probability
        ),
        (
            "DRAW",
            draw_probability
        ),
        (
            "AWAY",
            away_probability
        ),
    ]

    for selection, probability in selections:

        payload = {

            "match_id":
                match_id,

            "model_version":
                model_version,

            "market":
                "1X2",

            "selection":
                selection,

            "probability":
                float(probability),

            "predicted_at":
                now_iso(),

            "features_snapshot":
                features_snapshot,
        }

        try:

            result = await supabase_post(
                "predictions",
                payload,
            )

            saved_predictions.append({
                "selection":
                    selection,

                "probability":
                    probability,

                "saved":
                    True,

                "result":
                    result,
            })

        except Exception as exc:

            saved_predictions.append({
                "selection":
                    selection,

                "probability":
                    probability,

                "saved":
                    False,

                "error":
                    str(exc),
            })

    # --------------------------------------------------------
    # 9. Respuesta
    # --------------------------------------------------------

    return {

        "ok":
            True,

        "match_id":
            match_id,

        "historical":
            historical,

        "match_status":
            status,

        "home_team_id":
            match.get(
                "home_team_id"
            ),

        "away_team_id":
            match.get(
                "away_team_id"
            ),

        "model_version":
            model_version,

        "model_name":
            model.get(
                "model_name"
            ),

        "probabilities": {

            "home":
                round(
                    home_probability,
                    6
                ),

            "draw":
                round(
                    draw_probability,
                    6
                ),

            "away":
                round(
                    away_probability,
                    6
                ),
        },

        "percentages": {

            "home":
                round(
                    home_probability
                    * 100,
                    2
                ),

            "draw":
                round(
                    draw_probability
                    * 100,
                    2
                ),

            "away":
                round(
                    away_probability
                    * 100,
                    2
                ),
        },

        "prediction":
            predicted_selection,

        "prediction_probability":
            round(
                predicted_probability,
                6
            ),

        "prediction_percentage":
            round(
                predicted_probability
                * 100,
                2
            ),

        "features":
            {
                FEATURE_NAMES[index]:
                    feature_values[index]
                for index in range(
                    min(
                        len(
                            FEATURE_NAMES
                        ),
                        len(
                            feature_values
                        ),
                    )
                )
            },

        "saved_predictions":
            saved_predictions,
    }
