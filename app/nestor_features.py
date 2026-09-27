"""
Ingeniería de variables canónica para NESTOR 1X2.

Reglas:
- Solo información disponible antes del partido.
- Historial estrictamente cronológico.
- El entrenamiento exige historial suficiente.
- La inferencia permite historial parcial.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable


WINDOW = 5

FEATURE_SCHEMA_VERSION = (
    "NESTOR-1X2-FEATURES-v1.0"
)


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


TARGET_NAMES = {

    0: "HOME",

    1: "DRAW",

    2: "AWAY",
}


def _points(
    home_goals: int,
    away_goals: int
) -> tuple[int, int]:

    if home_goals > away_goals:

        return 3, 0

    if home_goals == away_goals:

        return 1, 1

    return 0, 3


def clean_match(
    row: dict[str, Any]
) -> dict[str, Any] | None:

    try:

        if (
            row.get("home_team_id")
            is None
            or row.get("away_team_id")
            is None
        ):
            return None

        if (
            row.get("home_goals")
            is None
            or row.get("away_goals")
            is None
        ):
            return None

        if (
            row.get("starting_at")
            is None
            or row.get("id")
            is None
        ):
            return None

        home_goals = int(
            row["home_goals"]
        )

        away_goals = int(
            row["away_goals"]
        )

        if (
            home_goals < 0
            or away_goals < 0
        ):
            return None

        return {

            "id": int(
                row["id"]
            ),

            "starting_at": str(
                row["starting_at"]
            ),

            "home_team_id": int(
                row["home_team_id"]
            ),

            "away_team_id": int(
                row["away_team_id"]
            ),

            "home_goals": home_goals,

            "away_goals": away_goals,

            "status": row.get(
                "status"
            ),
        }

    except (
        TypeError,
        ValueError
    ):

        return None


def _history_form(
    history:
        deque[
            dict[str, float]
        ]
) -> dict[str, float] | None:

    if not history:

        return None

    items = list(history)[-WINDOW:]

    n = float(
        len(items)
    )

    return {

        "goals_for": (
            sum(
                item["gf"]
                for item in items
            )
            / n
        ),

        "goals_against": (
            sum(
                item["ga"]
                for item in items
            )
            / n
        ),

        "points": (
            sum(
                item["points"]
                for item in items
            )
            / n
        ),

        "sample_size": float(
            len(items)
        ),
    }


def _feature_vector(
    home_form: dict[str, float],
    away_form: dict[str, float]
) -> list[float]:

    home_goal_diff = (
        home_form["goals_for"]
        - home_form["goals_against"]
    )

    away_goal_diff = (
        away_form["goals_for"]
        - away_form["goals_against"]
    )

    return [

        home_form["goals_for"],

        home_form["goals_against"],

        home_form["points"],

        away_form["goals_for"],

        away_form["goals_against"],

        away_form["points"],

        home_goal_diff
        - away_goal_diff,

        home_form["points"]
        - away_form["points"],

        1.0,
    ]


def build_training_dataset(
    matches:
        Iterable[
            dict[str, Any]
        ]
) -> tuple[
    list[list[float]],
    list[int],
    list[int]
]:

    histories: dict[
        int,
        deque[
            dict[str, float]
        ]
    ] = {}

    X: list[
        list[float]
    ] = []

    y: list[int] = []

    match_ids: list[int] = []

    ordered = sorted(
        matches,
        key=lambda item:
            item["starting_at"]
    )

    for match in ordered:

        home_id = int(
            match["home_team_id"]
        )

        away_id = int(
            match["away_team_id"]
        )

        home_history = (
            histories.setdefault(
                home_id,
                deque(
                    maxlen=WINDOW
                )
            )
        )

        away_history = (
            histories.setdefault(
                away_id,
                deque(
                    maxlen=WINDOW
                )
            )
        )

        home_form = _history_form(
            home_history
        )

        away_form = _history_form(
            away_history
        )

        home_goals = int(
            match["home_goals"]
        )

        away_goals = int(
            match["away_goals"]
        )

        home_points, away_points = _points(
            home_goals,
            away_goals
        )

        if (
            home_form is None
            or away_form is None
            or len(home_history) < WINDOW
            or len(away_history) < WINDOW
        ):

            pass

        else:

            X.append(
                _feature_vector(
                    home_form,
                    away_form
                )
            )

            y.append(
                0
                if home_goals > away_goals
                else (
                    1
                    if home_goals
                    == away_goals
                    else 2
                )
            )

            match_ids.append(
                int(match["id"])
            )

        # MUY IMPORTANTE:
        # el partido actual se incorpora al historial
        # solamente después de crear sus variables.

        home_history.append({

            "gf":
                float(home_goals),

            "ga":
                float(away_goals),

            "points":
                float(home_points),
        })

        away_history.append({

            "gf":
                float(away_goals),

            "ga":
                float(home_goals),

            "points":
                float(away_points),
        })

    return (
        X,
        y,
        match_ids
    )


def build_inference_features(
    home_history_matches:
        Iterable[dict[str, Any]],

    away_history_matches:
        Iterable[dict[str, Any]],

    feature_defaults:
        dict[str, float]
) -> tuple[
    list[float],
    dict[str, Any]
]:

    def team_form(
        rows:
            Iterable[dict[str, Any]]
    ) -> tuple[
        dict[str, float] | None,
        int
    ]:

        ordered = sorted(
            rows,
            key=lambda item:
                str(item["starting_at"]),
            reverse=True
        )

        recent = ordered[:WINDOW]

        history: list[
            dict[str, float]
        ] = []

        for item in recent:

            try:

                gf = float(
                    item["home_goals"]
                    if item["home_team_id"]
                    == item["team_id"]
                    else item["away_goals"]
                )

                ga = float(
                    item["away_goals"]
                    if item["home_team_id"]
                    == item["team_id"]
                    else item["home_goals"]
                )

            except (
                KeyError,
                TypeError,
                ValueError
            ):

                continue

            hpts, apts = _points(
                int(gf),
                int(ga)
            )

            pts = (
                hpts
                if item["home_team_id"]
                == item["team_id"]
                else apts
            )

            history.append({

                "gf": gf,

                "ga": ga,

                "points":
                    float(pts),
            })

        return (
            _history_form(
                deque(
                    history,
                    maxlen=WINDOW
                )
            ),
            len(history)
        )

    home_form, home_sample = team_form(
        home_history_matches
    )

    away_form, away_sample = team_form(
        away_history_matches
    )

    snapshot: dict[str, Any] = {

        "feature_schema_version":
            FEATURE_SCHEMA_VERSION,

        "home_sample_size":
            home_sample,

        "away_sample_size":
            away_sample,

        "imputed_home":
            home_form is None,

        "imputed_away":
            away_form is None,
    }

    if home_form is None:

        home_form = {

            "goals_for":
                float(
                    feature_defaults.get(
                        "home_goals_for_5",
                        1.0
                    )
                ),

            "goals_against":
                float(
                    feature_defaults.get(
                        "home_goals_against_5",
                        1.0
                    )
                ),

            "points":
                float(
                    feature_defaults.get(
                        "home_points_5",
                        1.0
                    )
                ),
        }

    if away_form is None:

        away_form = {

            "goals_for":
                float(
                    feature_defaults.get(
                        "away_goals_for_5",
                        1.0
                    )
                ),

            "goals_against":
                float(
                    feature_defaults.get(
                        "away_goals_against_5",
                        1.0
                    )
                ),

            "points":
                float(
                    feature_defaults.get(
                        "away_points_5",
                        1.0
                    )
                ),
        }

    vector = _feature_vector(
        home_form,
        away_form
    )

    snapshot["features"] = {

        name: float(value)

        for name, value in zip(
            FEATURE_NAMES,
            vector
        )
    }

    total_sample = (
        home_sample
        + away_sample
    )

    if (
        total_sample >= 10
        and not (
            snapshot["imputed_home"]
            or snapshot["imputed_away"]
        )
    ):

        quality = "high"

    elif total_sample >= 5:

        quality = "medium"

    else:

        quality = "low"

    snapshot["data_quality"] = quality

    return (
        vector,
        snapshot
    )


def target_name(
    value: int
) -> str:

    return TARGET_NAMES[
        int(value)
          ]
