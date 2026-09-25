# Fútbol IA 2.0 — Backend inicial

Backend FastAPI para Fútbol IA 2.0 usando API-Football.

## Proveedor

API-Football Free: 100 solicitudes por día y 10 por minuto. La clave se envía mediante el encabezado `x-apisports-key` y debe permanecer en el servidor.

## Variable de entorno

`API_FOOTBALL_KEY`

No pongas la clave real en este repositorio ni dentro del APK.

## Endpoints propios

- `GET /health`
- `GET /countries`
- `GET /leagues`
- `GET /fixtures`
- `GET /fixtures/{fixture_id}`
- `GET /fixtures/{fixture_id}/statistics`
- `GET /fixtures/{fixture_id}/events`
- `GET /fixtures/{fixture_id}/lineups`
- `GET /fixtures/{fixture_id}/odds`
- `GET /injuries`
- `GET /teams/{team_id}`
- `GET /standings?league=...&season=...`

Este backend todavía es la capa de conexión. La siguiente etapa será guardar los datos válidos en Supabase y después construir el modelo propio de Fútbol IA.
