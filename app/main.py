@app.get("/sync/statistics/batch")
async def sync_statistics_batch(
    league: int,
    season: int,
    limit: int = 5
):
    if limit < 1 or limit > 10:
        raise HTTPException(
            status_code=400,
            detail="El límite debe estar entre 1 y 10"
        )

    # Primero obtenemos los partidos de la liga/temporada
    data = await football_get(
        "/fixtures",
        {
            "league": league,
            "season": season
        }
    )

    fixtures_data = data.get("response", [])

    if not fixtures_data:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron partidos para esta liga y temporada"
        )

    # Solo partidos terminados
    finished_statuses = {"FT", "AET", "PEN"}

    finished_fixtures = []

    for item in fixtures_data:
        fixture_info = item.get("fixture", {})
        status_info = fixture_info.get("status", {})
        status_short = status_info.get("short")

        if status_short in finished_statuses:
            fixture_id = fixture_info.get("id")

            if fixture_id is not None:
                finished_fixtures.append({
                    "id": fixture_id,
                    "status": status_short,
                    "date": fixture_info.get("date")
                })

    if not finished_fixtures:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron partidos terminados"
        )

    supabase_url, supabase_key = get_supabase_config()

    # Consultamos qué partidos ya tienen estadísticas guardadas
    saved_fixture_ids = await get_saved_statistic_fixture_ids(
        supabase_url,
        supabase_key
    )

    # Quitamos los que ya fueron sincronizados
    pending_fixtures = [
        item
        for item in finished_fixtures
        if item["id"] not in saved_fixture_ids
    ]

    # Aplicamos el límite
    selected_fixtures = pending_fixtures[:limit]

    processed = []
    failed = []

    for item in selected_fixtures:
        fixture_id = item["id"]

        try:
            result = await sync_statistics(
                fixture=fixture_id
            )

            processed.append({
                "fixture": fixture_id,
                "status": "ok",
                "result": result
            })

        except HTTPException as exc:
            failed.append({
                "fixture": fixture_id,
                "status": "error",
                "http_status": exc.status_code,
                "detail": exc.detail
            })

        except Exception as exc:
            failed.append({
                "fixture": fixture_id,
                "status": "error",
                "detail": str(exc)
            })

    return {
        "ok": True,
        "league": league,
        "season": season,
        "limit": limit,
        "existing_statistics_fixtures": len(saved_fixture_ids),
        "finished_fixtures": len(finished_fixtures),
        "pending_fixtures": len(pending_fixtures),
        "selected": len(selected_fixtures),
        "processed": len(processed),
        "failed": len(failed),
        "details": {
            "processed": processed,
            "failed": failed
        }
    }
