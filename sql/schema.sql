create table if not exists public.leagues (
    id bigint primary key,
    name text not null,
    country text,
    type text,
    active boolean default true,
    created_at timestamptz default now()
);

create table if not exists public.seasons (
    id bigint primary key,
    league_id bigint references public.leagues(id),
    name text,
    starting_at timestamptz,
    ending_at timestamptz
);

create table if not exists public.teams (
    id bigint primary key,
    name text not null,
    short_code text,
    country text,
    venue_name text,
    league_id bigint references public.leagues(id),
    created_at timestamptz default now()
);

create table if not exists public.players (
    id bigint primary key,
    team_id bigint references public.teams(id),
    name text not null,
    position text,
    nationality text,
    date_of_birth date,
    active boolean default true
);

create table if not exists public.referees (
    id bigint primary key,
    name text not null,
    nationality text
);

create table if not exists public.matches (
    id bigint primary key,
    league_id bigint references public.leagues(id),
    season_id bigint references public.seasons(id),
    home_team_id bigint references public.teams(id),
    away_team_id bigint references public.teams(id),
    referee_id bigint references public.referees(id),
    starting_at timestamptz,
    status text,
    home_goals integer,
    away_goals integer,
    home_ht_goals integer,
    away_ht_goals integer,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create table if not exists public.match_statistics (
    id bigint generated always as identity primary key,
    match_id bigint references public.matches(id) on delete cascade,
    team_id bigint references public.teams(id),
    stat_type text not null,
    value_numeric double precision,
    value_text text,
    unique(match_id, team_id, stat_type)
);

create table if not exists public.match_xg (
    id bigint generated always as identity primary key,
    match_id bigint references public.matches(id) on delete cascade,
    team_id bigint references public.teams(id),
    metric text not null,
    value_numeric double precision,
    unique(match_id, team_id, metric)
);

create table if not exists public.match_events (
    id bigint generated always as identity primary key,
    match_id bigint references public.matches(id) on delete cascade,
    minute integer,
    event_type text,
    team_id bigint references public.teams(id),
    player_id bigint references public.players(id),
    detail text
);

create table if not exists public.player_availability (
    id bigint generated always as identity primary key,
    match_id bigint references public.matches(id) on delete cascade,
    player_id bigint references public.players(id),
    team_id bigint references public.teams(id),
    status text,
    reason text,
    source text
);

create table if not exists public.odds (
    id bigint generated always as identity primary key,
    match_id bigint references public.matches(id) on delete cascade,
    bookmaker text,
    market text,
    selection text,
    price double precision,
    captured_at timestamptz default now()
);

create table if not exists public.predictions (
    id bigint generated always as identity primary key,
    match_id bigint references public.matches(id) on delete cascade,
    model_version text not null,
    market text not null,
    selection text not null,
    probability double precision,
    predicted_at timestamptz default now(),
    features_snapshot jsonb
);

create table if not exists public.prediction_results (
    id bigint generated always as identity primary key,
    prediction_id bigint references public.predictions(id) on delete cascade,
    outcome boolean,
    actual_value text,
    evaluated_at timestamptz default now()
);

create table if not exists public.model_versions (
    version text primary key,
    model_name text not null,
    trained_at timestamptz,
    training_matches integer,
    metrics jsonb,
    active boolean default false
);

create index if not exists idx_matches_starting_at
on public.matches(starting_at);

create index if not exists idx_matches_teams
on public.matches(home_team_id, away_team_id);

create index if not exists idx_match_statistics_match
on public.match_statistics(match_id);

create index if not exists idx_match_xg_match
on public.match_xg(match_id);

create index if not exists idx_match_events_match
on public.match_events(match_id);

create index if not exists idx_player_availability_match
on public.player_availability(match_id);

create index if not exists idx_odds_match
on public.odds(match_id);

create index if not exists idx_odds_captured_at
on public.odds(captured_at);

create index if not exists idx_predictions_match
on public.predictions(match_id);

create index if not exists idx_prediction_results_prediction
on public.prediction_results(prediction_id);

alter table public.leagues enable row level security;
alter table public.seasons enable row level security;
alter table public.teams enable row level security;
alter table public.players enable row level security;
alter table public.referees enable row level security;
alter table public.matches enable row level security;
alter table public.match_statistics enable row level security;
alter table public.match_xg enable row level security;
alter table public.match_events enable row level security;
alter table public.player_availability enable row level security;
alter table public.odds enable row level security;
alter table public.predictions enable row level security;
alter table public.prediction_results enable row level security;
alter table public.model_versions enable row level security;

revoke all on all tables in schema public from anon;
revoke all on all tables in schema public from authenticated;
