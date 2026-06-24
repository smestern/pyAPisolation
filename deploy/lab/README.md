# Lab self-host

Runs the gigaseal analysis web app on a lab VM, no upload caps, with optional
bearer-token auth and an optional server-side file picker for NFS-mounted
recordings.

## Quick start

```bash
cd deploy/lab
cp .env.example .env
# edit .env: set GIGASEAL_API_TOKEN, GIGASEAL_SECRET_KEY, GIGASEAL_DATA_DIR
docker compose up -d
# open http://<host>:8000
```

Clients must send the token on every request:

```bash
curl -H "Authorization: Bearer $GIGASEAL_API_TOKEN" http://<host>:8000/api/config
```

The browser UI accepts the token via `?token=...` on the first visit (it then
sticks via the signed session cookie). For wider rollout, put the container
behind a reverse proxy (Caddy, nginx) and terminate TLS there.

## Bare-metal install (no Docker)

```bash
pip install 'gigaseal[server]'
export GIGASEAL_WEB_PROFILE=lab
export GIGASEAL_API_TOKEN=changeme
gunicorn -w 4 -b 0.0.0.0:8000 gigaseal.webViz.analysis_web.wsgi:app
```

Or for development:

```bash
gigaseal web-analysis --profile lab --dev --host 0.0.0.0
```

## What's different vs the public profile

| Setting              | public        | lab               |
|----------------------|---------------|-------------------|
| Max files / session  | 5             | unlimited         |
| Max file size        | 50 MB         | unlimited         |
| Auth                 | none          | bearer token      |
| Server-path picker   | off           | on (if mounted)   |
| Rate limit           | 20/min        | none              |
| Session TTL          | 1 h           | 24 h              |
| Concurrent jobs cap  | 2             | configurable      |
