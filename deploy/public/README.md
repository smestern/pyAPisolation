# Public deploy notes

This directory holds the manifests for hosting gigaseal as a **public demo**
with hard upload caps (5 files × 50 MB per session, anonymous, rate-limited).
Defaults are set in [`gigaseal/webViz/analysis_web/config.py`](../../gigaseal/webViz/analysis_web/config.py).

## Pick a host

| Host    | File             | Free tier   | Cold start | Notes |
|---------|------------------|-------------|------------|-------|
| Fly.io  | `fly.toml`       | 3 shared VMs | ~1–2 s     | Best balance. `flyctl deploy` from this dir. |
| Render  | `render.yaml`    | 1 web svc    | ~10–30 s   | Blueprint deploy from the dashboard. Sleeps after 15 min idle. |

Both reuse the repo-root [Dockerfile](../../Dockerfile).

## Defense layers (all enabled by `PROFILE=public`)

1. **Session quota** — 5 files × 50 MB enforced in [`session.py`](../../gigaseal/webViz/analysis_web/session.py).
2. **Concurrency cap** — global semaphore limits 2 simultaneous jobs ([`jobs.py`](../../gigaseal/webViz/analysis_web/jobs.py)).
3. **Rate limit** — 20 req/min/session via `flask-limiter`.
4. **Worker recycling** — gunicorn `--max-requests 50 --max-requests-jitter 10` (in the Dockerfile CMD).

## Smoke-test locally before deploying

```bash
docker build -t gigaseal-web .
docker run --rm -p 8000:8000 -e GIGASEAL_WEB_PROFILE=public gigaseal-web
# open http://localhost:8000, click "load demo data", run an analysis
```
