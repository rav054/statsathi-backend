# StatSathi backend

The FastAPI service behind [StatSathi](https://github.com/rav054/statsathi-frontend): accounts, projects
shared between colleagues, and the statistics itself — ANOVA with mean-separation letters, regression,
time series, G×E, PCA, clustering, SEM, data preparation and multiple imputation — plus the Word reports
and publication-quality figures built from them.

Every statistical result is checked against an independent reference implementation by the test suite
(`tests/`, 371 tests).

## Run it locally

```bash
python -m venv venv
venv/Scripts/activate          # macOS or Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # then fill in DATABASE_URL and JWT_SECRET_KEY
python -m uvicorn app.main:app --reload --port 8001
```

The API is then at `http://127.0.0.1:8001`, with its documentation at `/docs` and a health check at
`/health`. Local development uses SQLite, so it never touches the production database.

Run the tests from this folder:

```bash
python -m unittest discover -s tests -t .
```

## Settings

Every setting is an environment variable; `.env.example` lists them with comments. The two required
ones are `DATABASE_URL` and `JWT_SECRET_KEY`. The rest control the web app's address, account emails,
sign-in limits, upload sizes and who may call the API.

## Deploying to Render

`render.yaml` carries the build and start commands, the health check and the list of settings, so a
new service can be created from this repository as a Blueprint. Fill the secrets (`DATABASE_URL`,
`JWT_SECRET_KEY`, `FRONTEND_URL`, `CORS_ORIGINS`, and the `SMTP_*` values if emails should really be
sent) in the Render dashboard; they are never stored in the repository.

Two things worth knowing:

- A free Render instance sleeps after 15 minutes without traffic, and this service takes 20–30 seconds
  to start because it loads the statistics libraries. The first visitor after a quiet period waits that
  long. A paid instance, or a scheduled request every few minutes, avoids it.
- Until `EMAIL_BACKEND=smtp` and the `SMTP_*` settings are filled in, password-reset, address
  confirmation and project invitation emails are only written to the server log.
