# Docker Deployment Guide

## Prerequisites

- Docker Desktop installed and running.
- Project folder extracted from the zip file.

## Start the App

Open PowerShell in the project folder:

```powershell
docker compose up --build
```

Open the browser at:

```text
http://localhost:5000
```

## Stop the App

```powershell
docker compose down
```

## What Docker Does

Docker builds a Python 3.11 container, installs the dependencies from `requirements.txt`, copies the project files, and starts the Flask app with Waitress on port `5000`.

## Useful Checks

Health endpoint:

```text
http://localhost:5000/health
```

Expected response:

```json
{"status":"ok"}
```

## Troubleshooting

- If `docker` is not recognized, install Docker Desktop.
- If the app does not open, confirm Docker Desktop is running.
- If port `5000` is already used, change the left side of the port mapping in `docker-compose.yml`, for example `"5050:5000"`, then open `http://localhost:5050`.
