# Past Forward monorepo

The application is split into two independently deployable projects:

- `fe/` — static Vite frontend for Vercel
- `be/` — FastAPI backend for Railway

## Local development

Backend:

```powershell
cd be
Copy-Item .env.example .env
python -m pip install -e ".[dev]"
python -m sofia_harness.system_cli serve --port 8000
```

Frontend (in another terminal):

```powershell
cd fe
Copy-Item .env.example .env.local
npm install
npm run dev
```

The local defaults connect Vite on port 5173 to FastAPI on port 8000.

## Deploy Railway (`be/`)

1. Create a Railway service from this GitHub repository.
2. Set its **Root Directory** to `/be`.
3. Railway reads `be/railway.toml`; the health check is `/health`.
4. Add the backend variables from `be/.env.example` in Railway. Never upload `be/.env`.
5. Set `PAST_FORWARD_FRONTEND_URLS` to the final Vercel origin (and any custom frontend domain), separated by commas.

Generated digitization folders are intentionally excluded from Git. Railway's normal filesystem is ephemeral; use a Railway volume or object storage before relying on newly generated files as permanent archive data.

## Deploy Vercel (`fe/`)

1. Import the same GitHub repository as a separate Vercel project.
2. Set its **Root Directory** to `fe` and framework preset to **Vite**.
3. Add the variables from `fe/.env.example`.
4. Set `VITE_API_URL` to the public Railway URL, with no trailing slash.
5. Deploy, then copy the final Vercel URL into Railway's `PAST_FORWARD_FRONTEND_URLS` and redeploy the backend.

The frontend never receives `OPENAI_API_KEY` or any other server secret. Supabase's publishable browser key is intentionally public; authorization is still enforced by the backend.
