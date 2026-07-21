import { mkdirSync, writeFileSync } from "node:fs";

const value = (name, fallback = "") => process.env[name] ?? fallback;
const normalizeUrl = raw => {
  const url = raw.trim().replace(/\/$/, "");
  return url && !/^https?:\/\//i.test(url) ? `https://${url}` : url;
};
const config = {
  apiUrl: normalizeUrl(value("VITE_API_URL", "http://localhost:8000")),
  supabaseUrl: value("VITE_SUPABASE_URL"),
  supabasePublishableKey: value("VITE_SUPABASE_PUBLISHABLE_KEY"),
  authRequired: value("VITE_AUTH_REQUIRED", "false") === "true",
  authConfigured: value("VITE_AUTH_CONFIGURED", "false") === "true",
  landingEnabled: value("VITE_LANDING_ENABLED", "true") === "true"
};

mkdirSync("public", { recursive: true });
writeFileSync("public/config.js", `window.PF_CONFIG=${JSON.stringify(config)};\n`);
