import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` -> http://localhost:5173; /api is proxied to the FastAPI server.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
});
