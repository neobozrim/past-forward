import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        overlayInspection: resolve(import.meta.dirname, "overlay-inspection.html")
      }
    }
  }
});
