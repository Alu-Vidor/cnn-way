# CNN Way Vite Starter

This repository contains a minimal [Vite](https://vitejs.dev/) + [React](https://react.dev/) starter that is pre-configured for deployment to GitHub Pages.

## Getting Started

```bash
npm install
npm run dev
```

The development server runs at [http://localhost:5173](http://localhost:5173).

## Building

```bash
npm run build
```

The production build is output to the `dist/` directory.

## Deploying to GitHub Pages

The `base` option in [`vite.config.ts`](./vite.config.ts) is set to `"./"`, which works well for GitHub Pages repositories published from the `docs/` folder or the `gh-pages` branch. If you are publishing to `https://<username>.github.io/<repository>`, update the value to `"/repository/"` so that assets resolve correctly.

1. Build the project:
   ```bash
   npm run build
   ```
2. Publish the contents of the `dist/` directory to your `gh-pages` branch (you can automate this with tools such as [`gh-pages`](https://github.com/tschaub/gh-pages)).

For more details see the [official Vite static deployment guide](https://vitejs.dev/guide/static-deploy.html#github-pages).
