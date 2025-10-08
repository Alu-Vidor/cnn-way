import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import './App.css';

function App() {
  return (
    <div className="app">
      <header className="hero">
        <a href="https://vitejs.dev" target="_blank" rel="noreferrer">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank" rel="noreferrer">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
        <h1 className="title">Vite + React</h1>
        <p className="subtitle">
          This project is configured for GitHub Pages deployment. Update the <code>base</code>{' '}
          option in <code>vite.config.ts</code> to match your repository if you are deploying to a
          subdirectory.
        </p>
        <div className="actions">
          <a
            className="button"
            href="https://vitejs.dev/guide/static-deploy.html#github-pages"
            target="_blank"
            rel="noreferrer"
          >
            Deployment Guide
          </a>
        </div>
      </header>
    </div>
  );
}

export default App;
