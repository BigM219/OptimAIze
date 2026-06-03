import { useEffect, useState } from 'react';
import { ModulesPage } from '../features/modules/ModulesPage';
import { OverviewPage } from '../features/overview/OverviewPage';
import { getHealth, getModules, HealthResponse, launchOcrLegacyUi, ModuleStatus } from '../shared/api/parentApi';

export function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modules, setModules] = useState<ModuleStatus[]>([]);
  const [status, setStatus] = useState('Connecting to parent API...');
  const [launchStatus, setLaunchStatus] = useState('Legacy Python UIs are compatibility tools; production UX lives in web apps.');

  useEffect(() => {
    async function load() {
      try {
        const [nextHealth, nextModules] = await Promise.all([getHealth(), getModules()]);
        setHealth(nextHealth);
        setModules(nextModules.modules);
        setStatus('Parent API connected.');
      } catch (error) {
        setStatus(error instanceof Error ? error.message : 'Parent API unavailable.');
      }
    }
    void load();
  }, []);

  async function handleLaunchOcr() {
    try {
      const result = await launchOcrLegacyUi();
      setLaunchStatus(result.message);
      return result;
    } catch (error) {
      setLaunchStatus(error instanceof Error ? error.message : 'Could not launch legacy OCR UI.');
      return null;
    }
  }

  return (
    <main className="app-shell">
      <nav className="top-nav">
        <a className="brand-mark logo" href="#overview" aria-label="OptimAIze home">
          <span className="logo-bracket">[</span><strong>OPTIM<span>AI</span>ZE</strong><span className="logo-bracket">]</span>
          <small>v1.0.0</small>
        </a>
        <div className="nav-links" aria-label="Primary navigation">
          <a href="#overview">Workspace</a>
          <a href="#modules">Modules</a>
          <a href="http://127.0.0.1:8000/api/v1/health" target="_blank" rel="noreferrer">API health</a>
        </div>
        <a className="nav-cta" href="#modules">Open modules</a>
      </nav>

      <div id="overview">
        <OverviewPage health={health} status={status} moduleCount={modules.length} />
      </div>
      <div id="modules">
        <ModulesPage modules={modules} launchStatus={launchStatus} onLaunchOcr={handleLaunchOcr} />
      </div>
    </main>
  );
}
