import { useEffect, useState } from 'react';
import { HistoryRoutePage } from '../pages/HistoryRoutePage';
import { IntroPage } from '../pages/IntroPage';
import { WorkspacePage } from '../pages/WorkspacePage';
import { getHealth, getHistoryDocuments, getModels, getRuntimeConfig, HealthResponse, HistoryDocument, ModelListResponse, RuntimeConfigResponse } from '../shared/api/ocrApi';
import { OcrNav } from './OcrNav';
import { Page, pageFromHash } from './pageRouting';

export type OcrTheme = 'nebula' | 'andromeda' | 'void';

export function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelListResponse | null>(null);
  const [runtimeConfig, setRuntimeConfig] = useState<RuntimeConfigResponse | null>(null);
  const [documents, setDocuments] = useState<HistoryDocument[]>([]);
  const [status, setStatus] = useState('Connecting to OCR backend...');
  const [page, setPage] = useState<Page>(() => pageFromHash());
  const [theme, setTheme] = useState<OcrTheme>('nebula');

  useEffect(() => {
    function handleHashChange() {
      setPage(pageFromHash());
    }

    window.addEventListener('hashchange', handleHashChange);
    if (!window.location.hash) window.history.replaceState(null, '', '#intro');
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    async function load() {
      try {
        const [nextHealth, nextModels, nextRuntimeConfig, nextDocuments] = await Promise.all([
          getHealth(),
          getModels(),
          getRuntimeConfig(),
          getHistoryDocuments(),
        ]);
        setHealth(nextHealth);
        setModels(nextModels);
        setRuntimeConfig(nextRuntimeConfig);
        setDocuments(nextDocuments.documents);
        setStatus('Backend connected.');
      } catch (error) {
        setStatus(error instanceof Error ? error.message : 'Backend unavailable.');
      }
    }
    void load();
  }, []);

  return (
    <main className="app-shell" data-theme={theme}>
      <OcrNav page={page} theme={theme} onThemeChange={setTheme} />
      {page === 'intro' && <IntroPage health={health} models={models} documents={documents} status={status} />}
      {page === 'workspace' && <WorkspacePage models={models} runtimeConfig={runtimeConfig} />}
      {page === 'history' && <HistoryRoutePage documents={documents} />}
    </main>
  );
}
