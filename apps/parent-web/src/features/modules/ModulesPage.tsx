import { LaunchResult, ModuleStatus } from '../../shared/api/parentApi';

type Props = {
  modules: ModuleStatus[];
  launchStatus: string;
  onLaunchOcr: () => Promise<LaunchResult | null>;
};

export function ModulesPage({ modules, launchStatus, onLaunchOcr }: Props) {
  return (
    <section className="module-section">
      <div className="section-heading">
        <p className="eyebrow">Child systems</p>
        <h2>Launch specialized AI products from one parent workspace.</h2>
        <p>
          Each module keeps an independent frontend and backend, while the parent API exposes a clean orchestration surface.
        </p>
      </div>
      <div className="module-grid">
        {modules.map((module) => (
          <article className="module-card terminal-card" key={module.id}>
            <div className="terminal-card__command">$ optimaize module inspect {module.id}</div>
            <div className="module-card__top">
              <span>{module.kind}</span>
              <strong className={module.available ? 'status-chip status-chip--ok' : 'status-chip'}>{module.available ? 'Ready' : 'Missing'}</strong>
            </div>
            <h3>{module.name}</h3>
            <p>{module.message}</p>
            <dl className="capability-list">
              <div><dt>Runtime source</dt><dd>{module.source_available ? 'Available' : 'Missing'}</dd></div>
              <div><dt>FastAPI backend</dt><dd>{module.api_available ? 'Available' : 'Missing'}</dd></div>
              <div><dt>React web</dt><dd>{module.web_available ? 'Available' : 'Missing'}</dd></div>
              <div><dt>Legacy Python UI</dt><dd>{module.ui_available ? 'Compatibility only' : 'Missing'}</dd></div>
            </dl>
            <div className="module-actions">
              {module.web_url && <a className="primary-action" href={module.web_url} target="_blank" rel="noreferrer">Open web</a>}
              {module.api_url && <a className="secondary-action" href={`${module.api_url}/api/v1/health`} target="_blank" rel="noreferrer">API health</a>}
              {module.id === 'ocr' && <button className="ghost-action" type="button" onClick={() => void onLaunchOcr()}>Launch legacy UI</button>}
            </div>
          </article>
        ))}
      </div>
      <p className="launch-status">{launchStatus}</p>
    </section>
  );
}
