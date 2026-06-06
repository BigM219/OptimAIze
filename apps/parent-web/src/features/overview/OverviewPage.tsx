import { HealthResponse, HEALTH_URL } from '../../shared/api/parentApi';

type Props = {
  health: HealthResponse | null;
  status: string;
  moduleCount: number;
};

export function OverviewPage({ health, status, moduleCount }: Props) {
  const apiOnline = health?.ok === true;

  return (
    <section className="overview-grid">
      <article className="hero-card terminal-window">
        <div className="terminal-titlebar" aria-hidden="true">
          <span className="terminal-dot terminal-dot--red" />
          <span className="terminal-dot terminal-dot--yellow" />
          <span className="terminal-dot terminal-dot--green" />
          <code>~/optimaize/parent</code>
        </div>
        <div className="hero-copy terminal-pane">
          <p className="eyebrow">OptimAIze parent workspace</p>
          <p className="terminal-command"><span>root@optimaize:~$</span> ./init_parent_workspace --mode=modular</p>
          <p className="terminal-boot">&gt; Loading child registry... <strong>OK</strong></p>
          <h1 aria-label="Orchestrate AI modules without mixing product boundaries.">
            <span className="glitch" data-text="Orchestrate AI modules without mixing product boundaries." aria-hidden="true">Orchestrate AI modules without mixing product boundaries.</span>
          </h1>
          <p>
            A web-first command center for independent OptimAIze systems. React owns the workflow, FastAPI coordinates services, and each child module keeps its own frontend, backend, data, and AI runtime.
          </p>
          <div className="hero-actions">
            <a className="primary-action" href="#modules">View modules</a>
            <a className="secondary-action" href={HEALTH_URL} target="_blank" rel="noreferrer">Check API</a>
          </div>
        </div>

        <div className="console-card" aria-label="OptimAIze command center status">
          <div className="console-card__top">
            <span className="status-dot" />
            <strong>Command center</strong>
            <small>{apiOnline ? 'Live' : 'Waiting'}</small>
          </div>
          <div className="console-flow">
            <div>
              <span>Parent API</span>
              <strong>{apiOnline ? 'Online' : 'Unknown'}</strong>
            </div>
            <div>
              <span>Registry</span>
              <strong>{moduleCount} modules</strong>
            </div>
            <div>
              <span>Runtime boundary</span>
              <strong>FE / BE split</strong>
            </div>
          </div>
          <p>{status}</p>
        </div>
      </article>

      <article className="metric-card metric-card--wide">
        <span>Architecture</span>
        <strong>React UX · Python services</strong>
        <small>Production frontends stay separate from AI/data/job backends.</small>
      </article>
      <article className="metric-card">
        <span>Parent API</span>
        <strong>{apiOnline ? 'Online' : 'Unknown'}</strong>
        <small>{status}</small>
      </article>
      <article className="metric-card">
        <span>Modules</span>
        <strong>{moduleCount}</strong>
        <small>Registered child systems</small>
      </article>
    </section>
  );
}
