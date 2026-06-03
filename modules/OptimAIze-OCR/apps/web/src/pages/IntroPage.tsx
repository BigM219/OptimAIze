import { HealthResponse, HistoryDocument, ModelListResponse } from '../shared/api/ocrApi';

type Props = {
  health: HealthResponse | null;
  models: ModelListResponse | null;
  documents: HistoryDocument[];
  status: string;
};

export function IntroPage({ health, models, documents, status }: Props) {
  return (
    <div className="page-shell">
      <section className="hero terminal-window">
        <div className="terminal-titlebar" aria-hidden="true">
          <span className="terminal-dot terminal-dot--red" />
          <span className="terminal-dot terminal-dot--yellow" />
          <span className="terminal-dot terminal-dot--green" />
          <code>~/optimaize/modules/OptimAIze-OCR</code>
        </div>
        <div className="hero-copy terminal-pane">
          <p className="eyebrow">AI document workbench</p>
          <p className="terminal-command"><span>root@optimaize-ocr:~$</span> ./init_ocr_engine --mode=layout-aware</p>
          <p className="terminal-boot">&gt; Connecting to document matrix... <strong>OK</strong></p>
          <h1 aria-label="Extract documents through a focused OCR production UI.">
            <span className="glitch" data-text="Extract documents through a focused OCR production UI." aria-hidden="true">Extract documents through a focused OCR production UI.</span>
          </h1>
          <p>
            Upload images, choose OCR models, inspect markdown and regions, then keep durable results in the child module history database.
          </p>
          <div className="hero-actions">
            <a className="primary-action" href="#workspace">Start extraction</a>
            <a className="secondary-action" href="#history">Open history</a>
          </div>
        </div>
        <div className="workbench-card" aria-label="OCR workbench status">
          <div className="workbench-card__top">
            <span className="status-dot" />
            <strong>OCR pipeline</strong>
            <small>{health?.ok ? 'Online' : 'Waiting'}</small>
          </div>
          <div className="pipeline-steps">
            <div><span>01</span><strong>Image upload</strong></div>
            <div><span>02</span><strong>Layout-aware OCR</strong></div>
            <div><span>03</span><strong>Markdown + regions</strong></div>
            <div><span>04</span><strong>History DB</strong></div>
          </div>
          <p>{status}</p>
        </div>
      </section>

      <section className="status-grid">
        <article className="metric-card">
          <span>Backend</span>
          <strong>{health?.ok ? 'Online' : 'Unknown'}</strong>
          <small>{status}</small>
        </article>
        <article className="metric-card">
          <span>OCR models</span>
          <strong>{models?.models.length ?? 0}</strong>
          <small>{models?.default_model ?? 'Waiting for API'}</small>
        </article>
        <article className="metric-card">
          <span>History</span>
          <strong>{documents.length}</strong>
          <small>SQLite-backed documents ready for reuse</small>
        </article>
      </section>
    </div>
  );
}
