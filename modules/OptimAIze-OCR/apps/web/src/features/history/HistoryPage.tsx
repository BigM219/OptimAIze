import { HistoryDocument } from '../../shared/api/ocrApi';

type Props = {
  documents: HistoryDocument[];
};

export function HistoryPage({ documents }: Props) {
  return (
    <section className="panel">
      <div className="section-head">
        <div>
          <p className="eyebrow">History</p>
          <h2>Durable OCR documents</h2>
        </div>
        <span className="pill">{documents.length} documents</span>
      </div>
      <div className="history-list">
        {documents.length === 0 ? (
          <p className="empty-state">No history documents found in the configured SQLite database.</p>
        ) : documents.map((document) => (
          <article key={document.document_id} className="history-item">
            <div>
              <strong>{document.document_id}</strong>
              <span>{document.page_count} page(s)</span>
            </div>
            <code>{document.image_sha256.slice(0, 18)}...</code>
          </article>
        ))}
      </div>
    </section>
  );
}
