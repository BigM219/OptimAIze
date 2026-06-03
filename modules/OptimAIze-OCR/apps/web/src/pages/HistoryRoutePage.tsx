import { HistoryPage } from '../features/history/HistoryPage';
import { HistoryDocument } from '../shared/api/ocrApi';

type Props = {
  documents: HistoryDocument[];
};

export function HistoryRoutePage({ documents }: Props) {
  return (
    <div className="page-shell">
      <section className="page-heading" aria-labelledby="history-title">
        <p className="eyebrow">History</p>
        <h2 id="history-title">Durable OCR document library.</h2>
        <p className="muted">Review documents saved by the OCR backend without mixing history into the active extraction page.</p>
      </section>
      <HistoryPage documents={documents} />
    </div>
  );
}
