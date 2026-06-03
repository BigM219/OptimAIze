import { SingleImageOCRPage } from '../features/single-image-ocr/SingleImageOCRPage';
import { ModelListResponse, RuntimeConfigResponse } from '../shared/api/ocrApi';

type Props = {
  models: ModelListResponse | null;
  runtimeConfig: RuntimeConfigResponse | null;
};

export function WorkspacePage({ models, runtimeConfig }: Props) {
  return (
    <div className="page-shell">
      <section className="page-heading" aria-labelledby="workspace-title">
        <p className="eyebrow">OCR workspace</p>
        <h2 id="workspace-title">Manage, review, and inspect OCR output in one focused screen.</h2>
        <p className="muted">This page contains the active document workflow: controls, file preview, and result display.</p>
      </section>
      <SingleImageOCRPage models={models} runtimeConfig={runtimeConfig} />
    </div>
  );
}
