import { ModelListResponse, RuntimeConfigResponse } from '../../shared/api/ocrApi';
import { FileReviewColumn } from './components/FileReviewColumn';
import { ManageColumn } from './components/ManageColumn';
import { ResultDisplayColumn } from './components/ResultDisplayColumn';
import { useSingleImageOCR } from './hooks/useSingleImageOCR';

type Props = {
  models: ModelListResponse | null;
  runtimeConfig: RuntimeConfigResponse | null;
};

export function SingleImageOCRPage({ models, runtimeConfig }: Props) {
  const ocr = useSingleImageOCR(runtimeConfig);

  return (
    <section className="terminal-workspace" aria-label="OCR extraction workspace">
      <div className="terminal-titlebar" aria-hidden="true">
        <span className="terminal-dot terminal-dot--red" />
        <span className="terminal-dot terminal-dot--yellow" />
        <span className="terminal-dot terminal-dot--green" />
        <code>ocr-workspace tty://single-image</code>
      </div>
      <div className="ocr-workspace-grid">
        <ManageColumn models={models} ocr={ocr} runtimeConfig={runtimeConfig} />
        <FileReviewColumn ocr={ocr} />
        <ResultDisplayColumn ocr={ocr} />
      </div>
    </section>
  );
}
