import { useEffect, useMemo, useRef, useState } from 'react';
import { ModelListResponse, RuntimeConfigResponse } from '../../../shared/api/ocrApi';
import { SingleImageOCRState } from '../hooks/useSingleImageOCR';
import {
  BackendConfigTab,
  DocumentConfigTab,
  LayoutConfigTab,
  OutputConfigTab,
  PrecisionConfigTab,
  RuntimeConfigTab,
} from './ConfigTabs';

type Props = {
  models: ModelListResponse | null;
  ocr: SingleImageOCRState;
  runtimeConfig: RuntimeConfigResponse | null;
};

type ManageTab = 'document' | 'layout' | 'runtime' | 'precision' | 'backend' | 'output';

const tabs: { id: ManageTab; label: string }[] = [
  { id: 'document', label: 'Document' },
  { id: 'layout', label: 'Layout' },
  { id: 'runtime', label: 'Runtime' },
  { id: 'precision', label: 'Precision' },
  { id: 'backend', label: 'Backend' },
  { id: 'output', label: 'Output' },
];

function phaseProgress(ocr: SingleImageOCRState) {
  if (ocr.busy) {
    if (ocr.progress < 72) return [{ label: 'UPLOAD_PAYLOAD', value: Math.max(12, ocr.progress), state: 'run' }];
    if (!ocr.config.skipLayout && ocr.progress < 88) return [{ label: 'LAYOUT_SCAN', value: Math.min(92, ocr.progress), state: 'run' }];
    return [{ label: 'VLM_DECODE', value: Math.max(18, Math.min(96, ocr.progress)), state: 'run' }];
  }

  if (!ocr.result) return [];

  const cropCount = ocr.result.regions.length;
  if (cropCount <= 1) return [];

  return [{ label: `CROP_QUEUE · ${cropCount} regions completed`, value: 100, state: 'ok' }];
}

function outputLines(ocr: SingleImageOCRState, activeTab: ManageTab) {
  const lines = [
    { kind: 'info', text: '[SYS] OptimAIze-OCR terminal ready.' },
    { kind: 'info', text: `[UI] Active config pane: ${activeTab.toUpperCase()}` },
    ocr.image
      ? { kind: 'ok', text: `[INP] Mounted image stream: ${ocr.image.name}` }
      : { kind: 'warn', text: '[INP] Waiting for image stream...' },
    { kind: ocr.busy ? 'warn' : 'info', text: `[CFG] Model=${ocr.modelType} Threads=${ocr.config.threads ?? 'backend'} Layout=${ocr.config.skipLayout ? 'skip' : 'pp-doclayout'}` },
  ];

  if (ocr.searchQuery.trim()) lines.push({ kind: 'data', text: `[USR] Search query armed: "${ocr.searchQuery.trim()}"` });
  if (ocr.selectedRegionIndex !== null) lines.push({ kind: 'data', text: `[USR] Selected bbox region: #${ocr.selectedRegionIndex}` });

  if (ocr.busy) {
    lines.push(
      { kind: 'warn', text: '[PRO] Uploading payload and starting backend inference...' },
      { kind: 'info', text: '[OCR] Layout scan active. Detecting readable regions...' },
      { kind: 'info', text: '[OCR] Crop queue active. Progress is shown only for processing phases.' },
      { kind: 'info', text: '[OCR] VLM decode running. Waiting for backend response...' },
    );
  } else if (ocr.result) {
    const cropCount = ocr.result.regions.length;
    lines.push(
      { kind: 'ok', text: `[SYS] Extracted ${cropCount} regions from ${ocr.result.image_name}.` },
      { kind: 'data', text: `[TIM] OCR=${Number(ocr.result.timings.ocr_time_total ?? 0).toFixed(3)}s Layout=${Number(ocr.result.timings.layout_time ?? 0).toFixed(3)}s` },
      cropCount > 1
        ? { kind: 'ok', text: `[DONE] Multi-crop queue completed: ${cropCount} crops rendered.` }
        : { kind: 'ok', text: '[DONE] Single output buffer rendered.' },
    );
  } else {
    lines.push({ kind: 'info', text: '[SYS] Press RUN OCR to execute pipeline.' });
  }

  lines.push({ kind: ocr.progress === 100 ? 'ok' : ocr.progress > 0 ? 'warn' : 'info', text: `[BUF] ${ocr.status}` });
  return lines;
}

export function ManageColumn({ models, ocr, runtimeConfig }: Props) {
  const [activeTab, setActiveTab] = useState<ManageTab>('document');
  const bufferRef = useRef<HTMLDivElement | null>(null);
  const lines = useMemo(() => outputLines(ocr, activeTab), [activeTab, ocr.busy, ocr.config, ocr.image, ocr.modelType, ocr.progress, ocr.result, ocr.searchQuery, ocr.selectedRegionIndex, ocr.status]);
  const phases = useMemo(() => phaseProgress(ocr), [ocr.busy, ocr.config.skipLayout, ocr.progress, ocr.result]);

  useEffect(() => {
    const buffer = bufferRef.current;
    if (!buffer) return;
    const distanceFromBottom = buffer.scrollHeight - buffer.scrollTop - buffer.clientHeight;
    if (distanceFromBottom < 80) buffer.scrollTop = buffer.scrollHeight;
  }, [lines, phases]);

  return (
    <form className="workspace-column manage-column terminal-panel" onSubmit={ocr.submit}>
      <div className="terminal-panel__head" aria-hidden="true">
        <code>manage.sh</code>
        <span>input/config</span>
      </div>
      <div>
        <p className="eyebrow">Manage</p>
        <p className="terminal-command">$ configure --image --model --runtime</p>
        <h3>Upload, select, configure.</h3>
        <p className="muted">Choose a document image and tune every backend-supported OCR option.</p>
      </div>

      <div className="manage-tab-grid" role="tablist" aria-label="OCR configuration tabs">
        {tabs.map((tab) => (
          <button key={tab.id} type="button" className={activeTab === tab.id ? 'active' : ''} onClick={() => setActiveTab(tab.id)}>
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'document' && <DocumentConfigTab models={models} ocr={ocr} activeTab={activeTab} />}
      {activeTab === 'layout' && <LayoutConfigTab models={models} ocr={ocr} activeTab={activeTab} />}
      {activeTab === 'runtime' && <RuntimeConfigTab models={models} ocr={ocr} activeTab={activeTab} runtimeConfig={runtimeConfig} />}
      {activeTab === 'precision' && <PrecisionConfigTab models={models} ocr={ocr} activeTab={activeTab} />}
      {activeTab === 'backend' && <BackendConfigTab models={models} ocr={ocr} activeTab={activeTab} />}
      {activeTab === 'output' && <OutputConfigTab models={models} ocr={ocr} activeTab={activeTab} />}

      <button className="primary-action" disabled={ocr.busy}>{ocr.busy ? 'Running OCR...' : 'Run OCR'}</button>

      <div className="output-buffer" aria-label="OCR processing output buffer">
        <div className="term-pane-label">// OUTPUT_BUFFER</div>
        <div ref={bufferRef} className="output-buffer__body" tabIndex={0} aria-label="Scrollable OCR terminal output">
          <div className="output-lines">
            {lines.map((line, index) => (
              <div className={`output-line ${line.kind}`} key={`${line.kind}-${index}`}>{line.text}</div>
            ))}
          </div>
          {phases.length > 0 && (
            <div className="phase-progress-list" aria-label="OCR phase progress">
              {phases.map((phase) => (
                <div className={`phase-progress phase-progress--${phase.state}`} key={phase.label}>
                  <div className="phase-progress__top">
                    <span>{phase.label}</span>
                    <strong>{Math.round(Math.min(100, Math.max(0, phase.value)))}%</strong>
                  </div>
                  <div className="progress-track">
                    <span style={{ width: `${Math.min(100, Math.max(0, phase.value))}%` }} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </form>
  );
}
