import { ModelListResponse, RuntimeConfigResponse } from '../../../shared/api/ocrApi';
import { SingleImageOCRState } from '../hooks/useSingleImageOCR';
import { formatBytes } from '../lib/format';

type Props = {
  models: ModelListResponse | null;
  ocr: SingleImageOCRState;
  activeTab: string;
  runtimeConfig?: RuntimeConfigResponse | null;
};

export function DocumentConfigTab({ models, ocr }: Props) {
  return (
    <div className="tab-panel stack" role="tabpanel">
      <label htmlFor="ocr-image">
        Image
        <input id="ocr-image" name="image" type="file" accept="image/*" onChange={(event) => ocr.setImage(event.target.files?.[0] ?? null)} />
      </label>

      <div className="file-meta" aria-label="Selected file metadata">
        <div><span>Name</span><strong>{ocr.image?.name ?? 'No file selected'}</strong></div>
        <div><span>Type</span><strong>{ocr.image?.type || '—'}</strong></div>
        <div><span>Size</span><strong>{ocr.image ? formatBytes(ocr.image.size) : '—'}</strong></div>
      </div>

      <label htmlFor="ocr-model">
        OCR model
        <select id="ocr-model" name="modelType" value={ocr.modelType} onChange={(event) => ocr.setModelType(event.target.value)}>
          {(models?.models ?? ['falcon-ocr']).map((model) => (
            <option key={model} value={model}>{model}</option>
          ))}
        </select>
      </label>
    </div>
  );
}

export function LayoutConfigTab({ ocr }: Props) {
  const config = ocr.config;
  const supportsFullPage = ocr.modelType === 'dots-mocr';

  return (
    <div className="tab-panel stack" role="tabpanel">
      <label htmlFor="layout-model">
        Layout model
        <input
          id="layout-model"
          name="layoutModel"
          value={config.layoutModel}
          onChange={(event) => ocr.setConfig((current) => ({ ...current, layoutModel: event.target.value }))}
        />
      </label>

      <label htmlFor="layout-threshold">
        Layout threshold: {config.layoutThreshold.toFixed(2)}
        <input
          id="layout-threshold"
          name="layoutThreshold"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={config.layoutThreshold}
          onChange={(event) => ocr.setConfig((current) => ({ ...current, layoutThreshold: Number(event.target.value) }))}
        />
      </label>

      <label className="inline option-row" htmlFor="skip-layout">
        <input
          id="skip-layout"
          name="skipLayout"
          type="checkbox"
          checked={supportsFullPage && config.skipLayout}
          disabled={!supportsFullPage}
          onChange={(event) => ocr.setConfig((current) => ({ ...current, skipLayout: event.target.checked }))}
        />
        <span>Skip layout detector and use full-page mode {supportsFullPage ? '' : '(Dots-MOCR only)'}</span>
      </label>

      <label htmlFor="full-page-mode">
        Full-page mode
        <select
          id="full-page-mode"
          name="fullPageMode"
          value={config.fullPageMode}
          disabled={!supportsFullPage}
          onChange={(event) => ocr.setConfig((current) => ({ ...current, fullPageMode: event.target.value }))}
        >
          <option value="layout">layout</option>
          <option value="svg">svg</option>
        </select>
      </label>
    </div>
  );
}

export function RuntimeConfigTab({ ocr, runtimeConfig }: Props) {
  const config = ocr.config;
  const threadRange = runtimeConfig?.threads;
  const cpuRange = runtimeConfig?.cpu_percent;

  return (
    <div className="tab-panel stack" role="tabpanel">
      <div className="config-grid">
        <label htmlFor="runtime-threads">
          CPU threads
          <input
            id="runtime-threads"
            name="threads"
            type="number"
            min={threadRange?.min ?? 1}
            max={threadRange?.max ?? undefined}
            placeholder="Scanning..."
            value={config.threads ?? ''}
            disabled={!threadRange}
            onChange={(event) => ocr.setConfig((current) => {
              if (!event.target.value) return { ...current, threads: null };
              const nextValue = Number(event.target.value);
              const maxThreads = threadRange?.max ?? nextValue;
              return { ...current, threads: Math.min(Math.max(nextValue, 1), maxThreads) };
            })}
          />
          <small>{threadRange ? `Range: ${threadRange.min}–${threadRange.max}. Preselected: ${config.threads ?? threadRange.recommended ?? threadRange.max}.` : 'Scanning CPU thread range...'}</small>
        </label>

        <label htmlFor="runtime-cpu-percent">
          CPU budget (%)
          <input
            id="runtime-cpu-percent"
            name="cpuPercent"
            type="number"
            min={cpuRange?.min ?? 1}
            max={cpuRange?.max ?? 100}
            placeholder="Scanning..."
            value={config.cpuPercent ?? ''}
            onChange={(event) => ocr.setConfig((current) => ({
              ...current,
              cpuPercent: event.target.value ? Number(event.target.value) : null,
            }))}
          />
          <small>{cpuRange ? `Range: ${cpuRange.min}–${cpuRange.max}%. Leave empty only if you want backend default.` : 'Scanning CPU budget range...'}</small>
        </label>
      </div>

      <label htmlFor="auto-runtime">
        Auto runtime policy
        <select
          id="auto-runtime"
          name="autoRuntime"
          value={config.autoRuntime}
          onChange={(event) => ocr.setConfig((current) => ({ ...current, autoRuntime: event.target.value }))}
        >
          <option value="off">off</option>
          <option value="conservative">conservative</option>
          <option value="speed">speed</option>
          <option value="experimental">experimental</option>
        </select>
      </label>

      <div className="kernel-summary">
        <span>Runtime payload</span>
        <code>threads={config.threads ?? 'default'} · cpu={config.cpuPercent ?? 'default'} · policy={config.autoRuntime}</code>
      </div>
    </div>
  );
}

export function PrecisionConfigTab({ ocr }: Props) {
  const config = ocr.config;

  return (
    <div className="tab-panel stack" role="tabpanel">
      <label htmlFor="quantize-int8">
        INT8 quantization
        <select
          id="quantize-int8"
          name="quantizeInt8"
          value={config.quantizeInt8}
          onChange={(event) => ocr.setConfig((current) => ({ ...current, quantizeInt8: event.target.value }))}
        >
          <option value="auto">default per model</option>
          <option value="true">force on</option>
          <option value="false">force off</option>
        </select>
      </label>

      <label htmlFor="quantize-mode">
        Quantize mode
        <select
          id="quantize-mode"
          name="quantizeMode"
          value={config.quantizeMode}
          onChange={(event) => ocr.setConfig((current) => ({ ...current, quantizeMode: event.target.value }))}
        >
          <option value="selective">selective</option>
          <option value="auto">auto</option>
          <option value="none">none</option>
          <option value="full">full</option>
          <option value="fp16">fp16</option>
          <option value="lm_head">lm_head</option>
          <option value="mlp">mlp</option>
          <option value="mlp_lm_head">mlp_lm_head</option>
        </select>
      </label>

      <div className="kernel-summary">
        <span>Precision payload</span>
        <code>int8={config.quantizeInt8 === 'auto' ? 'default' : config.quantizeInt8} · mode={config.quantizeMode}</code>
      </div>
    </div>
  );
}

export function BackendConfigTab({ ocr }: Props) {
  const config = ocr.config;

  return (
    <div className="tab-panel stack" role="tabpanel">
      <label className="inline option-row" htmlFor="use-optimized-dots">
        <input
          id="use-optimized-dots"
          name="useOptimizedDots"
          type="checkbox"
          checked={config.useOptimizedDots}
          onChange={(event) => ocr.setConfig((current) => ({ ...current, useOptimizedDots: event.target.checked }))}
        />
        <span>Use optimized Dots-MOCR backend</span>
      </label>

      <label className="inline option-row" htmlFor="dots-fuse-mlp-swiglu">
        <input
          id="dots-fuse-mlp-swiglu"
          name="dotsFuseMlpSwiglu"
          type="checkbox"
          checked={config.dotsFuseMlpSwiglu}
          onChange={(event) => ocr.setConfig((current) => ({ ...current, dotsFuseMlpSwiglu: event.target.checked }))}
        />
        <span>Dots fused INT8 SwiGLU MLP kernel</span>
      </label>

      <label className="inline option-row" htmlFor="dots-int8-lm-head">
        <input
          id="dots-int8-lm-head"
          name="dotsInt8LmHead"
          type="checkbox"
          checked={config.dotsInt8LmHead}
          onChange={(event) => ocr.setConfig((current) => ({ ...current, dotsInt8LmHead: event.target.checked }))}
        />
        <span>Dots INT8 lm_head GEMV kernel</span>
      </label>

      <label htmlFor="paddle-table-prompt">
        Paddle table prompt
        <select
          id="paddle-table-prompt"
          name="paddleTablePrompt"
          value={config.paddleTablePrompt}
          onChange={(event) => ocr.setConfig((current) => ({ ...current, paddleTablePrompt: event.target.value }))}
        >
          <option value="fast">fast</option>
          <option value="balanced">balanced</option>
          <option value="verbose">verbose</option>
        </select>
      </label>
    </div>
  );
}

export function OutputConfigTab({ ocr }: Props) {
  const config = ocr.config;

  return (
    <div className="tab-panel stack" role="tabpanel">
      <label className="inline option-row" htmlFor="save-crops">
        <input
          id="save-crops"
          name="saveCrops"
          type="checkbox"
          checked={config.saveCrops}
          onChange={(event) => ocr.setConfig((current) => ({ ...current, saveCrops: event.target.checked }))}
        />
        <span>Save crops for inspection</span>
      </label>

      <div className="kernel-summary">
        <span>Output payload</span>
        <code>save_crops={String(config.saveCrops)}</code>
      </div>
    </div>
  );
}
