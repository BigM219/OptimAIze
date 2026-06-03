import { ReactNode } from 'react';
import { OCRRegion } from '../../../shared/api/ocrApi';
import { SingleImageOCRState } from '../hooks/useSingleImageOCR';
import { bboxLabel, normalizeBbox } from '../lib/regions';
import { HtmlPreview } from './HtmlPreview';

type Props = {
  ocr: SingleImageOCRState;
};

function highlightedText(text: string, query: string): ReactNode {
  const trimmedQuery = query.trim();
  if (!trimmedQuery || !text) return text || 'No text';
  const lowerText = text.toLocaleLowerCase();
  const lowerQuery = trimmedQuery.toLocaleLowerCase();
  const parts: ReactNode[] = [];
  let cursor = 0;
  let matchIndex = lowerText.indexOf(lowerQuery);

  while (matchIndex >= 0) {
    if (matchIndex > cursor) parts.push(text.slice(cursor, matchIndex));
    const end = matchIndex + trimmedQuery.length;
    parts.push(<mark key={`${matchIndex}-${end}`}>{text.slice(matchIndex, end)}</mark>);
    cursor = end;
    matchIndex = lowerText.indexOf(lowerQuery, cursor);
  }

  if (cursor < text.length) parts.push(text.slice(cursor));
  return parts;
}

function formatScore(score: OCRRegion['score']) {
  return typeof score === 'number' ? score.toFixed(3) : String(score ?? '—');
}

export function ResultDisplayColumn({ ocr }: Props) {
  const regions = ocr.result?.regions ?? [];
  const matchedCount = ocr.searchQuery.trim() ? ocr.matchingRegionIndexes.length : regions.length;

  return (
    <article className="workspace-column result-column terminal-panel">
      <div className="terminal-panel__head" aria-hidden="true">
        <code>stdout.log</code>
        <span>render/json</span>
      </div>
      <div className="column-head">
        <div>
          <p className="eyebrow">Result display</p>
          <p className="terminal-command">$ tail --format html markdown json</p>
          <h3>Preview, raw text, JSON.</h3>
        </div>
        <span className="pill">{regions.length} regions</span>
      </div>

      <label className="result-search" htmlFor="ocr-result-search">
        Search OCR text
        <input
          id="ocr-result-search"
          value={ocr.searchQuery}
          placeholder="Find text and highlight bbox..."
          onChange={(event) => ocr.setSearchQuery(event.target.value)}
        />
      </label>
      <p className="search-summary">{ocr.result ? `${matchedCount} matching regions` : 'Run OCR to enable search and bbox highlights.'}</p>

      <div className="result-tabs" role="tablist" aria-label="OCR result views">
        <button type="button" className={ocr.activeResultView === 'preview' ? 'active' : ''} onClick={() => ocr.setActiveResultView('preview')}>Preview</button>
        <button type="button" className={ocr.activeResultView === 'raw' ? 'active' : ''} onClick={() => ocr.setActiveResultView('raw')}>Raw markdown</button>
        <button type="button" className={ocr.activeResultView === 'json' ? 'active' : ''} onClick={() => ocr.setActiveResultView('json')}>Page JSON</button>
      </div>

      <div className="result-display">
        {ocr.activeResultView === 'preview' && (
          <div className="markdown-preview">
            <HtmlPreview html={ocr.result?.html} markdown={ocr.result?.markdown ?? ''} />
          </div>
        )}
        {ocr.activeResultView === 'raw' && <pre>{ocr.result?.markdown || 'Raw markdown will appear here after a successful extraction.'}</pre>}
        {ocr.activeResultView === 'json' && <pre>{ocr.pageJson}</pre>}
      </div>

      <details className="advanced-regions">
        <summary>
          <span>Advanced crop details</span>
          <strong>{regions.length} regions</strong>
        </summary>
        <div className="regions compact-regions">
          {regions.length ? regions.map((region) => {
            const bbox = normalizeBbox(region.bbox);
            const isSelected = ocr.selectedRegionIndex === region.index;
            const isMatched = ocr.matchingRegionIndexes.includes(region.index);
            return (
              <button
                key={region.index}
                type="button"
                className={`region-detail ${isSelected ? 'selected' : ''} ${isMatched ? 'matched' : ''}`}
                onClick={() => ocr.setSelectedRegionIndex(region.index)}
              >
                <strong>#{region.index} {region.category || 'region'}</strong>
                <small>score={formatScore(region.score)} · {bboxLabel(bbox)}</small>
                <span>{highlightedText(region.text, ocr.searchQuery)}</span>
              </button>
            );
          }) : <p className="muted">Region details will appear after a successful extraction.</p>}
        </div>
      </details>
    </article>
  );
}
