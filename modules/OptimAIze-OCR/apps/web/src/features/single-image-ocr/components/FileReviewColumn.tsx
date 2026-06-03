import { PointerEvent, useEffect, useMemo, useRef, useState } from 'react';
import { SingleImageOCRState } from '../hooks/useSingleImageOCR';
import { normalizeBbox } from '../lib/regions';

type Props = {
  ocr: SingleImageOCRState;
};

type PanState = {
  dragging: boolean;
  startX: number;
  startY: number;
  originX: number;
  originY: number;
};

const MIN_ZOOM = 0.35;
const MAX_ZOOM = 4;

function clampZoom(value: number) {
  return Math.min(Math.max(value, MIN_ZOOM), MAX_ZOOM);
}

function fitZoom(frame: HTMLDivElement | null, width: number, height: number) {
  if (!frame || width <= 0 || height <= 0) return 0.65;
  const frameWidth = Math.max(1, frame.clientWidth - 36);
  const frameHeight = Math.max(1, frame.clientHeight - 36);
  return clampZoom(Number(Math.min(frameWidth / width, frameHeight / height).toFixed(2)));
}

export function FileReviewColumn({ ocr }: Props) {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const [zoom, setZoom] = useState(0.65);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [drag, setDrag] = useState<PanState | null>(null);

  const bboxes = useMemo(() => (ocr.result?.regions ?? [])
    .map((region) => ({ region, bbox: normalizeBbox(region.bbox) }))
    .filter((item) => item.bbox !== null), [ocr.result]);

  useEffect(() => {
    const frame = frameRef.current;
    if (!frame || !ocr.previewUrl) return undefined;

    function handleWheel(event: WheelEvent) {
      event.preventDefault();
      const direction = event.deltaY > 0 ? -0.12 : 0.12;
      setZoom((current) => clampZoom(Number((current + direction).toFixed(2))));
    }

    frame.addEventListener('wheel', handleWheel, { passive: false });
    return () => frame.removeEventListener('wheel', handleWheel);
  }, [ocr.previewUrl]);

  function resetView() {
    setZoom(fitZoom(frameRef.current, imageSize.width, imageSize.height));
    setPan({ x: 0, y: 0 });
  }

  function onPointerDown(event: PointerEvent<HTMLDivElement>) {
    if (!ocr.previewUrl) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrag({ dragging: true, startX: event.clientX, startY: event.clientY, originX: pan.x, originY: pan.y });
  }

  function onPointerMove(event: PointerEvent<HTMLDivElement>) {
    if (!drag?.dragging) return;
    setPan({ x: drag.originX + event.clientX - drag.startX, y: drag.originY + event.clientY - drag.startY });
  }

  function onPointerUp(event: PointerEvent<HTMLDivElement>) {
    event.currentTarget.releasePointerCapture(event.pointerId);
    setDrag(null);
  }

  return (
    <article className="workspace-column review-column terminal-panel">
      <div className="terminal-panel__head" aria-hidden="true">
        <code>preview.tty</code>
        <span>bbox viewport</span>
      </div>
      <div className="column-head">
        <div>
          <p className="eyebrow">File review</p>
          <p className="terminal-command">$ open --page 1 --overlay bbox</p>
          <h3>Current page</h3>
        </div>
        <span className="pill">Page 1</span>
      </div>

      <div className="review-toolbar" aria-label="Image review controls">
        <button type="button" onClick={() => setZoom((current) => clampZoom(current - 0.2))} disabled={!ocr.previewUrl}>−</button>
        <button type="button" onClick={resetView} disabled={!ocr.previewUrl}>{Math.round(zoom * 100)}%</button>
        <button type="button" onClick={() => setZoom((current) => clampZoom(current + 0.2))} disabled={!ocr.previewUrl}>+</button>
      </div>

      <div
        ref={frameRef}
        className={`file-review-frame ${drag ? 'is-dragging' : ''}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => setDrag(null)}
      >
        {ocr.previewUrl ? (
          <div
            className="review-stage"
            style={{
              width: imageSize.width || undefined,
              height: imageSize.height || undefined,
              transform: `translate(-50%, -50%) translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            }}
          >
            <img
              src={ocr.previewUrl}
              alt={ocr.image ? `Preview of ${ocr.image.name}` : 'Selected OCR file preview'}
              draggable={false}
              onLoad={(event) => {
                const nextSize = { width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight };
                setImageSize(nextSize);
                setZoom(fitZoom(frameRef.current, nextSize.width, nextSize.height));
                setPan({ x: 0, y: 0 });
              }}
            />
            <div className="bbox-layer" aria-label="OCR bounding boxes">
              {bboxes.map(({ region, bbox }) => {
                if (!bbox) return null;
                const isSelected = ocr.selectedRegionIndex === region.index;
                const isMatched = ocr.matchingRegionIndexes.includes(region.index);
                return (
                  <button
                    key={region.index}
                    type="button"
                    className={`bbox ${isSelected ? 'selected' : ''} ${isMatched ? 'matched' : ''}`}
                    style={{ left: bbox.x, top: bbox.y, width: bbox.width, height: bbox.height }}
                    title={`#${region.index} ${region.category || 'region'}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      ocr.setSelectedRegionIndex(region.index);
                    }}
                  />
                );
              })}
            </div>
          </div>
        ) : (
          <div className="empty-preview">
            <strong>No file selected</strong>
            <span>Upload an image to review the page before extraction.</span>
          </div>
        )}
      </div>

      <div className="review-note">
        <strong>{ocr.image?.name ?? 'Waiting for upload'}</strong>
        <span>Single-image API mode · {bboxes.length} bboxes · wheel zoom / drag pan</span>
      </div>
    </article>
  );
}
