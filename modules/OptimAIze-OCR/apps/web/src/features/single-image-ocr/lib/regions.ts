import { OCRRegion } from '../../../shared/api/ocrApi';

export type NormalizedBbox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function toNumber(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function normalizeBbox(bbox: OCRRegion['bbox']): NormalizedBbox | null {
  if (Array.isArray(bbox) && bbox.length >= 4) {
    const [x1, y1, x2, y2] = bbox.map(toNumber);
    if (x1 === null || y1 === null || x2 === null || y2 === null) return null;
    return { x: Math.min(x1, x2), y: Math.min(y1, y2), width: Math.abs(x2 - x1), height: Math.abs(y2 - y1) };
  }

  if (!isRecord(bbox)) return null;

  const x = toNumber(bbox.x);
  const y = toNumber(bbox.y);
  const width = toNumber(bbox.width ?? bbox.w);
  const height = toNumber(bbox.height ?? bbox.h);
  if (x !== null && y !== null && width !== null && height !== null) return { x, y, width, height };

  const x1 = toNumber(bbox.x1 ?? bbox.left);
  const y1 = toNumber(bbox.y1 ?? bbox.top);
  const x2 = toNumber(bbox.x2 ?? bbox.right);
  const y2 = toNumber(bbox.y2 ?? bbox.bottom);
  if (x1 === null || y1 === null || x2 === null || y2 === null) return null;
  return { x: Math.min(x1, x2), y: Math.min(y1, y2), width: Math.abs(x2 - x1), height: Math.abs(y2 - y1) };
}

export function bboxLabel(bbox: NormalizedBbox | null) {
  if (!bbox) return 'bbox unavailable';
  return `x=${Math.round(bbox.x)}, y=${Math.round(bbox.y)}, w=${Math.round(bbox.width)}, h=${Math.round(bbox.height)}`;
}
