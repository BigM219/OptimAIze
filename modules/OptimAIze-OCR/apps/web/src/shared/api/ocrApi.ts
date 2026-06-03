import { apiGet, apiPostForm } from './client';

export type HealthResponse = {
  ok: boolean;
  service: string;
  version: string;
  history_db?: string;
};

export type ModelListResponse = {
  models: string[];
  default_model: string;
};

export type RuntimeRange = {
  default: number | null;
  min: number;
  max: number;
  recommended: number | null;
};

export type RuntimeConfigResponse = {
  logical_cpus: number;
  threads: RuntimeRange;
  cpu_percent: RuntimeRange;
  labels: Record<string, string>;
};

export type OCRBbox = number[] | Record<string, unknown> | null;

export type OCRRegion = {
  index: number;
  category: string;
  bbox: OCRBbox;
  score: unknown;
  text: string;
};

export type SingleImageOCRResponse = {
  markdown: string;
  html: string;
  regions: OCRRegion[];
  timings: Record<string, unknown>;
  output_dir: string;
  image_name: string;
};

export type HistoryDocument = {
  document_id: string;
  source_path: string | null;
  image_sha256: string;
  page_count: number;
  created_at: number;
  updated_at: number;
  metadata: Record<string, unknown>;
};

export type OCRConfig = {
  layoutModel: string;
  layoutThreshold: number;
  skipLayout: boolean;
  fullPageMode: string;
  threads: number | null;
  cpuPercent: number | null;
  autoRuntime: string;
  quantizeInt8: string;
  quantizeMode: string;
  useOptimizedDots: boolean;
  dotsFuseMlpSwiglu: boolean;
  dotsInt8LmHead: boolean;
  paddleTablePrompt: string;
  saveCrops: boolean;
};

export async function getHealth() {
  return apiGet<HealthResponse>('/api/v1/health');
}

export async function getModels() {
  return apiGet<ModelListResponse>('/api/v1/ocr/models');
}

export async function getRuntimeConfig() {
  return apiGet<RuntimeConfigResponse>('/api/v1/ocr/runtime-config');
}

export async function getHistoryDocuments() {
  return apiGet<{ documents: HistoryDocument[] }>('/api/v1/history/documents');
}

export async function runSingleImageOCR(input: {
  image: File;
  modelType: string;
  config: OCRConfig;
}) {
  const form = new FormData();
  form.append('image', input.image);
  form.append('model_type', input.modelType);
  form.append('layout_model', input.config.layoutModel);
  form.append('layout_threshold', String(input.config.layoutThreshold));
  form.append('skip_layout', String(input.config.skipLayout));
  form.append('full_page_mode', input.config.fullPageMode);
  if (input.config.threads !== null) form.append('threads', String(input.config.threads));
  if (input.config.cpuPercent !== null) form.append('cpu_percent', String(input.config.cpuPercent));
  if (input.config.quantizeInt8 !== 'auto') form.append('quantize_int8', input.config.quantizeInt8);
  form.append('quantize_mode', input.config.quantizeMode);
  form.append('auto_runtime', input.config.autoRuntime);
  form.append('use_optimized_dots', String(input.config.useOptimizedDots));
  form.append('dots_fuse_mlp_swiglu', String(input.config.dotsFuseMlpSwiglu));
  form.append('dots_int8_lm_head', String(input.config.dotsInt8LmHead));
  form.append('paddle_table_prompt', input.config.paddleTablePrompt);
  form.append('save_crops', String(input.config.saveCrops));
  return apiPostForm<SingleImageOCRResponse>('/api/v1/ocr/single-image', form);
}
