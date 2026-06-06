import { apiGet, apiPost, API_BASE } from './client';

/** Absolute URL to the parent API health endpoint (for direct browser links). */
export const HEALTH_URL = `${API_BASE}/api/v1/health`;

export type HealthResponse = {
  ok: boolean;
  service: string;
  version: string;
};

export type ModuleStatus = {
  id: string;
  name: string;
  kind: string;
  available: boolean;
  source_available: boolean;
  ui_available: boolean;
  api_available: boolean;
  web_available: boolean;
  path: string;
  message: string;
  web_url: string | null;
  api_url: string | null;
};

export type ModulesResponse = {
  modules: ModuleStatus[];
};

export type LaunchResult = {
  ok: boolean;
  pid: number | null;
  url: string | null;
  message: string;
};

export function getHealth() {
  return apiGet<HealthResponse>('/api/v1/health');
}

export function getModules() {
  return apiGet<ModulesResponse>('/api/v1/modules');
}

export function launchOcrLegacyUi(serverPort = 7860) {
  return apiPost<LaunchResult>('/api/v1/modules/ocr/launch-ui', {
    server_port: serverPort,
    share: false,
    inbrowser: false,
  });
}
