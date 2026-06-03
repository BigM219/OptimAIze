import { FormEvent, useEffect, useMemo, useState } from 'react';
import { OCRConfig, RuntimeConfigResponse, SingleImageOCRResponse, runSingleImageOCR } from '../../../shared/api/ocrApi';

export type ResultView = 'preview' | 'raw' | 'json';

export function useSingleImageOCR(runtimeConfig: RuntimeConfigResponse | null = null) {
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [modelType, setModelType] = useState('falcon-ocr');
  const [config, setConfig] = useState<OCRConfig>({
    layoutModel: 'PaddlePaddle/PP-DocLayoutV3_safetensors',
    layoutThreshold: 0.3,
    skipLayout: false,
    fullPageMode: 'layout',
    threads: null,
    cpuPercent: null,
    autoRuntime: 'off',
    quantizeInt8: 'auto',
    quantizeMode: 'selective',
    useOptimizedDots: true,
    dotsFuseMlpSwiglu: true,
    dotsInt8LmHead: true,
    paddleTablePrompt: 'fast',
    saveCrops: false,
  });
  const [result, setResult] = useState<SingleImageOCRResponse | null>(null);
  const [status, setStatus] = useState('Upload an image to run OCR through the backend API.');
  const [progress, setProgress] = useState(0);
  const [activeResultView, setActiveResultView] = useState<ResultView>('preview');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedRegionIndex, setSelectedRegionIndex] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!runtimeConfig) return;
    setConfig((current) => ({
      ...current,
      threads: current.threads ?? runtimeConfig.threads.recommended ?? runtimeConfig.threads.max,
      cpuPercent: current.cpuPercent ?? runtimeConfig.cpu_percent.recommended,
      quantizeInt8: current.quantizeInt8 === 'auto' ? 'false' : current.quantizeInt8,
    }));
  }, [runtimeConfig]);

  useEffect(() => {
    if (!image) {
      setPreviewUrl(null);
      return;
    }

    const nextUrl = URL.createObjectURL(image);
    setPreviewUrl(nextUrl);
    setResult(null);
    setSearchQuery('');
    setSelectedRegionIndex(null);
    setProgress((current) => Math.max(current, 20));
    setStatus(`Selected ${image.name}. Ready to run OCR.`);

    return () => URL.revokeObjectURL(nextUrl);
  }, [image]);

  const pageJson = useMemo(() => {
    if (!result) return 'Run OCR to inspect current page JSON.';
    return JSON.stringify({
      image_name: result.image_name,
      output_dir: result.output_dir,
      timings: result.timings,
      regions: result.regions,
    }, null, 2);
  }, [result]);

  const matchingRegionIndexes = useMemo(() => {
    const query = searchQuery.trim().toLocaleLowerCase();
    if (!query || !result) return [] as number[];
    return result.regions
      .filter((region) => region.text.toLocaleLowerCase().includes(query))
      .map((region) => region.index);
  }, [result, searchQuery]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const requestConfig = modelType === 'dots-mocr' ? config : { ...config, skipLayout: false, fullPageMode: 'layout' };
    if (!image) {
      setStatus('Please choose an image first.');
      setProgress(0);
      return;
    }
    setBusy(true);
    setProgress(65);
    setStatus('Running OCR on backend...');
    try {
      const nextResult = await runSingleImageOCR({ image, modelType, config: requestConfig });
      setResult(nextResult);
      setSearchQuery('');
      setSelectedRegionIndex(nextResult.regions[0]?.index ?? null);
      setProgress(100);
      setActiveResultView('preview');
      setStatus(`Done. ${nextResult.regions.length} regions extracted from ${nextResult.image_name}.`);
    } catch (error) {
      setProgress(20);
      setStatus(error instanceof Error ? error.message : 'OCR failed.');
    } finally {
      setBusy(false);
    }
  }

  return {
    activeResultView,
    busy,
    config,
    image,
    matchingRegionIndexes,
    modelType,
    pageJson,
    previewUrl,
    progress,
    result,
    searchQuery,
    selectedRegionIndex,
    setActiveResultView,
    setConfig,
    setImage,
    setModelType,
    setSearchQuery,
    setSelectedRegionIndex,
    status,
    submit,
  };
}

export type SingleImageOCRState = ReturnType<typeof useSingleImageOCR>;
