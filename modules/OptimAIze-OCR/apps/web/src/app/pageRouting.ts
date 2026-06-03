export type Page = 'intro' | 'workspace' | 'history';

export function pageFromHash(): Page {
  const hash = window.location.hash.replace('#', '');
  if (hash === 'workspace' || hash === 'history') return hash;
  return 'intro';
}
