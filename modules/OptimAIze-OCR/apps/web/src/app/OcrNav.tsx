import { OcrTheme } from './App';
import { Page } from './pageRouting';

type Props = {
  page: Page;
  theme: OcrTheme;
  onThemeChange: (theme: OcrTheme) => void;
};

export function OcrNav({ page, theme, onThemeChange }: Props) {
  return (
    <nav className="top-nav">
      <a className="brand-mark logo" href="#intro" aria-label="OptimAIze OCR home">
        <span className="logo-bracket">[</span><strong>OPTIM<span>AI</span>ZE-OCR</strong><span className="logo-bracket">]</span>
        <small>v0.1.0</small>
      </a>
      <div className="nav-links" aria-label="OCR navigation">
        <a className={page === 'intro' ? 'active' : ''} href="#intro">Intro</a>
        <a className={page === 'workspace' ? 'active' : ''} href="#workspace">Workspace</a>
        <a className={page === 'history' ? 'active' : ''} href="#history">History</a>
      </div>
      <div className="nav-tools">
        <label className="theme-select">
          <span>Theme</span>
          <select value={theme} onChange={(event) => onThemeChange(event.target.value as OcrTheme)}>
            <option value="nebula">Nebula</option>
            <option value="andromeda">Andromeda</option>
            <option value="void">Void</option>
          </select>
        </label>
        <a className={`nav-cta ${page === 'workspace' ? 'active' : ''}`} href="#workspace">Run OCR</a>
      </div>
    </nav>
  );
}
