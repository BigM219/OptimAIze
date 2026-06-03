import { MarkdownPreview } from './MarkdownPreview';

type Props = {
  html?: string;
  markdown?: string;
};

export function HtmlPreview({ html = '', markdown = '' }: Props) {
  if (html.trim()) {
    return <iframe className="html-preview-frame" title="Rendered OCR HTML preview" sandbox="" srcDoc={html} />;
  }

  return <MarkdownPreview markdown={markdown} />;
}
