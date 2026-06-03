type Props = {
  markdown: string;
};

export function MarkdownPreview({ markdown }: Props) {
  if (!markdown.trim()) {
    return <p className="muted">Rendered markdown preview will appear here.</p>;
  }

  return markdown.split(/\n{2,}/).map((block, index) => {
    const trimmed = block.trim();
    if (!trimmed) return null;

    if (trimmed.startsWith('#')) {
      const text = trimmed.replace(/^#+\s*/, '');
      return <h4 key={`${index}-${text}`}>{text}</h4>;
    }

    if (/^[-*]\s/m.test(trimmed)) {
      return (
        <ul key={`${index}-${trimmed.slice(0, 12)}`}>
          {trimmed.split('\n').map((line) => (
            <li key={line}>{line.replace(/^[-*]\s*/, '')}</li>
          ))}
        </ul>
      );
    }

    return <p key={`${index}-${trimmed.slice(0, 12)}`}>{trimmed}</p>;
  });
}
