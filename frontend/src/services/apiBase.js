export function getApiBase() {
  const raw = process.env.VUE_APP_API_BASE;
  if (typeof raw !== 'string') return '';
  const trimmed = raw.trim();
  if (!trimmed) return '';
  return trimmed.replace(/\/+$/, '');
}

