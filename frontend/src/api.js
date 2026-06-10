const BASE = import.meta.env.VITE_API_URL || '/api'

export async function uploadCSV(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Upload failed')
  }
  return res.json()
}

export async function startTranslation({ sessionId, targetLanguages, confirmedColumns }) {
  const res = await fetch(`${BASE}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      target_languages: targetLanguages,
      confirmed_columns: confirmedColumns,
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to start translation')
  }
  return res.json()
}

export async function getStatus(jobId) {
  const res = await fetch(`${BASE}/status/${jobId}`)
  if (!res.ok) throw new Error('Status check failed')
  return res.json()
}

export function downloadUrl(jobId) {
  return `${BASE}/download/${jobId}`
}

export async function getLanguages() {
  const res = await fetch(`${BASE}/languages`)
  if (!res.ok) throw new Error('Failed to load supported languages')
  return res.json()
}

export async function chatAgent({ sessionId, message, phase, targetLanguages }) {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      phase,
      target_languages: targetLanguages || [],
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Chat request failed')
  }
  return res.json()
}
