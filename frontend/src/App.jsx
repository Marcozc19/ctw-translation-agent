import { useState, useRef, useEffect, useCallback } from 'react'
import UploadZone from './components/UploadZone.jsx'
import ChatMessage from './components/ChatMessage.jsx'
import { uploadCSV, startTranslation, getStatus, downloadUrl, getLanguages, chatAgent } from './api.js'

// Fallback display names — overwritten by GET /languages on mount, but kept
// so the UI still renders sensible labels if that call fails.
const DEFAULT_LANG_NAMES = {
  en: 'English', es: 'Spanish', fr: 'French', de: 'German',
  ja: 'Japanese', ko: 'Korean', pt: 'Portuguese', vi: 'Vietnamese',
  th: 'Thai', id: 'Indonesian', ar: 'Arabic', hi: 'Hindi',
  ru: 'Russian', it: 'Italian', nl: 'Dutch', ms: 'Malay',
  tr: 'Turkish', pl: 'Polish', sv: 'Swedish', 'zh-TW': 'Traditional Chinese',
}

// ── Phase machine ──────────────────────────────────────────────────────────
// idle → uploaded → translating → done | error
// (the user can also just chat at any point — language selection and
// general conversation are both handled by the backend chat agent)

const WELCOME = {
  role: 'agent',
  id: 'welcome',
  content: 'Hello! I\'m the CTW Translation Agent. Upload a CSV file with Chinese content and tell me which languages to translate it into — or just ask me anything about how this works.',
}

export default function App() {
  const [messages, setMessages] = useState([WELCOME])
  const [phase, setPhase] = useState('idle')
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const [sessionId, setSessionId] = useState(null)
  const [detectedCols, setDetectedCols] = useState([])
  const [rowCount, setRowCount] = useState(0)

  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [targetLangs, setTargetLangs] = useState([])
  const [langNames, setLangNames] = useState(DEFAULT_LANG_NAMES)

  const bottomRef = useRef(null)
  const pollRef = useRef(null)

  const addMessage = useCallback((msg) => {
    setMessages((prev) => [...prev, { ...msg, id: Math.random().toString(36).slice(2) }])
  }, [])

  // Load supported languages from the backend (falls back to defaults on error)
  useEffect(() => {
    getLanguages().then(setLangNames).catch(() => {})
  }, [])

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Polling
  useEffect(() => {
    if (phase !== 'translating' || !jobId) return

    const poll = async () => {
      try {
        const status = await getStatus(jobId)
        setJobStatus(status)

        if (status.status === 'completed') {
          clearInterval(pollRef.current)
          setPhase('done')

          const flaggedNote = status.flagged > 0
            ? ` ${status.flagged} row${status.flagged === 1 ? '' : 's'} flagged for human review.`
            : ' All rows passed quality evaluation.'

          addMessage({
            role: 'agent',
            content: `✅ Translation complete!\n\n• ${status.total} rows translated\n•${flaggedNote}\n• Columns: ${targetLangs.map(l => langNames[l] || l).join(', ')}\n\nYour file is ready to download. Want me to translate to any other languages too? Just ask.`,
          })
        } else if (status.status === 'failed') {
          clearInterval(pollRef.current)
          setPhase('error')
          addMessage({
            role: 'agent',
            content: `❌ Translation failed: ${status.error || 'Unknown error. Please try again.'}`,
          })
        }
      } catch {
        // Ignore transient poll errors
      }
    }

    poll()
    pollRef.current = setInterval(poll, 2000)
    return () => clearInterval(pollRef.current)
  }, [phase, jobId, targetLangs, langNames, addMessage])

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleFile = async (file) => {
    setIsLoading(true)
    addMessage({ role: 'user', content: `📎 ${file.name}` })

    try {
      const data = await uploadCSV(file)
      setSessionId(data.session_id)
      setDetectedCols(data.detected_columns)
      setRowCount(data.row_count)

      if (data.detected_columns.length === 0) {
        addMessage({
          role: 'agent',
          content: `I couldn't find any Chinese text columns in "${file.name}". Please upload a CSV that contains Chinese (Simplified or Traditional) content.`,
        })
        setPhase('idle')
      } else {
        const colList = data.detected_columns.map(c => `**${c}**`).join(', ')
        addMessage({
          role: 'agent',
          content: `Got it! I analyzed **${file.name}**:\n\n• **${data.row_count} rows** detected\n• Chinese text found in: ${colList}\n\nWhich languages would you like to translate to? (e.g. "English and Spanish", "French, German, Japanese")`,
        })
        setPhase('uploaded')
      }
    } catch (err) {
      addMessage({ role: 'agent', content: `❌ Upload error: ${err.message}` })
      setPhase('idle')
    } finally {
      setIsLoading(false)
    }
  }

  // Every message goes through the chat agent, which decides whether the
  // user is selecting target language(s) (intent: 'set_languages' — handles
  // typos, abbreviations, any language) or just chatting/asking a question
  // (intent: 'chat' — answered directly, no translation job started).
  const handleSend = async () => {
    const text = input.trim()
    if (!text || isLoading || phase === 'translating') return
    setInput('')
    addMessage({ role: 'user', content: text })
    setIsLoading(true)

    try {
      const result = await chatAgent({
        sessionId,
        message: text,
        phase,
        targetLanguages: targetLangs,
      })

      const canTranslate = !!sessionId && detectedCols.length > 0 && phase !== 'translating'

      if (result.intent === 'set_languages' && canTranslate) {
        const langs = result.languages
        setTargetLangs(langs)

        addMessage({ role: 'agent', content: result.reply })

        setPhase('translating')
        setJobStatus(null)
        setJobId(null)

        try {
          const { job_id } = await startTranslation({
            sessionId,
            targetLanguages: langs,
            confirmedColumns: detectedCols,
          })
          setJobId(job_id)
        } catch (err) {
          addMessage({ role: 'agent', content: `❌ Failed to start: ${err.message}` })
          setPhase(detectedCols.length > 0 ? 'uploaded' : 'idle')
        }
      } else {
        addMessage({ role: 'agent', content: result.reply })
      }
    } catch (err) {
      addMessage({ role: 'agent', content: `❌ ${err.message}` })
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────

  // Chat is available at any point except mid-translation — users can ask
  // questions, chat, or (re)select languages whenever it makes sense.
  const canType = phase !== 'translating'
  const isTranslating = phase === 'translating'

  const progressPct = jobStatus?.total
    ? Math.round((jobStatus.processed / jobStatus.total) * 100)
    : 0

  const batchLabel = jobStatus?.batch_total
    ? `Batch ${jobStatus.batch_processed} of ${jobStatus.batch_total}`
    : 'Starting...'

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 to-blue-50 flex flex-col items-center justify-center p-4">
      {/* Card */}
      <div className="w-full max-w-2xl bg-white rounded-2xl shadow-xl flex flex-col overflow-hidden" style={{ height: '90vh', maxHeight: '780px' }}>

        {/* Header */}
        <div className="bg-gradient-to-r from-blue-700 to-blue-500 px-5 py-4 flex items-center gap-3">
          <div className="text-2xl">🌐</div>
          <div>
            <h1 className="text-white font-semibold text-base leading-tight">CTW Translation Agent</h1>
            <p className="text-blue-200 text-xs">Multi-language CSV translator · DeepSeek · Gemini · Claude</p>
          </div>
          <div className="ml-auto flex items-center gap-1.5">
            <div className={`w-2 h-2 rounded-full ${isTranslating ? 'bg-yellow-300 animate-pulse' : 'bg-green-400'}`} />
            <span className="text-blue-100 text-xs">{isTranslating ? 'Translating' : 'Ready'}</span>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto scrollbar-hide px-4 py-5 flex flex-col gap-4">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} msg={msg} />
          ))}

          {/* Upload zone — shown when idle */}
          {phase === 'idle' && (
            <div className="px-2">
              <UploadZone onFile={handleFile} disabled={isLoading} />
            </div>
          )}

          {/* Progress — shown while translating */}
          {isTranslating && jobStatus && (
            <div className="mx-2 bg-blue-50 border border-blue-100 rounded-xl p-4">
              <div className="flex justify-between text-xs text-slate-600 mb-2">
                <span className="font-medium">{batchLabel}</span>
                <span>{jobStatus.processed} / {jobStatus.total} rows</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
              {jobStatus.flagged > 0 && (
                <p className="text-xs text-amber-600 mt-2">
                  ⚠ {jobStatus.flagged} row{jobStatus.flagged === 1 ? '' : 's'} flagged for review so far
                </p>
              )}
            </div>
          )}

          {/* Download — shown when done */}
          {phase === 'done' && jobId && (
            <div className="mx-2">
              <a
                href={downloadUrl(jobId)}
                download
                className="flex items-center justify-center gap-2 w-full bg-green-600 hover:bg-green-700 active:scale-95 text-white font-semibold py-3 rounded-xl transition-all duration-150 shadow-md"
              >
                <span>⬇</span>
                <span>Download Translated CSV</span>
              </a>
              <p className="text-center text-xs text-slate-400 mt-2">
                Includes original columns + translated columns + confidence scores
              </p>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input bar */}
        <div className="border-t border-slate-100 px-4 py-3 bg-white">
          <div className="flex gap-2 items-end">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!canType || isLoading}
              rows={1}
              placeholder={
                phase === 'idle' ? 'Ask me anything, or upload a CSV above to get started…'
                : phase === 'uploaded' ? 'e.g. "English and Spanish" — or ask me anything'
                : phase === 'translating' ? 'Translation in progress…'
                : phase === 'done' ? 'Ask for more languages, or download your file above'
                : 'Type a message…'
              }
              className="flex-1 resize-none rounded-xl border border-slate-200 bg-slate-50 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed leading-relaxed"
              style={{ maxHeight: '120px' }}
            />
            <button
              onClick={handleSend}
              disabled={!canType || !input.trim() || isLoading}
              className="flex-shrink-0 w-10 h-10 rounded-xl bg-blue-600 hover:bg-blue-700 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed text-white flex items-center justify-center transition-all duration-150 shadow-sm"
            >
              {isLoading ? (
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                </svg>
              ) : (
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 19V5m0 0l-7 7m7-7l7 7"/>
                </svg>
              )}
            </button>
          </div>
          {phase === 'done' && (
            <button
              onClick={() => {
                setMessages([WELCOME])
                setPhase('idle')
                setSessionId(null)
                setJobId(null)
                setJobStatus(null)
                setTargetLangs([])
                setDetectedCols([])
              }}
              className="w-full mt-2 text-xs text-blue-600 hover:text-blue-800 transition-colors"
            >
              ↺ Start over with a new file
            </button>
          )}
        </div>
      </div>

      {/* Footer */}
      <p className="mt-3 text-xs text-slate-400">
        CTW AI Product Manager Assessment · Marco · June 2026
      </p>
    </div>
  )
}
