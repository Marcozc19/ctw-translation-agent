// Lightweight markdown rendering for chat bubbles — supports **bold** and
// line breaks, which is all the chat/translation agents currently emit.
// Avoids pulling in a full markdown dependency for such a small need.
function renderInline(text, keyPrefix) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return <strong key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</strong>
    }
    return <span key={`${keyPrefix}-${i}`}>{part}</span>
  })
}

function renderContent(content) {
  return String(content).split('\n').map((line, i) => (
    <div key={i}>{line ? renderInline(line, i) : ' '}</div>
  ))
}

export default function ChatMessage({ msg }) {
  const isAgent = msg.role === 'agent'

  return (
    <div className={`flex gap-3 ${isAgent ? 'justify-start' : 'justify-end'}`}>
      {isAgent && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white text-sm font-bold shadow-sm">
          🌐
        </div>
      )}

      <div
        className={`
          max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm
          ${isAgent
            ? 'bg-white text-slate-800 rounded-tl-sm border border-slate-100'
            : 'bg-blue-600 text-white rounded-tr-sm'
          }
        `}
      >
        {renderContent(msg.content)}
      </div>

      {!isAgent && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center text-sm shadow-sm">
          👤
        </div>
      )}
    </div>
  )
}
