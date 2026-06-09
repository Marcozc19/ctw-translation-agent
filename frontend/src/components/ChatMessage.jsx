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
        {msg.content}
      </div>

      {!isAgent && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center text-sm shadow-sm">
          👤
        </div>
      )}
    </div>
  )
}
