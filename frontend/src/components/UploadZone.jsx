import { useRef, useState } from 'react'

export default function UploadZone({ onFile, disabled }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handle = (file) => {
    if (!file || !file.name.toLowerCase().endsWith('.csv')) {
      alert('Please upload a CSV file.')
      return
    }
    onFile(file)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handle(file)
  }

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`
        relative flex flex-col items-center justify-center gap-2
        rounded-xl border-2 border-dashed px-6 py-8 cursor-pointer
        transition-all duration-150 select-none
        ${disabled
          ? 'opacity-40 cursor-not-allowed border-slate-300 bg-slate-50'
          : dragging
            ? 'border-blue-500 bg-blue-50 scale-[1.01]'
            : 'border-slate-300 bg-white hover:border-blue-400 hover:bg-blue-50/50'
        }
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        className="hidden"
        disabled={disabled}
        onChange={(e) => handle(e.target.files[0])}
      />
      <div className="text-3xl">📄</div>
      <p className="text-sm font-medium text-slate-700">
        Drop your CSV here, or <span className="text-blue-600">click to browse</span>
      </p>
      <p className="text-xs text-slate-400">Chinese-language CSV · up to any size</p>
    </div>
  )
}
