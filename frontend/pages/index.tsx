import { useState, useRef, useEffect } from 'react'
import Navbar from '@/components/Navbar'

type MediaType = 'image' | 'video' | 'audio'

export default function Home() {
  const [activeTab, setActiveTab] = useState<MediaType>('image')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const tabs: MediaType[] = ['image', 'video', 'audio']

  const acceptMap: Record<MediaType, string> = {
    image: 'image/jpeg,image/png,image/webp',
    video: 'video/mp4,video/avi,video/quicktime',
    audio: 'audio/wav,audio/mpeg,audio/flac'
  }

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return
    }

    const url = URL.createObjectURL(file)
    setPreview(url)

    return () => URL.revokeObjectURL(url)
  }, [file])

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) setFile(dropped)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0]
    if (selected) setFile(selected)
  }

  const removeFile = () => {
    setFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const capitalisedTab = activeTab.charAt(0).toUpperCase() + activeTab.slice(1)

  const renderPreview = () => {
    if (!file || !preview) return null

    switch (activeTab) {
      case 'image':
        return (
          <img
            src={preview}
            alt={file.name}
            className="max-h-40 rounded-lg object-contain border border-white/10"
          />
        )
      case 'video':
        return (
          <video
            src={preview}
            controls
            className="max-h-40 rounded-lg border border-white/10"
          />
        )
      case 'audio':
        return <audio src={preview} controls className="w-full" />
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-[#020617] via-[#0B1120] to-[#0F172A]">
      <title>Fake Media Detection</title>
      <Navbar />

      <div className="max-w-6xl mx-auto px-8 pt-12">
        <p className="text-xs uppercase tracking-[0.25em] text-blue-400 font-semibold">
          AI Detection Platform
        </p>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white mt-3 max-w-3xl">
          Detect AI Generated and Manipulated Media Instantly
        </h1>
        <p className="text-slate-400 mt-4 text-sm md:text-base leading-relaxed max-w-2xl">
          Upload image, video or audio files and analyse them for signs of synthetic
          generation, deepfake manipulation, voice cloning, and digital tampering.
          Get confidence scores and forensic indicators in seconds.
        </p>
      </div>

      {/* main content */}
      <div className="max-w-6xl mx-auto p-8 grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">
        {/* LEFT: upload card */}
        <div className="bg-[#111827]/90 backdrop-blur-sm rounded-2xl border border-white/10 shadow-2xl p-6 flex flex-col gap-5">
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-[0.2em] font-semibold">
              Select media type
            </p>
          </div>

          <div className="flex gap-2 flex-wrap">
            {tabs.map(tab => (
              <button
                key={tab}
                onClick={() => {
                  setActiveTab(tab)
                  removeFile()
                }}
                className={`px-4 py-2 rounded-full text-sm font-medium uppercase tracking-wide transition-all duration-200 border ${
                  activeTab === tab
                    ? 'bg-blue-500/15 border-blue-400/40 text-blue-300 shadow-[0_0_20px_rgba(59,130,246,0.15)]'
                    : 'bg-white/5 border-white/10 text-slate-400 hover:bg-white/10 hover:text-slate-200'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept={acceptMap[activeTab]}
            onChange={handleFileChange}
            className="hidden"
          />

          <div
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => !file && fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl flex flex-col items-center justify-center h-72 transition-all duration-200 ${
              file ? '' : 'cursor-pointer'
            } ${
              dragging
                ? 'border-blue-400 bg-blue-500/10 shadow-[0_0_30px_rgba(59,130,246,0.12)] scale-[1.01]'
                : 'border-blue-400/20 bg-[#020617]/70 hover:border-blue-400/40 hover:bg-[#020617]'
            }`}
          >
            {file ? (
              <div className="flex flex-col items-center gap-3 px-4 text-center">
                {renderPreview()}
                <p className="text-xs text-slate-300 mt-1 break-all">{file.name}</p>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    removeFile()
                  }}
                  className="text-xs text-red-400 hover:text-red-300 transition"
                >
                  ✕ Remove file
                </button>
              </div>
            ) : (
              <>
                <div
                  className={`w-14 h-14 rounded-full border flex items-center justify-center mb-4 transition-all duration-200 ${
                    dragging
                      ? 'bg-blue-500 text-white border-blue-300 shadow-[0_0_20px_rgba(59,130,246,0.35)]'
                      : 'bg-white/5 text-blue-300 border-blue-400/20'
                  }`}
                >
                  <span className="text-xl">↑</span>
                </div>

                <p className="text-slate-300 text-sm font-medium">
                  Drag & drop your {activeTab} here
                </p>
                <p className="text-slate-500 text-xs mt-1">
                  or <span className="text-blue-400 underline">click to browse</span>
                </p>
                <p className="text-slate-500 text-xs mt-3 text-center px-4">
                  {activeTab === 'image' && 'Accepted: JPG, PNG, WEBP · Max 20 MB'}
                  {activeTab === 'video' && 'Accepted: MP4, AVI, MOV · Max 200 MB'}
                  {activeTab === 'audio' && 'Accepted: WAV, MP3, FLAC · Max 50 MB'}
                </p>
              </>
            )}
          </div>

          <button
            disabled={!file}
            className={`w-full py-3 rounded-xl font-semibold tracking-wide transition-all duration-200 ${
              file
                ? 'bg-gradient-to-r from-blue-600 to-cyan-500 text-white hover:scale-[1.01] active:scale-[0.99] shadow-[0_0_25px_rgba(59,130,246,0.25)]'
                : 'bg-white/10 text-slate-500 cursor-not-allowed'
            }`}
          >
            Analyse {capitalisedTab}
          </button>
        </div>

        {/* RIGHT: feature cards */}
        <div className="flex flex-col gap-4">
          <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5 flex gap-4 items-start hover:border-blue-400/20 transition">
            <div className="w-11 h-11 rounded-xl bg-blue-500/15 border border-blue-400/20 flex items-center justify-center shrink-0">
              insert image here
            </div>    
            <div>
              <h3 className="text-sm font-bold text-white">Image Detection</h3>
              <p className="text-sm text-slate-400 mt-1 leading-relaxed">
                Analyses photos for GAN-generated artifacts, inconsistent facial geometry,
                lighting anomalies, and pixel-level statistical irregularities.
              </p>
            </div>
          </div>

          <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5 flex gap-4 items-start hover:border-purple-400/20 transition">
            <div className="w-11 h-11 rounded-xl bg-purple-500/15 border border-purple-400/20 flex items-center justify-center text-purple-300 text-lg shrink-0">
            insert image here
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">Video Detection</h3>
              <p className="text-sm text-slate-400 mt-1 leading-relaxed">
                Frame-by-frame temporal analysis identifies face-swap deepfakes,
                lip-sync inconsistencies, and inter-frame blending artifacts.
              </p>
            </div>
          </div>

          <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5 flex gap-4 items-start hover:border-amber-400/20 transition">
            <div className="w-11 h-11 rounded-xl bg-amber-500/15 border border-amber-400/20 flex items-center justify-center text-amber-300 text-lg shrink-0">
            insert image here
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">Audio Detection</h3>
              <p className="text-sm text-slate-400 mt-1 leading-relaxed">
                Spectral and waveform analysis identifies AI-generated speech,
                voice cloning artifacts, and audio splicing markers.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* stats row */}
      <div className="max-w-6xl mx-auto px-8 pb-10">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5 text-center">
            <p className="text-2xl font-bold text-white">-</p>
            <p className="text-xs text-slate-400 uppercase tracking-[0.2em] mt-2">
              Detection Accuracy
            </p>
          </div>

          <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5 text-center">
            <p className="text-2xl font-bold text-white">-</p>
            <p className="text-xs text-slate-400 uppercase tracking-[0.2em] mt-2">
              Average Analysis
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}