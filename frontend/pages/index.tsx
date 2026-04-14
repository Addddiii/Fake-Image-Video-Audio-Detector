import { useState, useRef, useEffect } from 'react'
import Navbar from '@/components/Navbar'
import { auth, hasFirebaseConfig } from '@/utils/firebase'
import { onAuthStateChanged } from 'firebase/auth'

type MediaType = 'image' | 'video' | 'audio'

export default function Home() {
  const [activeTab, setActiveTab] = useState<MediaType>('image')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [userName, setUserName] = useState<string | null>(null)
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [loginPrompt, setLoginPrompt] = useState('')

  const tabs: MediaType[] = ['image', 'video', 'audio']

  const acceptMap: Record<MediaType, string> = {
    image: 'image/jpeg,image/png,image/webp',
    video: 'video/mp4,video/avi,video/quicktime',
    audio: 'audio/wav,audio/mpeg,audio/flac'
  }

  const ImageIcon = () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      className="w-5 h-5"
    >
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <circle cx="9" cy="10" r="1.5" />
      <path d="M21 16l-5-5-4 4-2-2-4 4" />
    </svg>
  )

  const VideoIcon = () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      className="w-5 h-5"
    >
      <rect x="3" y="6" width="13" height="12" rx="2" />
      <path d="M16 10l5-3v10l-5-3z" />
    </svg>
  )

  const AudioIcon = () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      className="w-5 h-5"
    >
      <path d="M11 5L6 9H3v6h3l5 4V5z" />
      <path d="M15.5 8.5a5 5 0 010 7" />
      <path d="M18 6a8 8 0 010 12" />
    </svg>
  )

  useEffect(() => {
    if (!auth) return

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setIsLoggedIn(Boolean(user))
      setUserName(user?.displayName || user?.email?.split('@')[0] || null)

      if (user) {
        setLoginPrompt('')
      } else {
        setFile(null)
        setPreview(null)
        setDragging(false)
      }
    })

    return unsubscribe
  }, [])

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return
    }

    const url = URL.createObjectURL(file)
    setPreview(url)

    return () => URL.revokeObjectURL(url)
  }, [file])

  const showLoginPrompt = () => {
    setLoginPrompt('Please log in first to upload and analyse files.')
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()

    if (!isLoggedIn) {
      setDragging(false)
      showLoginPrompt()
      return
    }

    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) setFile(dropped)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!isLoggedIn) {
      showLoginPrompt()
      return
    }

    const selected = e.target.files?.[0]
    if (selected) setFile(selected)
  }

  const removeFile = () => {
    setFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleUploadAreaClick = () => {
    if (!isLoggedIn) {
      showLoginPrompt()
      return
    }

    if (!file) {
      fileInputRef.current?.click()
    }
  }

  const handleUpload = async () => {
    if (!file) return

    if (!hasFirebaseConfig || !auth) {
      alert('Firebase is not configured yet.')
      return
    }

    const user = auth.currentUser

    if (!user) {
      showLoginPrompt()
      return
    }

    const token = await user.getIdToken()
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://127.0.0.1:8000/upload', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      })

      const data = await response.json()
      console.log(data)
      alert('Upload successful')
    } catch (error) {
      console.error(error)
      alert('Upload failed')
    }
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

      <div className="max-w-6xl mx-auto px-8 pt-12 grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-slate-500 font-semibold">
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

        {userName ? (
          <div className="bg-blue-500/10 border border-blue-400/20 rounded-2xl px-6 py-4 flex items-center gap-4 mt-20">
            <div className="w-10 h-10 rounded-full bg-blue-500/20 border border-blue-400/30 flex items-center justify-center text-blue-300 font-bold text-lg shrink-0">
              {userName.charAt(0).toUpperCase()}
            </div>
            <div>
              <p className="text-white font-semibold text-base">Welcome back, {userName}</p>
              <p className="text-slate-400 text-xs mt-0.5">Ready to analyse some media?</p>
            </div>
          </div>
        ) : (
          <div />
        )}
      </div>

      <div className="max-w-6xl mx-auto p-8 grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">
        <div className="bg-[#111827]/90 backdrop-blur-sm rounded-2xl border border-white/10 shadow-2xl p-6 flex flex-col gap-5">
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-[0.2em] font-semibold">
              Select media type
            </p>
          </div>

          <div className="flex gap-2 flex-wrap">
            {tabs.map((tab) => (
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

          {!isLoggedIn && (
            <div className="rounded-xl border border-amber-400/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
              Please log in first to upload and analyse files.
            </div>
          )}

          {loginPrompt && !isLoggedIn && (
            <p className="text-sm text-red-400">{loginPrompt}</p>
          )}

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

              if (!isLoggedIn) {
                setDragging(false)
                showLoginPrompt()
                return
              }

              setDragging(true)
            }}
            onDragLeave={() => {
              if (isLoggedIn) setDragging(false)
            }}
            onDrop={handleDrop}
            onClick={handleUploadAreaClick}
            className={`border-2 border-dashed rounded-2xl flex flex-col items-center justify-center h-72 transition-all duration-200 ${
              isLoggedIn && !file ? 'cursor-pointer' : ''
            } ${
              !isLoggedIn
                ? 'border-white/10 bg-[#020617]/40 opacity-80 cursor-not-allowed'
                : dragging
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
            ) : !isLoggedIn ? (
              <>
                <div className="w-14 h-14 rounded-full border flex items-center justify-center mb-4 bg-white/5 text-slate-300 border-white/10">
                  <span className="text-xl">🔒</span>
                </div>

                <p className="text-slate-200 text-sm font-medium text-center px-4">
                  Log in to upload image, video, or audio files
                </p>
                <p className="text-slate-500 text-xs mt-2 text-center px-4">
                  Upload and analysis are only available for signed-in users.
                </p>
              </>
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
            onClick={handleUpload}
            disabled={!file || !isLoggedIn}
            className={`w-full py-3 rounded-xl font-semibold tracking-wide transition-all duration-200 ${
              file && isLoggedIn
                ? 'bg-gradient-to-r from-blue-600 to-cyan-500 text-white hover:scale-[1.01] active:scale-[0.99] shadow-[0_0_25px_rgba(59,130,246,0.25)]'
                : 'bg-white/10 text-slate-500 cursor-not-allowed'
            }`}
          >
            {isLoggedIn ? `Analyse ${capitalisedTab}` : 'Log in to analyse'}
          </button>
        </div>

        <div className="flex flex-col gap-4">
          <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5 flex gap-4 items-start hover:border-blue-400/20 transition">
            <div className="w-11 h-11 rounded-xl bg-blue-500/15 border border-blue-400/20 flex items-center justify-center text-blue-300 shrink-0">
              <ImageIcon />
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
            <div className="w-11 h-11 rounded-xl bg-purple-500/15 border border-purple-400/20 flex items-center justify-center text-purple-300 shrink-0">
              <VideoIcon />
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
            <div className="w-11 h-11 rounded-xl bg-amber-500/15 border border-amber-400/20 flex items-center justify-center text-amber-300 shrink-0">
              <AudioIcon />
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