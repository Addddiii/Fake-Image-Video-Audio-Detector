import { useState, useRef, useEffect } from 'react'
import Navbar from '@/components/Navbar'
import { auth, hasFirebaseConfig } from '@/utils/firebase'
import { onAuthStateChanged } from 'firebase/auth'
import Head from 'next/head'

type MediaType = 'image' | 'video' | 'audio'

interface PredictionResult {
  prediction: 'fake' | 'real'
  confidence: number
  probabilities: {
    fake: number
    real: number
  }
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<MediaType>('image')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [userName, setUserName] = useState<string | null>(null)
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [loginPrompt, setLoginPrompt] = useState('')
  const [pageReady, setPageReady] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [analysisTime, setAnalysisTime] = useState<number>(0)
  const [totalAnalyses, setTotalAnalyses] = useState<number>(0)

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
    const timer = window.setTimeout(() => setPageReady(true), 60)
    return () => window.clearTimeout(timer)
  }, [])

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
    setLoginPrompt('Log in or sign up first to start uploading and analysing files.')
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
    setPredictionResult(null)
    setError(null)
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

    setIsAnalyzing(true)
    setPredictionResult(null)
    setError(null)

    const startTime = performance.now()

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
      const endTime = performance.now()
      const analysisTimeSeconds = ((endTime - startTime) / 1000).toFixed(2)

      console.log(data)

      if (data.prediction) {
        setPredictionResult(data.prediction)
        setAnalysisTime(parseFloat(analysisTimeSeconds))
        setTotalAnalyses(prev => prev + 1)
      } else if (data.error) {
        setError(data.error)
      } else {
        setError('Unable to analyze image. Please try again.')
      }
    } catch (error) {
      console.error(error)
      setError('Upload failed. Please check if the backend server is running.')
    } finally {
      setIsAnalyzing(false)
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
    <main className="min-h-screen bg-[#020617] relative overflow-hidden">
      <Head>
        <title>LatFakeCheck</title>
        <link rel="icon" href="/assets/LFC-logo-trans.png" />
      </Head>

      <div
        className="pointer-events-none absolute inset-0 z-0"
        style={{
          backgroundSize: '100% 200%',
          backgroundImage: 'radial-gradient(ellipse 80% 50% at 50% 0%, rgba(12,185,235,0.28) 0%, transparent 65%)',
          animation: 'drift 6s ease-in-out infinite'
        }}
      />

      <div
        className="pointer-events-none absolute inset-0 z-0"
        style={{
          backgroundImage: 'radial-gradient(rgba(12,185,235,0.15) 1px, transparent 1px)',
          backgroundSize: '28px 28px'
        }}
      />

      <Navbar />

      <div
        className={`transition-all duration-700 ease-out ${pageReady ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'
          }`}
      >
        <div
          className={`max-w-6xl mx-auto px-8 ${isLoggedIn ? 'pt-12' : 'pt-8'
            } ${isLoggedIn ? 'grid grid-cols-1 lg:grid-cols-2 gap-10 items-start' : ''}`}
        >
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-[#0cb9eb]/80 font-semibold">
              AI Detection Platform
            </p>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white mt-3 max-w-3xl">
              Detect AI Generated and Manipulated Media Instantly
            </h1>
            <p className="text-slate-400 mt-6 text-sm md:text-base leading-relaxed max-w-2xl">
              Upload image, video or audio files and analyse them for signs of synthetic
              generation, deepfake manipulation, voice cloning, and digital tampering.
              Get confidence scores and forensic indicators in seconds.
            </p>
          </div>

          {isLoggedIn && userName ? (
            <div className="bg-blue-500/10 border border-blue-400/20 rounded-2xl px-6 py-4 flex items-center gap-4 mt-20 shadow-[0_0_24px_rgba(59,130,246,0.08)]">
              <div className="w-10 h-10 rounded-full bg-[#0cb9eb]/20 border border-[#0cb9eb]/40 flex items-center justify-center text-[#0cb9eb] font-bold text-lg shrink-0">
                {userName.charAt(0).toUpperCase()}
              </div>
              <div>
                <p className="text-white font-semibold text-base">Welcome back, {userName}</p>
                <p className="text-slate-400 text-xs mt-0.5">Ready to analyse some media?</p>
              </div>
            </div>
          ) : null}
        </div>

        <div className={`max-w-6xl mx-auto px-8 ${isLoggedIn ? 'pt-8 pb-10' : 'pt-6 pb-10'}`}>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">
            <div className="bg-[#111827]/90 backdrop-blur-sm rounded-2xl border border-white/10 shadow-2xl p-6 flex flex-col gap-5">
              <div>
                <p className="text-xs text-[#0cb9eb]/80 uppercase tracking-[0.2em] font-semibold">
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
                    className={`px-4 py-2 rounded-full text-sm font-semibold uppercase tracking-wide transition-all duration-200 border ${activeTab === tab
                        ? 'bg-[#0cb9eb]/20 border-[#0cb9eb]/70 text-[#0cb9eb] shadow-[0_0_24px_rgba(12,185,235,0.3)]'
                        : 'bg-white/5 border-white/10 text-slate-400 hover:bg-white/10 hover:text-slate-200'
                      }`}
                  >
                    {tab}
                  </button>
                ))}
              </div>

              {!isLoggedIn && (
                <div className="rounded-xl border border-blue-400/20 bg-blue-500/10 px-4 py-3 flex items-start gap-3 shadow-[0_0_18px_rgba(59,130,246,0.06)]">
                  <div className="w-8 h-8 rounded-full bg-blue-500/15 border border-blue-400/20 flex items-center justify-center text-blue-300 shrink-0 mt-0.5">
                    <span className="text-sm font-bold">i</span>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-blue-200">
                      Please log in first to upload and analyse files.
                    </p>
                    <p className="text-xs text-slate-400 mt-1">
                      Use the Login / Signup button above to get started.
                    </p>
                  </div>
                </div>
              )}

              {loginPrompt && !isLoggedIn && (
                <p className="text-sm text-blue-300">{loginPrompt}</p>
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
                className={`border-2 border-dashed rounded-2xl flex flex-col items-center justify-center h-72 transition-all duration-200 ${isLoggedIn && !file ? 'cursor-pointer' : ''
                  } ${!isLoggedIn
                    ? 'border-white/10 bg-[#020617]/40 opacity-80 cursor-not-allowed'
                    : dragging
                      ? 'border-blue-400 bg-blue-500/10 shadow-[0_0_30px_rgba(59,130,246,0.12)] scale-[1.01]'
                      : 'border-[#0cb9eb]/50 bg-[#020617]/70 hover:border-[#0cb9eb]/70 hover:bg-[#03102a] hover:shadow-[0_0_26px_rgba(12,185,235,0.12)]'
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
                    <div className="w-14 h-14 rounded-full border flex items-center justify-center mb-4 bg-blue-500/10 text-blue-300 border-blue-400/20 shadow-[0_0_18px_rgba(59,130,246,0.08)]">
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
                      className={`w-14 h-14 rounded-full border flex items-center justify-center mb-4 transition-all duration-200 ${dragging
                          ? 'bg-blue-500 text-white border-blue-300 shadow-[0_0_20px_rgba(59,130,246,0.35)]'
                          : 'bg-white/5 text-blue-300 border-blue-400/20'
                        }`}
                    >
                      <span className="text-xl">↑</span>
                    </div>

                    <p className="text-slate-200 text-sm font-medium">
                      Drag & drop your {activeTab} here
                    </p>
                    <p className="text-slate-500 text-xs mt-1">
                      or <span className="text-blue-300 underline">click to browse</span>
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
                disabled={!file || !isLoggedIn || isAnalyzing}
                className={`w-full py-3 rounded-xl font-bold tracking-[0.02em] transition-all duration-200 ${file && isLoggedIn && !isAnalyzing
                    ? 'bg-gradient-to-r from-blue-600 to-cyan-500 text-white hover:scale-[1.01] active:scale-[0.99] shadow-[0_0_25px_rgba(59,130,246,0.25)]'
                    : 'bg-white/10 text-slate-400 cursor-not-allowed'
                  }`}
              >
                {isAnalyzing ? 'Analyzing...' : isLoggedIn ? `Analyse ${capitalisedTab}` : 'Log in to analyse'}
              </button>

              {/* Prediction Results Display */}
              {predictionResult && (
                <div className="bg-[#111827]/90 backdrop-blur-sm rounded-2xl border border-white/10 shadow-2xl p-6 animate-fadeIn">
                  <h3 className="text-sm font-bold text-[#0cb9eb] uppercase tracking-[0.2em] mb-4">
                    Analysis Result
                  </h3>

                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center text-2xl ${predictionResult.prediction === 'real'
                          ? 'bg-green-500/20 border-2 border-green-400/50'
                          : 'bg-red-500/20 border-2 border-red-400/50'
                        }`}>
                        {predictionResult.prediction === 'real' ? '✓' : '⚠'}
                      </div>
                      <div>
                        <p className={`text-2xl font-bold ${predictionResult.prediction === 'real' ? 'text-green-400' : 'text-red-400'
                          }`}>
                          {predictionResult.prediction === 'real' ? 'Real Image' : 'Fake/AI Generated'}
                        </p>
                        <p className="text-sm text-slate-400">
                          {predictionResult.confidence}% confidence
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Real Probability</span>
                        <span className="text-green-400 font-semibold">{predictionResult.probabilities.real}%</span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-green-600 to-green-400 h-2.5 rounded-full transition-all duration-500"
                          style={{ width: `${predictionResult.probabilities.real}%` }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Fake Probability</span>
                        <span className="text-red-400 font-semibold">{predictionResult.probabilities.fake}%</span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-red-600 to-red-400 h-2.5 rounded-full transition-all duration-500"
                          style={{ width: `${predictionResult.probabilities.fake}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Error Display */}
              {error && (
                <div className="bg-red-500/10 border border-red-400/30 rounded-xl px-4 py-3 flex items-start gap-3">
                  <span className="text-red-400 text-lg">⚠</span>
                  <div>
                    <p className="text-sm font-medium text-red-300">Error</p>
                    <p className="text-xs text-slate-400 mt-1">{error}</p>
                  </div>
                </div>
              )}
            </div>

            <div className="flex flex-col gap-4">
              <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5 flex gap-4 items-start transition-all duration-200 hover:border-blue-400/30 hover:-translate-y-1 hover:shadow-[0_16px_32px_rgba(0,0,0,0.22)]">
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

              <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5 flex gap-4 items-start transition-all duration-200 hover:border-purple-400/30 hover:-translate-y-1 hover:shadow-[0_16px_32px_rgba(0,0,0,0.22)]">
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

              <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5 flex gap-4 items-start transition-all duration-200 hover:border-amber-400/30 hover:-translate-y-1 hover:shadow-[0_16px_32px_rgba(0,0,0,0.22)]">
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

          <div className="max-w-6xl mx-auto pt-4 pb-10">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5 text-center">
                <p className="text-2xl font-bold text-[#0cb9eb]">
                  {predictionResult ? `${predictionResult.confidence.toFixed(1)}%` : '-'}
                </p>
                <p className="text-xs text-slate-400 uppercase tracking-[0.2em] mt-2">
                  Detection Confidence
                </p>
              </div>

              <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5 text-center">
                <p className="text-2xl font-bold text-[#0cb9eb]">
                  {analysisTime > 0 ? `${analysisTime}s` : '-'}
                </p>
                <p className="text-xs text-slate-400 uppercase tracking-[0.2em] mt-2">
                  Analysis Time
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}