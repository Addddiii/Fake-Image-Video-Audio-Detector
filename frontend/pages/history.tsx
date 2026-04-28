import Head from 'next/head'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { onAuthStateChanged } from 'firebase/auth'
import { useEffect, useMemo, useState } from 'react'
import Navbar from '@/components/Navbar'
import { auth } from '@/utils/firebase'
import {
  HistoryEntry,
  clearHistoryEntries,
  getHistoryEntries,
} from '@/utils/historyStorage'

type FilterType = 'all' | 'image' | 'video' | 'audio'

function formatDate(dateString: string) {
  const date = new Date(dateString)
  return date.toLocaleString([], {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

function getMediaLabel(type: string) {
  return type.charAt(0).toUpperCase() + type.slice(1)
}

function getVerdictLabel(verdict: 'fake' | 'real') {
  return verdict === 'real' ? 'Authentic' : 'Fake'
}

function FileTypeBubble({ type }: { type: 'image' | 'video' | 'audio' }) {
  const styles = {
    image: 'bg-blue-500/15 border-blue-400/20 text-blue-300',
    video: 'bg-purple-500/15 border-purple-400/20 text-purple-300',
    audio: 'bg-amber-500/15 border-amber-400/20 text-amber-300',
  }

  const labels = {
    image: 'IMG',
    video: 'VID',
    audio: 'AUD',
  }

  return (
    <div className={`w-11 h-11 rounded-xl border flex items-center justify-center text-[11px] font-bold shrink-0 ${styles[type]}`}>
      {labels[type]}
    </div>
  )
}

export default function HistoryPage() {
  const router = useRouter()
  const [loading, setLoading] = useState(true)
  const [userKey, setUserKey] = useState('')
  const [userName, setUserName] = useState('')
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<FilterType>('all')

  useEffect(() => {
    if (!auth) {
      router.replace('/login')
      return
    }

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (!user) {
        router.replace('/login')
        return
      }

      const key = user.uid || user.email || ''
      setUserKey(key)
      setUserName(user.displayName || user.email?.split('@')[0] || 'User')
      setEntries(getHistoryEntries(key))
      setLoading(false)
    })

    return unsubscribe
  }, [router])

  const filteredEntries = useMemo(() => {
    return entries.filter((entry) => {
      const matchesFilter = filter === 'all' ? true : entry.mediaType === filter
      const matchesSearch = entry.fileName.toLowerCase().includes(search.toLowerCase())
      return matchesFilter && matchesSearch
    })
  }, [entries, filter, search])

  const totalScans = entries.length
  const fakeDetected = entries.filter((entry) => entry.verdict === 'fake').length
  const avgConfidence =
    entries.length > 0
      ? entries.reduce((sum, entry) => sum + entry.confidence, 0) / entries.length
      : 0
  const avgProcessing =
    entries.length > 0
      ? entries.reduce((sum, entry) => sum + entry.analysisTime, 0) / entries.length
      : 0

  const mediaCounts = {
    image: entries.filter((entry) => entry.mediaType === 'image').length,
    video: entries.filter((entry) => entry.mediaType === 'video').length,
    audio: entries.filter((entry) => entry.mediaType === 'audio').length,
  }

  const verdictCounts = {
    real: entries.filter((entry) => entry.verdict === 'real').length,
    fake: entries.filter((entry) => entry.verdict === 'fake').length,
  }

  const maxMediaCount = Math.max(mediaCounts.image, mediaCounts.video, mediaCounts.audio, 1)
  const verdictTotal = Math.max(entries.length, 1)

  const handleClearHistory = () => {
    if (!userKey) return
    clearHistoryEntries(userKey)
    setEntries([])
  }

  if (loading) {
    return (
      <main className="min-h-screen bg-[#020617] text-white">
        <Navbar />
        <div className="max-w-6xl mx-auto px-8 pt-16">
          <p className="text-slate-400">Loading history...</p>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-[#020617] relative overflow-hidden">
      <Head>
        <title>History | LatFakeCheck</title>
      </Head>

      <div
        className="pointer-events-none absolute inset-0 z-0"
        style={{
          backgroundSize: '100% 200%',
          backgroundImage:
            'radial-gradient(ellipse 80% 50% at 50% 0%, rgba(12,185,235,0.22) 0%, transparent 65%)',
        }}
      />

      <div
        className="pointer-events-none absolute inset-0 z-0"
        style={{
          backgroundImage: 'radial-gradient(rgba(12,185,235,0.15) 1px, transparent 1px)',
          backgroundSize: '28px 28px',
        }}
      />

      <Navbar />

      <div className="relative z-10 max-w-6xl mx-auto px-8 pt-10 pb-16">
        <p className="text-xs uppercase tracking-[0.25em] text-[#0cb9eb]/80 font-semibold mb-2">
          User Activity
        </p>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white">
          Scan History
        </h1>
        <p className="text-slate-400 mt-4 text-sm md:text-base leading-relaxed max-w-2xl">
          Review your previous image, video, and audio analyses. This history is stored per signed-in user on this device.
        </p>

        <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5">
            <p className="text-3xl font-bold text-white">{totalScans}</p>
            <p className="text-xs text-slate-400 mt-2 uppercase tracking-[0.2em]">Total Scans</p>
          </div>

          <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5">
            <p className="text-3xl font-bold text-red-400">{fakeDetected}</p>
            <p className="text-xs text-slate-400 mt-2 uppercase tracking-[0.2em]">Fakes Detected</p>
          </div>

          <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5">
            <p className="text-3xl font-bold text-[#0cb9eb]">
              {entries.length ? `${avgConfidence.toFixed(1)}%` : '-'}
            </p>
            <p className="text-xs text-slate-400 mt-2 uppercase tracking-[0.2em]">Avg Confidence</p>
          </div>

          <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-5">
            <p className="text-3xl font-bold text-[#0cb9eb]">
              {entries.length ? `${avgProcessing.toFixed(2)}s` : '-'}
            </p>
            <p className="text-xs text-slate-400 mt-2 uppercase tracking-[0.2em]">Avg Processing</p>
          </div>
        </div>

        <div className="mt-8 grid grid-cols-1 lg:grid-cols-[1.7fr_1fr] gap-6">
          <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-6">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-[#0cb9eb]/80 font-semibold">
                  Recent Activity
                </p>
                <h2 className="text-xl font-semibold text-white mt-2">
                  {userName}&apos;s analysis history
                </h2>
              </div>

              <button
                onClick={handleClearHistory}
                className="px-4 py-2 rounded-xl border border-red-400/20 bg-red-500/10 text-sm text-red-300 hover:bg-red-500/15 transition"
              >
                Clear History
              </button>
            </div>

            <div className="flex flex-col md:flex-row gap-3 mb-6">
              <input
                type="text"
                placeholder="Search by file name..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="flex-1 rounded-xl border border-white/10 bg-[#020617]/70 px-4 py-3 text-white outline-none focus:border-[#0cb9eb]/50"
              />

              <div className="flex gap-2 flex-wrap">
                {(['all', 'image', 'video', 'audio'] as FilterType[]).map((item) => (
                  <button
                    key={item}
                    onClick={() => setFilter(item)}
                    className={`px-4 py-2 rounded-full text-sm font-medium uppercase tracking-wide transition-all duration-200 border ${
                      filter === item
                        ? 'bg-[#0cb9eb]/20 border-[#0cb9eb]/70 text-[#0cb9eb] shadow-[0_0_18px_rgba(12,185,235,0.2)]'
                        : 'bg-white/5 border-white/10 text-slate-400 hover:bg-white/10 hover:text-slate-200'
                    }`}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>

            {filteredEntries.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-white/10 bg-[#020617]/50 px-6 py-12 text-center">
                <p className="text-white font-medium text-lg">No history yet</p>
                <p className="text-slate-400 text-sm mt-2">
                  Run a scan from the home page and your results will appear here.
                </p>
                <Link
                  href="/"
                  className="inline-flex mt-5 px-5 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 text-white font-semibold shadow-[0_0_25px_rgba(59,130,246,0.2)]"
                >
                  Go to Scanner
                </Link>
              </div>
            ) : (
              <div className="space-y-4">
                {filteredEntries.map((entry) => (
                  <div
                    key={entry.id}
                    className="rounded-2xl border border-white/10 bg-[#0B1220]/85 p-4 hover:border-[#0cb9eb]/30 transition"
                  >
                    <div className="flex flex-col md:flex-row md:items-center gap-4">
                      <div className="flex items-center gap-4 min-w-0 flex-1">
                        <FileTypeBubble type={entry.mediaType} />

                        <div className="min-w-0">
                          <p className="text-white font-medium truncate">{entry.fileName}</p>
                          <div className="flex flex-wrap items-center gap-2 mt-2 text-xs">
                            <span className="px-2.5 py-1 rounded-full border border-white/10 bg-white/5 text-slate-300">
                              {getMediaLabel(entry.mediaType)}
                            </span>

                            <span
                              className={`px-2.5 py-1 rounded-full border ${
                                entry.verdict === 'real'
                                  ? 'border-green-400/20 bg-green-500/10 text-green-300'
                                  : 'border-red-400/20 bg-red-500/10 text-red-300'
                              }`}
                            >
                              {getVerdictLabel(entry.verdict)}
                            </span>

                            <span className="text-slate-500">
                              {formatDate(entry.analyzedAt)}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:flex md:items-center gap-4 md:gap-6 text-sm">
                        <div>
                          <p className="text-slate-500 text-xs uppercase tracking-wide">Confidence</p>
                          <p className="text-white font-semibold">{entry.confidence.toFixed(1)}%</p>
                        </div>

                        <div>
                          <p className="text-slate-500 text-xs uppercase tracking-wide">Time</p>
                          <p className="text-white font-semibold">{entry.analysisTime.toFixed(2)}s</p>
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-400">Authentic</span>
                          <span className="text-green-400 font-semibold">{entry.realProbability.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-green-600 to-green-400 h-2.5 rounded-full"
                            style={{ width: `${entry.realProbability}%` }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-400">Fake</span>
                          <span className="text-red-400 font-semibold">{entry.fakeProbability.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-red-600 to-red-400 h-2.5 rounded-full"
                            style={{ width: `${entry.fakeProbability}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="space-y-6">
            <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-6">
              <p className="text-xs uppercase tracking-[0.2em] text-[#0cb9eb]/80 font-semibold mb-4">
                Scans by Media Type
              </p>

              <div className="space-y-4">
                {(['image', 'video', 'audio'] as const).map((type) => {
                  const count = mediaCounts[type]
                  const width = `${(count / maxMediaCount) * 100}%`

                  return (
                    <div key={type}>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-slate-300">{getMediaLabel(type)}</span>
                        <span className="text-white font-semibold">{count}</span>
                      </div>
                      <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-2.5 rounded-full ${
                            type === 'image'
                              ? 'bg-blue-500'
                              : type === 'video'
                              ? 'bg-purple-500'
                              : 'bg-amber-400'
                          }`}
                          style={{ width }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-6">
              <p className="text-xs uppercase tracking-[0.2em] text-[#0cb9eb]/80 font-semibold mb-4">
                Verdict Breakdown
              </p>

              <div className="w-full h-4 rounded-full bg-slate-800 overflow-hidden flex mb-4">
                <div
                  className="bg-green-500"
                  style={{ width: `${(verdictCounts.real / verdictTotal) * 100}%` }}
                />
                <div
                  className="bg-red-500"
                  style={{ width: `${(verdictCounts.fake / verdictTotal) * 100}%` }}
                />
              </div>

              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-slate-300">
                    <span className="w-3 h-3 rounded-full bg-green-500" />
                    Authentic
                  </div>
                  <span className="text-white font-semibold">{verdictCounts.real}</span>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-slate-300">
                    <span className="w-3 h-3 rounded-full bg-red-500" />
                    Fake
                  </div>
                  <span className="text-white font-semibold">{verdictCounts.fake}</span>
                </div>
              </div>
            </div>

            <div className="bg-[#111827]/90 rounded-2xl border border-white/10 shadow-xl p-6">
              <p className="text-xs uppercase tracking-[0.2em] text-[#0cb9eb]/80 font-semibold mb-4">
                Quick Insights
              </p>

              <div className="space-y-3 text-sm">
                <div className="rounded-xl bg-white/5 border border-white/10 px-4 py-3">
                  <p className="text-slate-400 text-xs">Most used media type</p>
                  <p className="text-white font-semibold mt-1">
                    {totalScans === 0
                      ? 'No data yet'
                      : Object.entries(mediaCounts).sort((a, b) => b[1] - a[1])[0][0].toUpperCase()}
                  </p>
                </div>

                <div className="rounded-xl bg-white/5 border border-white/10 px-4 py-3">
                  <p className="text-slate-400 text-xs">Latest scan</p>
                  <p className="text-white font-semibold mt-1">
                    {entries[0] ? formatDate(entries[0].analyzedAt) : 'No scans yet'}
                  </p>
                </div>

                <div className="rounded-xl bg-white/5 border border-white/10 px-4 py-3">
                  <p className="text-slate-400 text-xs">Detected fake rate</p>
                  <p className="text-white font-semibold mt-1">
                    {totalScans ? `${((fakeDetected / totalScans) * 100).toFixed(1)}%` : '0%'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}