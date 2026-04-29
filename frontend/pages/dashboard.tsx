'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/router'
import Navbar from '@/components/Navbar'
import { auth } from '@/utils/firebase'
import { onAuthStateChanged } from 'firebase/auth'
import { HistoryEntry, getHistoryEntries } from '@/utils/historyStorage'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts'

const BAR_COLORS = ['#3b82f6', '#a78bfa', '#f59e0b']
const PIE_COLORS = ['#3b82f6', '#a78bfa', '#f59e0b']

const BarTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#1e293b] border border-white/10 rounded-xl px-4 py-2 text-sm text-white shadow-xl">
        <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">{label}</p>
        <p className="font-bold">{payload[0].value} scans</p>
      </div>
    )
  }
  return null
}

const PieTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#1e293b] border border-white/10 rounded-xl px-4 py-2 text-sm text-white shadow-xl">
        <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">{payload[0].name}</p>
        <p className="font-bold">{payload[0].value}%</p>
      </div>
    )
  }
  return null
}

export default function Dashboard() {
  const router = useRouter()
  const [entries, setEntries] = useState<HistoryEntry[]>([])

  useEffect(() => {
    if (!auth) return

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (!user) {
        router.replace('/login')
        return
      }
      const userKey = user.uid || user.email || ''
      setEntries(getHistoryEntries(userKey))
    })

    return unsubscribe
  }, [router])
  const totalScans = entries.length
  const fakesDetected = entries.filter(e => e.verdict === 'fake').length
  const avgConfidence = totalScans > 0
    ? (entries.reduce((sum, e) => sum + e.confidence, 0) / totalScans).toFixed(1) + '%'
    : '—'
  const avgProcessing = totalScans > 0
    ? (entries.reduce((sum, e) => sum + e.analysisTime, 0) / totalScans).toFixed(2) + 's'
    : '—'

  const summaryStats = [
    {
      label: 'Total Scans',
      value: totalScans > 0 ? String(totalScans) : '—',
      delta: totalScans > 0 ? 'All time' : 'No scans yet',
    },
    {
      label: 'Fakes Detected',
      value: totalScans > 0 ? String(fakesDetected) : '—',
      delta: totalScans > 0 ? `${((fakesDetected / totalScans) * 100).toFixed(0)}% of total` : 'No scans yet',
    },
    {
      label: 'Avg Confidence',
      value: avgConfidence,
      delta: 'Per scan',
    },
    {
      label: 'Avg Processing',
      value: avgProcessing,
      delta: 'Per file',
    },
  ]

  const mediaTypeData = [
    { name: 'Image', scans: entries.filter(e => e.mediaType === 'image').length },
    { name: 'Video', scans: entries.filter(e => e.mediaType === 'video').length },
    { name: 'Audio', scans: entries.filter(e => e.mediaType === 'audio').length },
  ]

  const realCount = entries.filter(e => e.verdict === 'real').length
  const fakeCount = entries.filter(e => e.verdict === 'fake').length
  const verdictTotal = realCount + fakeCount
  const verdictData = verdictTotal > 0
    ? [
        { name: 'Authentic', value: Math.round((realCount / verdictTotal) * 100) },
        { name: 'Fake', value: Math.round((fakeCount / verdictTotal) * 100) },
      ]
    : [
        { name: 'Authentic', value: 0 },
        { name: 'Fake', value: 0 },
      ]

  return (
    <main className="min-h-screen bg-[#020617] relative overflow-hidden">
      <title>Dashboard — Fake Media Detection</title>

      <div
        className="pointer-events-none absolute inset-0 z-0"
        style={{
          backgroundSize: '100% 200%',
          backgroundImage: 'radial-gradient(ellipse 80% 50% at 50% 0%, rgba(12,185,235,0.28) 0%, transparent 65%)',
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

        <p className="text-xs uppercase tracking-[0.25em] text-blue-400 font-semibold mb-2">
          Overview
        </p>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white mt-3">
          Dashboard
        </h1>
        <p className="text-slate-400 mt-4 text-sm md:text-base leading-relaxed max-w-2xl mb-10">
          A summary of all scans, detections, and model performance across your account.
        </p>
        <p className="text-xs uppercase tracking-[0.2em] text-slate-500 font-semibold mb-4">
          Summary Statistics
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {summaryStats.map((stat) => (
            <div
              key={stat.label}
              className="bg-[#111827]/90 backdrop-blur-sm rounded-2xl border border-white/10 shadow-2xl p-5 flex flex-col gap-2"
            >
              <p className="text-3xl font-bold text-white">{stat.value}</p>
              <p className="text-xs text-slate-400 leading-snug">{stat.label}</p>
              <p className="text-xs text-blue-400 font-medium">{stat.delta}</p>
            </div>
          ))}
        </div>




        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-[#111827]/90 backdrop-blur-sm rounded-2xl border border-white/10 shadow-2xl p-6">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500 font-semibold mb-1">
              Scans by Media Type
            </p>
            <p className="text-sm text-slate-400 mb-6">
              Total scans broken down by file type.
            </p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={mediaTypeData} barCategoryGap="40%">
                <XAxis
                  dataKey="name"
                  tick={{ fill: '#94a3b8', fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#475569', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={28}
                />
                <Tooltip content={<BarTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Bar dataKey="scans" radius={[6, 6, 0, 0]}>
                  {mediaTypeData.map((_, index) => (
                    <Cell key={index} fill={BAR_COLORS[index]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-[#111827]/90 backdrop-blur-sm rounded-2xl border border-white/10 shadow-2xl p-6">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500 font-semibold mb-1">
              Verdict Breakdown
            </p>
            <p className="text-sm text-slate-400 mb-6">
              Distribution of authentic, fake, and suspicious results.
            </p>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={verdictData}
                  cx="50%"
                  cy="50%"
                  innerRadius={65}
                  outerRadius={100}
                  paddingAngle={3}
                  dataKey="value"
                  strokeWidth={0}
                >
                  {verdictData.map((_, index) => (
                    <Cell key={index} fill={PIE_COLORS[index]} opacity={0.9} />
                  ))}
                </Pie>
                <Tooltip content={<PieTooltip />} />
                <Legend
                  iconType="circle"
                  iconSize={8}
                  formatter={(value) => (
                    <span style={{ color: '#94a3b8', fontSize: '12px' }}>{value}</span>
                  )}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

        </div>
      </div>
    </main>
  )
}

