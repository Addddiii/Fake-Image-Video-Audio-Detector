'use client'

import Navbar from '@/components/Navbar'
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

// PLACE HOLDER DATA
const summaryStats = [
  { label: 'Total Scans', value: '21', delta: '+3 this week' },
  { label: 'Fakes Detected', value: '4', delta: '19% of total' },
  { label: 'Avg Confidence', value: '86%', delta: '+2% vs last week' },
  { label: 'Avg Processing', value: '2.4s', delta: 'Per file' },
]

const mediaTypeData = [
  { name: 'Image', scans: 12 },
  { name: 'Video', scans: 5 },
  { name: 'Audio', scans: 4 },
]

const verdictData = [
  { name: 'Authentic', value: 55 },
  { name: 'Fake', value: 31 },
  { name: 'Suspicious', value: 14 },
]

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
  return (
    <main className="min-h-screen bg-gradient-to-b from-[#020617] via-[#0B1120] to-[#0F172A]">
      <title>Dashboard — Fake Media Detection</title>
      <Navbar />

      <div className="max-w-6xl mx-auto px-8 pt-10 pb-16">

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
