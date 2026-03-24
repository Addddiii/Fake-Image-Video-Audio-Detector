import { useState } from 'react'
import Navbar from '@/components/Navbar'

type MediaType = 'image' | 'video' | 'audio'

export default function Home() {
  const [activeTab, setActiveTab] = useState<MediaType>('image')
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)

  const tabs: MediaType[] = ['image', 'video', 'audio']

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) setFile(dropped)
  }

  return (
    <main className="min-h-screen bg-stone-100">
      <Navbar />
      <div className="max-w-5xl mx-auto p-8 grid grid-cols-2 gap-12 mt-8">
        <div className="bg-white rounded-xl shadow p-6 flex flex-col gap-4">
          <div className="flex gap-2">
            {tabs.map(tab => (
              <button
                key={tab}
                onClick={() => { setActiveTab(tab); setFile(null) }}
                className={`px-4 py-1.5 rounded-full text-sm font-medium uppercase tracking-wide transition ${
                  activeTab === tab ? 'bg-gray-800 text-white' : 'bg-stone-200 text-gray-600 hover:bg-stone-300'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
          <div
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl flex flex-col items-center justify-center h-56 cursor-pointer transition ${
              dragging ? 'border-gray-500 bg-stone-100' : 'border-stone-300 bg-stone-50'
            }`}
          >
            {file ? (
              <p className="text-sm text-gray-700 font-medium">{file.name}</p>
            ) : (
              <>
                <p className="text-gray-400 text-sm">Drag & drop your {activeTab} here</p>
                <p className="text-gray-300 text-xs mt-1">or click to browse</p>
              </>
            )}
          </div>
          <button className="w-full bg-gray-800 text-white py-2.5 rounded-lg font-semibold hover:bg-gray-700 transition">
            ANALYSE
          </button>
        </div>

        <div className="flex flex-col justify-center gap-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-800">Detect AI Generated Media Instantly!</h1>
            <p className="text-gray-500 mt-2 text-sm leading-relaxed">
              Upload any image, video or audio file and our AI-powered detection model will 
              analyse it for signs of AI creation or manipulation.
            </p>
            <br></br>
            <p className="text-gray-500 mt-2 text-sm leading-relaxed">
              Get confidence scores and a detailed breakdown in less than 3 seconds!
            </p>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white rounded-xl shadow p-4">
              <p className="text-xs text-gray-400 uppercase tracking-wide">Confidence Score</p>
              <p className="text-2xl font-bold text-gray-800 mt-1">—</p>
            </div>
            <div className="bg-white rounded-xl shadow p-4">
              <p className="text-xs text-gray-400 uppercase tracking-wide">Analysis Time</p>
              <p className="text-2xl font-bold text-gray-800 mt-1">—</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
