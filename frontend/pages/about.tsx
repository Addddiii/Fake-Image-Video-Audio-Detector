import Head from 'next/head'
import Link from 'next/link'
import Navbar from '@/components/Navbar'

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-[#020617] relative overflow-hidden">
      <Head>
        <title>About | LatFakeCheck</title>
      </Head>

      <div
        className="pointer-events-none absolute inset-0 z-0"
        style={{
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

      <div className="relative z-10 max-w-4xl mx-auto px-8 pt-10 pb-20">
        <p className="text-xs uppercase tracking-[0.25em] text-[#0cb9eb]/80 font-semibold mb-2">
          About
        </p>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white mt-3">
          LatFakeCheck
        </h1>
        <p className="text-slate-400 mt-4 text-sm md:text-base leading-relaxed max-w-2xl mb-16">
          An AI-powered platform for detecting synthetic and manipulated media — built by LatMatrix.
        </p>

        <Section eyebrow="Company" title="About LatMatrix">
          <p className="text-slate-400 leading-relaxed text-sm">
            LatMatrix is an India-based technology company focused on building intelligent software solutions across AI, data analytics, and digital infrastructure. With a mission to make advanced technology accessible and trustworthy, LatMatrix develops tools that address real-world challenges — from media authenticity to enterprise automation.
          </p>
          <p className="text-slate-400 leading-relaxed text-sm mt-3">
            LatFakeCheck is one of LatMatrix's applied AI initiatives, developed in partnership with RMIT University as a capstone research project exploring the frontier of deepfake and synthetic media detection.
          </p>
          <div className="flex flex-wrap gap-3 mt-6">
            <a
              href="https://www.latmatrix.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-[#0cb9eb]/30 bg-[#0cb9eb]/10 text-[#0cb9eb] text-sm font-medium hover:bg-[#0cb9eb]/20 transition"
            >
              latmatrix.com
            </a>
            <a
              href="mailto:info@latmatrix.com"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-[#0cb9eb]/30 bg-[#0cb9eb]/10 text-[#0cb9eb] text-sm font-medium hover:bg-[#0cb9eb]/20 transition"
            >
              info@latmatrix.com
            </a>

            <a
              href=""
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-[#0cb9eb]/30 bg-[#0cb9eb]/10 text-[#0cb9eb] text-sm font-medium hover:bg-[#0cb9eb]/20 transition"
            >
              +91-8308346346
            </a>

          </div>
        </Section>

       {/* project about */}
        <Section eyebrow="Project" title="What is LatFakeCheck?">
          <p className="text-slate-400 leading-relaxed text-sm">
            LatFakeCheck is a multimodal deepfake detection platform that analyses images, videos, and audio files for signs of AI generation, manipulation, or digital tampering. Users upload a file, and trained machine learning models return a verdict — Authentic or Fake — along with a confidence score and probability breakdown.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            {[
              { type: 'Image Detection', desc: 'Analyses photos for GAN artifacts, facial geometry inconsistencies, and pixel-level anomalies using EfficientNet-B0.' },
              { type: 'Video Detection', desc: 'Extracts and analyses frames for face-swap deepfakes, lip-sync inconsistencies, and temporal blending artifacts.' },
              { type: 'Audio Detection', desc: 'Uses mel spectrogram analysis to identify AI-generated speech, voice cloning, and audio splicing markers.' },
            ].map(item => (
              <div key={item.type} className="bg-[#111827]/90 rounded-2xl border border-white/10 p-4">
                <p className="text-white font-semibold text-sm mb-2">{item.type}</p>
                <p className="text-slate-400 text-xs leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>
        </Section>

            {/* tech stack */}
        <Section eyebrow="Technology" title="Tech Stack">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <p className="text-xs uppercase tracking-[0.15em] text-[#0cb9eb]/70 font-semibold mb-3">Frontend</p>
              <div className="flex flex-wrap gap-2">
                {['Next.js 14', 'TypeScript', 'Tailwind CSS', 'Firebase Auth', 'Recharts', 'React'].map(t => (
                  <Chip key={t} label={t} />
                ))}
              </div>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.15em] text-[#0cb9eb]/70 font-semibold mb-3">Backend</p>
              <div className="flex flex-wrap gap-2">
                {['FastAPI', 'Python 3.11', 'PyTorch', 'Torchvision', 'OpenCV', 'Librosa', 'Firebase Admin'].map(t => (
                  <Chip key={t} label={t} />
                ))}
              </div>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.15em] text-[#0cb9eb]/70 font-semibold mb-3">ML Models</p>
              <div className="flex flex-wrap gap-2">
                {['EfficientNet-B0', 'Custom VideoClassifier', 'AudioClassifier', 'Mel Spectrogram'].map(t => (
                  <Chip key={t} label={t} />
                ))}
              </div>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.15em] text-[#0cb9eb]/70 font-semibold mb-3">Infrastructure</p>
              <div className="flex flex-wrap gap-2">
                {['Local inference', 'SQLite / Firebase', 'localStorage history', 'REST API'].map(t => (
                  <Chip key={t} label={t} />
                ))}
              </div>
            </div>
          </div>
        </Section>

                {/* training */}
        <Section eyebrow="Research" title="Training Methodology">
          <div className="space-y-4">
            <MethodCard
              title="Image Model — EfficientNet-B0"
              points={[
                'Fine-tuned EfficientNet-B0 pretrained on ImageNet for binary classification (fake vs real).',
                'Trained on a labelled dataset of authentic and AI-generated/manipulated images.',
                'Input images are resized to 224×224 and normalised using ImageNet statistics.',
                'Output is a softmax probability over two classes with the higher confidence determining the verdict.',
              ]}
            />
            <MethodCard
              title="Video Model — Frame-based Classifier"
              points={[
                '20 frames are uniformly sampled from each uploaded video for analysis.',
                'Each frame is preprocessed and passed through the video classifier independently.',
                'Predictions are aggregated across all frames to produce a final video-level verdict.',
                'Designed to detect temporal inconsistencies and face-swap artifacts across frames.',
              ]}
            />
            <MethodCard
              title="Audio Model — Mel Spectrogram Analysis"
              points={[
                'Audio is resampled to 16kHz and trimmed/padded to a fixed 4-second window.',
                'A 128-band mel spectrogram is computed using a 1024-point FFT and 256 hop length.',
                'The spectrogram is passed through a custom AudioClassifier CNN.',
                'Trained to distinguish natural speech from AI-generated or voice-cloned audio.',
              ]}
            />
          </div>
        </Section>

              {/* known limitations */}
        <Section eyebrow="Transparency" title="Known Limitations">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { title: '1', desc: 'description of limitation 1' },
              { title: '2', desc: 'description of limitation 2' },
    
            ].map(item => (
              <div key={item.title} className="bg-[#111827]/90 rounded-2xl border border-white/10 p-4">
                <p className="text-white font-semibold text-sm mb-1">⚠ {item.title}</p>
                <p className="text-slate-400 text-xs leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>
        </Section>

            {/* T&C's */}
        <Section eyebrow="Legal" title="Terms &amp; Conditions">
          <div className="space-y-4 text-sm text-slate-400 leading-relaxed">
            <TermItem title="For educational and research use only">
              LatFakeCheck is a research prototype developed as part of an academic capstone project. It is not a certified forensic tool and is intended solely for educational, experimental, and research purposes.
            </TermItem>
            <TermItem title="No accuracy guarantee">
              The detection models are trained on publicly available datasets and may produce incorrect results. LatMatrix and the project team make no representations or warranties regarding the accuracy, reliability, or completeness of any analysis results.
            </TermItem>
            <TermItem title="Not suitable for legal decisions">
              Results produced by LatFakeCheck must not be used as evidence in legal proceedings, criminal investigations, journalistic fact-checking, or any context where a false result could cause harm to an individual or organisation.
            </TermItem>
            <TermItem title="Your data stays local">
              Uploaded files are processed locally on your machine via the backend server running on your device. Files and scan results are not transmitted to any external server or third-party service. Scan history is stored in your browser only.
            </TermItem>
            <TermItem title="No liability">
              By using this platform, you agree that LatMatrix and the project contributors are not liable for any direct, indirect, or consequential damages arising from your use of or reliance on the platform's outputs.
            </TermItem>
            <TermItem title="Responsible use">
              You agree to use this platform responsibly and not to upload content that violates applicable laws, including content involving minors, non-consensual material, or content intended to defame or harass individuals.
            </TermItem>
          </div>
        </Section>

            {/* footer */}
        <div className="mt-16 pt-8 border-t border-white/10 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <p className="text-xs text-slate-600">
            © {new Date().getFullYear()} LatMatrix. Built in partnership with RMIT University.
          </p>
          <Link
            href="/"
            className="text-xs text-[#0cb9eb]/70 hover:text-[#0cb9eb] transition"
          >
            ← Back to Scanner
          </Link>
        </div>

      </div>
    </main>
  )
}


function Section({ eyebrow, title, children }: { eyebrow: string; title: string; children: React.ReactNode }) {
  return (
    <section className="mb-14">
      <p className="text-xs uppercase tracking-[0.2em] text-[#0cb9eb]/70 font-semibold mb-1">
        {eyebrow}
      </p>
      <h2 className="text-2xl font-bold text-white mb-6" dangerouslySetInnerHTML={{ __html: title }} />
      {children}
    </section>
  )
}


function Chip({ label }: { label: string }) {
  return (
    <span className="px-3 py-1.5 rounded-full border border-white/10 bg-white/5 text-slate-300 text-xs font-medium">
      {label}
    </span>
  )
}


function MethodCard({ title, points }: { title: string; points: string[] }) {
  return (
    <div className="bg-[#111827]/90 rounded-2xl border border-white/10 p-5">
      <p className="text-white font-semibold text-sm mb-3">{title}</p>
      <ul className="space-y-2">
        {points.map((p, i) => (
          <li key={i} className="flex gap-2 text-xs text-slate-400 leading-relaxed">
            <span className="text-[#0cb9eb]/60 mt-0.5 shrink-0">›</span>
            {p}
          </li>
        ))}
      </ul>
    </div>
  )
}


function TermItem({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-[#111827]/90 rounded-2xl border border-white/10 p-5">
      <p className="text-white font-semibold text-sm mb-2">{title}</p>
      <p className="text-slate-400 text-xs leading-relaxed">{children}</p>
    </div>
  )
}
