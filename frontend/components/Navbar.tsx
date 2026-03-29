import Link from 'next/link'

const navItems = [
  { label: 'Login', href: '#' },
  { label: 'Results', href: '#' },
  { label: 'Dashboard', href: '#' },
  { label: 'History', href: '#' },
  { label: 'About', href: '#' }
]

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 px-6 pt-4">
      <div className="max-w-6xl mx-auto">
        <nav className="flex items-center justify-between rounded-2xl border border-white/10 bg-[#0B1220]/70 backdrop-blur-xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] px-6 py-4">
          <Link
            href="/"
            className="text-white font-semibold text-2xl tracking-tight"
          >
            Fake Media Detection
          </Link>

          <div className="hidden md:flex items-center gap-2">
            {navItems.map((item) => (
              <Link
                key={item.label}
                href={item.href}
                className="px-4 py-2 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-white/5 transition"
              >
                {item.label}
              </Link>
            ))}
          </div>
        </nav>
      </div>
    </header>
  )
}