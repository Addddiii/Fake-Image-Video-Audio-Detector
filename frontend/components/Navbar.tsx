import Link from 'next/link'
import { useRouter } from 'next/router'
import { useEffect, useState } from 'react'
import { onAuthStateChanged, signOut } from 'firebase/auth'
import { auth } from '@/utils/firebase'

const navItems = [
  { label: 'Results', href: '#' },
  { label: 'Dashboard', href: '#' },
  { label: 'History', href: '#' },
  { label: 'About', href: '#' }
]

export default function Navbar() {
  const router = useRouter()
  const [userEmail, setUserEmail] = useState('')

  useEffect(() => {
    if (!auth) return

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUserEmail(user?.email || '')
    })

    return unsubscribe
  }, [])

  const handleLogout = async () => {
    if (!auth) return

    try {
      await signOut(auth)
      router.push('/login?logout=success')
    } catch (error) {
      console.error('Logout failed:', error)
    }
  }

  const isLoggedIn = Boolean(userEmail)

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
            {!isLoggedIn && (
              <>
                <Link
                  href="/login"
                  className="px-4 py-2 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-white/5 transition"
                >
                  Login
                </Link>
                <Link
                  href="/signup"
                  className="px-4 py-2 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-white/5 transition"
                >
                  Sign Up
                </Link>
              </>
            )}

            {navItems.map((item) => (
              <Link
                key={item.label}
                href={item.href}
                className="px-4 py-2 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-white/5 transition"
              >
                {item.label}
              </Link>
            ))}

            {isLoggedIn && (
              <>
                <span className="hidden lg:block max-w-[180px] truncate px-3 text-xs text-slate-500">
                  {userEmail}
                </span>
                <button
                  type="button"
                  onClick={handleLogout}
                  className="px-4 py-2 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-white/5 transition"
                >
                  Logout
                </button>
              </>
            )}
          </div>
        </nav>
      </div>
    </header>
  )
}