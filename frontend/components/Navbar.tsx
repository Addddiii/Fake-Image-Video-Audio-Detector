import Link from 'next/link'
import { useRouter } from 'next/router'
import { useEffect, useState } from 'react'
import { onAuthStateChanged, signOut } from 'firebase/auth'
import { auth } from '@/utils/firebase'
import Image from 'next/image'

const privateNavItems = [
  { label: 'Results', href: '#' },
  { label: 'History', href: '#' },
  { label: 'Dashboard', href: '#' },
  { label: 'About', href: '#' }
]

export default function Navbar() {
  const router = useRouter()
  const [userEmail, setUserEmail] = useState('')
  const [userName, setUserName] = useState('')

  useEffect(() => {
    if (!auth) return

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUserEmail(user?.email || '')
      setUserName(user?.displayName || user?.email?.split('@')[0] || '')
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
          <Link href="/" className="flex items-center">
            <Image
              src="/assets/logo-trans.png"
              alt="LatFakeCheck Logo"
              width={140}
              height={36}
              priority
            />
          </Link>

          <div className="hidden md:flex items-center gap-2">
            {!isLoggedIn ? (
              <Link
                href="/login"
                className="px-4 py-2 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-white/5 transition"
              >
                Login
              </Link>
            ) : (
              <>
                {privateNavItems.map((item) => (
                  <Link
                    key={item.label}
                    href={item.href}
                    className="px-4 py-2 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-white/5 transition"
                  >
                    {item.label}
                  </Link>
                ))}

                <button
                  type="button"
                  onClick={handleLogout}
                  className="px-4 py-2 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-white/5 transition"
                >
                  Logout
                </button>

                <span className="hidden lg:block px-3 text-xs text-slate-500">
                  Welcome, <span className="text-slate-300">{userName || 'User'}</span>
                </span>
              </>
            )}
          </div>
        </nav>
      </div>
    </header>
  )
}