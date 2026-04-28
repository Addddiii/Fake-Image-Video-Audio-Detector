import Link from 'next/link'
import { useRouter } from 'next/router'
import { useEffect, useState } from 'react'
import { onAuthStateChanged, signOut } from 'firebase/auth'
import { auth } from '@/utils/firebase'
import Image from 'next/image'

const privateNavItems = [
  { label: 'Results', href: '#' },
  { label: 'History', href: '/history' },
  { label: 'Dashboard', href: '/dashboard' },
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
        <nav className="flex items-center justify-between rounded-2xl border border-white/10 bg-[#0B1220]/75 backdrop-blur-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35),0_0_0_1px_rgba(59,130,246,0.08)] px-6 py-4">
          <Link href="/" className="flex items-center shrink-0">
            <Image
              src="/assets/logo-trans.png"
              alt="LatFakeCheck Logo"
              width={182}
              height={47}
              priority
            />
          </Link>

          <div className="hidden md:flex items-center gap-1 lg:gap-2">
            {!isLoggedIn ? (
              <Link
                href="/login"
                className="px-5 py-2.5 rounded-lg text-base font-medium text-slate-200 hover:text-white hover:bg-white/5 transition"
              >
                Login / Signup
              </Link>
            ) : (
              <>
                {privateNavItems.map((item) => (
                  <Link
                    key={item.label}
                    href={item.href}
                    className="px-3 lg:px-4 py-2 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-white/5 transition whitespace-nowrap"
                  >
                    {item.label}
                  </Link>
                ))}

                <button
                  type="button"
                  onClick={handleLogout}
                  className="px-3 lg:px-4 py-2 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-white/5 transition whitespace-nowrap"
                >
                  Logout
                </button>

                <span className="hidden xl:flex items-center px-3.5 py-2 rounded-full border border-blue-400/20 bg-blue-500/10 text-sm font-medium text-slate-200 shadow-[0_0_18px_rgba(59,130,246,0.08)] whitespace-nowrap">
                  Welcome, {userName || 'User'}
                </span>
              </>
            )}
          </div>
        </nav>
      </div>
    </header>
  )
}