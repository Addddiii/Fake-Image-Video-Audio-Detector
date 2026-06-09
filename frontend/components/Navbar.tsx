import Image from 'next/image'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { useEffect, useState } from 'react'
import { onAuthStateChanged, signOut } from 'firebase/auth'

import { auth } from '@/utils/firebase'

const privateNavItems = [
  { label: 'Scan', href: '/' },
  { label: 'History', href: '/history' },
  { label: 'Dashboard', href: '/dashboard' },
  { label: 'About', href: '/about' }
]

export default function Navbar() {
  const router = useRouter()
  const [userName, setUserName] = useState('')

  useEffect(() => {
    if (!auth) return

    const unsubscribe = onAuthStateChanged(auth, (user) => {
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

  const isLoggedIn = Boolean(userName)

  return (
    <header className="sticky top-0 z-50 px-6 pt-4">
      <div className="mx-auto max-w-6xl">
        <nav className="flex items-center justify-between rounded-2xl border border-white/10 bg-[#0B1220]/75 px-6 py-4 shadow-[0_10px_40px_rgba(0,0,0,0.35),0_0_0_1px_rgba(59,130,246,0.08)] backdrop-blur-2xl">
          <Link href="/" className="flex shrink-0 items-center">
            <Image
              src="/assets/logo-trans.png"
              alt="LatFakeCheck Logo"
              width={182}
              height={47}
              priority
            />
          </Link>

          <div className="hidden items-center gap-1 md:flex lg:gap-2">
            {!isLoggedIn ? (
              <Link
                href="/login"
                className="rounded-lg px-5 py-2.5 text-base font-medium text-slate-200 transition hover:bg-white/5 hover:text-white"
              >
                Login / Signup
              </Link>
            ) : (
              <>
                {privateNavItems.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="whitespace-nowrap rounded-lg px-3 py-2 text-sm text-slate-300 transition hover:bg-white/5 hover:text-white lg:px-4"
                  >
                    {item.label}
                  </Link>
                ))}

                <button
                  type="button"
                  onClick={handleLogout}
                  className="whitespace-nowrap rounded-lg px-3 py-2 text-sm text-slate-300 transition hover:bg-white/5 hover:text-white lg:px-4"
                >
                  Logout
                </button>

                <span className="hidden whitespace-nowrap rounded-full border border-blue-400/20 bg-blue-500/10 px-3.5 py-2 text-sm font-medium text-slate-200 shadow-[0_0_18px_rgba(59,130,246,0.08)] xl:flex">
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