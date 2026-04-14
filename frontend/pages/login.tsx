import Link from 'next/link'
import { useRouter } from 'next/router'
import { FormEvent, useEffect, useState } from 'react'
import {
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signInWithPopup
} from 'firebase/auth'
import Navbar from '@/components/Navbar'
import {
  auth,
  googleProvider,
  hasFirebaseConfig
} from '@/utils/firebase'

function GoogleIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5 shrink-0" aria-hidden="true">
      <path
        fill="#EA4335"
        d="M12 10.2v3.9h5.4c-.2 1.3-.8 2.4-1.8 3.2l3 2.3c1.8-1.6 2.8-4 2.8-6.8 0-.7-.1-1.3-.2-1.9H12z"
      />
      <path
        fill="#34A853"
        d="M12 21c2.4 0 4.5-.8 6-2.2l-3-2.3c-.8.6-1.8 1-3 .9-2.3 0-4.3-1.6-5-3.7H3.8V16c1.5 3 4.6 5 8.2 5z"
      />
      <path
        fill="#4A90E2"
        d="M7 13.7c-.2-.6-.3-1.2-.3-1.7s.1-1.2.3-1.7V8H3.8C3.3 9.2 3 10.6 3 12s.3 2.8.8 4L7 13.7z"
      />
      <path
        fill="#FBBC05"
        d="M12 6.6c1.3 0 2.5.5 3.4 1.3l2.6-2.6C16.4 3.8 14.4 3 12 3 8.4 3 5.3 5 3.8 8L7 10.3c.7-2.1 2.7-3.7 5-3.7z"
      />
    </svg>
  )
}

function getLoginErrorMessage(error: any) {
  switch (error?.code) {
    case 'auth/invalid-credential':
    case 'auth/wrong-password':
    case 'auth/user-not-found':
      return 'Incorrect email or password.'
    case 'auth/invalid-email':
      return 'Please enter a valid email address.'
    case 'auth/network-request-failed':
      return 'Network error. Please check your internet connection and try again.'
    case 'auth/popup-closed-by-user':
      return 'The sign-in popup was closed before finishing.'
    case 'auth/cancelled-popup-request':
      return 'Another sign-in popup is already open.'
    default:
      return 'Login failed. Please try again.'
  }
}

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const signupSuccess = router.query.signup === 'success'
  const logoutSuccess = router.query.logout === 'success'

  useEffect(() => {
    if (!auth) return

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        router.replace('/')
      }
    })

    return unsubscribe
  }, [router])

  const handleEmailLogin = async (e: FormEvent) => {
    e.preventDefault()
    setError('')

    const firebaseAuth = auth

    if (!hasFirebaseConfig || !firebaseAuth) {
      setError('Firebase is not configured yet. Add the frontend env values first.')
      return
    }

    try {
      setLoading(true)
      await signInWithEmailAndPassword(firebaseAuth, email, password)
      router.push('/')
    } catch (err: any) {
      setError(getLoginErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  const handleGoogleLogin = async () => {
    setError('')

    const firebaseAuth = auth

    if (!hasFirebaseConfig || !firebaseAuth) {
      setError('Firebase is not configured yet. Add the frontend env values first.')
      return
    }

    try {
      setLoading(true)
      await signInWithPopup(firebaseAuth, googleProvider)
      router.push('/')
    } catch (err: any) {
      setError(getLoginErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-[#020617] via-[#0B1120] to-[#0F172A]">
      <title>Login | Fake Media Detection</title>
      <Navbar />

      <div className="max-w-6xl mx-auto px-8 pt-12 pb-12">
        <div className="max-w-md mx-auto bg-[#111827]/90 backdrop-blur-sm rounded-2xl border border-white/10 shadow-2xl p-6">
          <p className="text-xs uppercase tracking-[0.25em] text-blue-400 font-semibold">
            Account Access
          </p>
          <h1 className="text-3xl font-bold tracking-tight text-white mt-3">
            Login
          </h1>
          <p className="text-slate-400 mt-3 text-sm leading-relaxed">
            Sign in with your email and password, or continue with Google.
          </p>

          {signupSuccess && (
            <p className="mt-4 rounded-xl border border-emerald-400/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-300">
              Account created successfully. Please log in.
            </p>
          )}

          {logoutSuccess && (
            <p className="mt-4 rounded-xl border border-blue-400/20 bg-blue-500/10 px-4 py-3 text-sm text-blue-300">
              Logged out successfully.
            </p>
          )}

          <form onSubmit={handleEmailLogin} className="mt-6 flex flex-col gap-4">
            <div>
              <label className="block text-sm text-slate-300 mb-2">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full rounded-xl border border-white/10 bg-[#020617]/70 px-4 py-3 text-white outline-none focus:border-blue-400/50"
                placeholder="Enter your email"
              />
            </div>

            <div>
              <label className="block text-sm text-slate-300 mb-2">Password</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="w-full rounded-xl border border-white/10 bg-[#020617]/70 px-4 py-3 pr-12 text-white outline-none focus:border-blue-400/50"
                  placeholder="Enter your password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((prev) => !prev)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white transition"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? (
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      className="w-5 h-5"
                    >
                      <path d="M3 3l18 18" />
                      <path d="M10.6 10.7a2 2 0 002.7 2.7" />
                      <path d="M9.9 4.2A10.9 10.9 0 0112 4c5 0 9.3 3.1 11 8-1 2.6-2.8 4.7-5 6" />
                      <path d="M6.2 6.2C4.4 7.6 3 9.6 2 12c1.7 4.9 6 8 10 8 1.6 0 3.1-.3 4.4-.9" />
                    </svg>
                  ) : (
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      className="w-5 h-5"
                    >
                      <path d="M2 12s3.5-8 10-8 10 8 10 8-3.5 8-10 8-10-8-10-8z" />
                      <circle cx="12" cy="12" r="3" />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {error && <p className="text-sm text-red-400">{error}</p>}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-xl font-semibold tracking-wide transition-all duration-200 bg-gradient-to-r from-blue-600 to-cyan-500 text-white hover:scale-[1.01] active:scale-[0.99] shadow-[0_0_25px_rgba(59,130,246,0.25)] disabled:opacity-70"
            >
              {loading ? 'Signing in...' : 'Login'}
            </button>
          </form>

          <div className="mt-6 flex items-center gap-3">
            <div className="h-px flex-1 bg-white/10" />
            <span className="text-xs uppercase tracking-[0.2em] text-slate-500">or</span>
            <div className="h-px flex-1 bg-white/10" />
          </div>

          <div className="mt-6 flex flex-col gap-3">
            <button
              type="button"
              onClick={handleGoogleLogin}
              disabled={loading}
              className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-white hover:bg-white/10 transition flex items-center justify-center gap-3"
            >
              <GoogleIcon />
              <span>Continue with Google</span>
            </button>
          </div>

          <p className="text-sm text-slate-400 mt-6 text-center">
            Don&apos;t have an account?{' '}
            <Link href="/signup" className="text-blue-400 hover:text-blue-300">
              Sign up
            </Link>
          </p>
        </div>
      </div>
    </main>
  )
}