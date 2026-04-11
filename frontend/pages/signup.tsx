import Link from 'next/link'
import { useRouter } from 'next/router'
import { FormEvent, useEffect, useRef, useState } from 'react'
import {
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signOut,
  updateProfile
} from 'firebase/auth'
import Navbar from '@/components/Navbar'
import { auth, hasFirebaseConfig } from '@/utils/firebase'

function getSignupErrorMessage(error: any) {
  switch (error?.code) {
    case 'auth/email-already-in-use':
      return 'An account with this email already exists.'
    case 'auth/invalid-email':
      return 'Please enter a valid email address.'
    case 'auth/weak-password':
      return 'Password should be at least 6 characters.'
    case 'auth/network-request-failed':
      return 'Network error. Please check your internet connection and try again.'
    default:
      return 'Sign up failed. Please try again.'
  }
}

export default function SignupPage() {
  const router = useRouter()
  const skipAutoRedirectRef = useRef(false)

  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!auth) return

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user && !skipAutoRedirectRef.current) {
        router.replace('/')
      }
    })

    return unsubscribe
  }, [router])

  const handleSignup = async (e: FormEvent) => {
    e.preventDefault()
    setError('')

    if (!fullName.trim()) {
      setError('Please enter your full name.')
      return
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match.')
      return
    }

    const firebaseAuth = auth

    if (!hasFirebaseConfig || !firebaseAuth) {
      setError('Firebase is not configured yet. Add the frontend env values first.')
      return
    }

    try {
      setLoading(true)
      skipAutoRedirectRef.current = true

      const userCredential = await createUserWithEmailAndPassword(
        firebaseAuth,
        email,
        password
      )

      await updateProfile(userCredential.user, {
        displayName: fullName.trim()
      })

      await signOut(firebaseAuth)
      router.push('/login?signup=success')
    } catch (err: any) {
      setError(getSignupErrorMessage(err))
      skipAutoRedirectRef.current = false
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-[#020617] via-[#0B1120] to-[#0F172A]">
      <title>Sign Up | Fake Media Detection</title>
      <Navbar />

      <div className="max-w-6xl mx-auto px-8 pt-12 pb-12">
        <div className="max-w-md mx-auto bg-[#111827]/90 backdrop-blur-sm rounded-2xl border border-white/10 shadow-2xl p-6">
          <p className="text-xs uppercase tracking-[0.25em] text-blue-400 font-semibold">
            Account Creation
          </p>
          <h1 className="text-3xl font-bold tracking-tight text-white mt-3">
            Sign Up
          </h1>
          <p className="text-slate-400 mt-3 text-sm leading-relaxed">
            Create an account with your full name, email, and password.
          </p>

          <form onSubmit={handleSignup} className="mt-6 flex flex-col gap-4">
            <div>
              <label className="block text-sm text-slate-300 mb-2">Full name</label>
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                required
                className="w-full rounded-xl border border-white/10 bg-[#020617]/70 px-4 py-3 text-white outline-none focus:border-blue-400/50"
                placeholder="Enter your full name"
              />
            </div>

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
                  placeholder="Create a password"
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

            <div>
              <label className="block text-sm text-slate-300 mb-2">Confirm password</label>
              <div className="relative">
                <input
                  type={showConfirmPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  className="w-full rounded-xl border border-white/10 bg-[#020617]/70 px-4 py-3 pr-12 text-white outline-none focus:border-blue-400/50"
                  placeholder="Confirm your password"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword((prev) => !prev)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white transition"
                  aria-label={showConfirmPassword ? 'Hide password' : 'Show password'}
                >
                  {showConfirmPassword ? (
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
              {loading ? 'Creating account...' : 'Sign Up'}
            </button>
          </form>

          <p className="text-sm text-slate-400 mt-6 text-center">
            Already have an account?{' '}
            <Link href="/login" className="text-blue-400 hover:text-blue-300">
              Login
            </Link>
          </p>
        </div>
      </div>
    </main>
  )
}