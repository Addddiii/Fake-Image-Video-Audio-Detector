import Link from 'next/link'
export default function Navbar() {
  return (
    <nav className="bg-gray-800 text-white px-8 py-4 flex justify-between items-center">
      <span className="font-bold text-lg">Fake Media Detection</span>
      <div className="flex gap-6 text-sm">
        {['Login', 'Results', 'Dashboard', 'History', 'About'].map(link => (
          <Link key={link} href={`/${link.toLowerCase()}`} className="hover:text-gray-300">{link}</Link>
        ))}
      </div>
    </nav>
  )
}
