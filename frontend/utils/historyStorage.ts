export type HistoryMediaType = 'image' | 'video' | 'audio'
export type HistoryVerdict = 'fake' | 'real'

export interface HistoryEntry {
  id: string
  fileName: string
  mediaType: HistoryMediaType
  verdict: HistoryVerdict
  confidence: number
  fakeProbability: number
  realProbability: number
  analysisTime: number
  analyzedAt: string
}

const MAX_HISTORY_ITEMS = 50

function getStorageKey(userKey: string) {
  return `latfakecheck_history_${userKey}`
}

export function getHistoryEntries(userKey: string): HistoryEntry[] {
  if (typeof window === 'undefined' || !userKey) return []

  try {
    const raw = localStorage.getItem(getStorageKey(userKey))
    if (!raw) return []

    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function addHistoryEntry(
  userKey: string,
  entry: Omit<HistoryEntry, 'id'>
): HistoryEntry[] {
  if (typeof window === 'undefined' || !userKey) return []

  const existing = getHistoryEntries(userKey)

  const next: HistoryEntry[] = [
    {
      id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      ...entry,
    },
    ...existing,
  ].slice(0, MAX_HISTORY_ITEMS)

  localStorage.setItem(getStorageKey(userKey), JSON.stringify(next))
  return next
}

export function clearHistoryEntries(userKey: string) {
  if (typeof window === 'undefined' || !userKey) return
  localStorage.removeItem(getStorageKey(userKey))
}