/**
 * One rule for a background read of a slow endpoint, so a few open windows cannot
 * fill the browser's 6 connections per origin and queue a composer send behind them.
 *
 * - Live only while the document is visible AND the owning panel/tab is selected.
 *   Becoming live again re-reads once (`usePollRule`).
 * - Single flight: a re-read never cancels one in flight (`cancelRefetch: false`),
 *   so a key has at most one request at a time.
 * - Backoff: `pollGapMs` is the wait before the next read. While the last answer
 *   took longer than the base wait it doubles (up to `POLL_MAX_MS`), and it doubles
 *   again for each consecutive 502/503/504, dropped link or deadline. A fast
 *   answer resets it to the base.
 *
 * Polled reads use `usePollRule`; reads driven by pushes use `requestRefresh`.
 * `lib/crewWorkersPoll.ts` can adopt `timedPoll` and `pollGapMs` for its schedule.
 */
import { useQueryClient, type Query, type QueryClient, type QueryKey } from '@tanstack/react-query'
import { useEffect, useRef, useSyncExternalStore } from 'react'
import { isDeadlineError, isUnansweredRead } from '../api/queryClient'

export const POLL_MAX_MS = 60_000

const lastMs = new Map<string, number>()
const failures = new Map<string, number>()
/** Per client: `lastRun` is when a pushed read last started, `trailing` the one read queued behind the gap. */
const pushes = new WeakMap<QueryClient, { lastRun: Map<string, number>; trailing: Map<string, ReturnType<typeof setTimeout>> }>()

export const documentVisible = (): boolean => typeof document === 'undefined' || document.visibilityState !== 'hidden'

const subscribeVisibility = (cb: () => void) => {
  document.addEventListener('visibilitychange', cb)
  return () => document.removeEventListener('visibilitychange', cb)
}

/** Whether the document is visible, re-rendering on change. */
export const useDocumentVisible = (): boolean => useSyncExternalStore(subscribeVisibility, documentVisible, () => true)

/** Run one read of `id`, timing the answer and counting a consecutive failed one. */
export async function timedPoll<T>(id: string, read: () => Promise<T>): Promise<T> {
  const t0 = Date.now()
  try {
    const value = await read()
    failures.delete(id)
    return value
  } catch (error) {
    if (isUnansweredRead(error) || isDeadlineError(error)) failures.set(id, (failures.get(id) ?? 0) + 1)
    throw error
  } finally {
    lastMs.set(id, Date.now() - t0)
  }
}

/** The wait before the next read of `id` (`baseMs` when answers are fast and healthy). */
export function pollGapMs(id: string, baseMs: number): number {
  const last = lastMs.get(id) ?? 0
  let gap = baseMs
  while (gap < last && gap < POLL_MAX_MS) gap *= 2
  gap *= 2 ** Math.min(failures.get(id) ?? 0, 6)
  return Math.max(baseMs, Math.min(gap, POLL_MAX_MS))
}

/**
 * The `refetchInterval` for one polled `useQuery`; wrap its `queryFn` in
 * `timedPoll(id, ...)` so slow and failed answers stretch the gap.
 * `selected` is whether the owning panel/tab is the one on screen.
 */
export function usePollRule(id: string, baseMs: number, selected: boolean, queryKey: QueryKey) {
  const queryClient = useQueryClient()
  const visible = useDocumentVisible()
  const live = visible && selected
  const wasLive = useRef(live)
  const key = useRef(queryKey)
  key.current = queryKey
  useEffect(() => {
    if (live && !wasLive.current) readAfterCurrent(queryClient, key.current)
    wasLive.current = live
  }, [live, queryClient])
  return (): number | false => (live ? pollGapMs(id, baseMs) : false)
}

/** Queries with a follow-up read already owed, however many pushes asked for it. */
const owed = new WeakSet<Query>()

/**
 * One read of every active query under `queryKey`, never cancelling one in flight. A
 * read already out began before the push that asks for this one, so it owes exactly one
 * more read once it settles (`again`); the newer answer lands last.
 */
function readAfterCurrent(queryClient: QueryClient, queryKey: QueryKey, again?: () => void): void {
  const settled = again ?? (() => { if (documentVisible()) readAfterCurrent(queryClient, queryKey) })
  for (const query of queryClient.getQueryCache().findAll({ queryKey, type: 'active', exact: !again })) {
    if (query.state.fetchStatus !== 'fetching' || !query.promise) {
      void queryClient.refetchQueries({ queryKey: query.queryKey, exact: true }, { cancelRefetch: false })
    } else if (!owed.has(query)) {
      owed.add(query)
      void query.promise.catch(() => undefined).finally(() => { owed.delete(query); settled() })
    }
  }
}

/**
 * A push (socket frame, list change) asks for a re-read of `queryKey`. The key is marked
 * stale at once, so a reader that mounts later reads fresh. The request is single flight
 * and a push that lands during a read owes one more after it (`readAfterCurrent`); bursts
 * inside the gap collapse into one trailing read, and a hidden document waits: a stale key
 * is re-read on focus.
 */
export function requestRefresh(queryClient: QueryClient, id: string, queryKey: QueryKey, baseMs: number): void {
  void queryClient.invalidateQueries({ queryKey, refetchType: 'none' })
  if (!documentVisible()) return
  let mine = pushes.get(queryClient)
  if (!mine) pushes.set(queryClient, (mine = { lastRun: new Map(), trailing: new Map() }))
  const { lastRun, trailing } = mine
  if (trailing.has(id)) return
  const run = () => {
    trailing.delete(id)
    if (!documentVisible()) return
    lastRun.set(id, Date.now())
    readAfterCurrent(queryClient, queryKey, () => requestRefresh(queryClient, id, queryKey, baseMs))
  }
  const wait = (lastRun.get(id) ?? 0) + pollGapMs(id, baseMs) - Date.now()
  if (wait <= 0) run()
  else trailing.set(id, setTimeout(run, wait))
}

/** Test seam: forget every answer time and failure count. */
export function resetPollRule(): void {
  lastMs.clear()
  failures.clear()
}
