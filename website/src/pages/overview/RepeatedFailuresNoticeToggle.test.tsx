import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { renderWithProviders } from '../../test/helpers'
import RepeatedFailuresNoticeToggle, { REPEATED_FAILURES_NOTICE_KEY } from './RepeatedFailuresNoticeToggle'

const { api } = vi.hoisted(() => ({
  api: {
    kirocrewConfig: vi.fn(),
    patchConfig: vi.fn(),
    // ThemeProvider in the shared wrapper reads these on mount.
    themes: vi.fn(),
    themeDetail: vi.fn(),
    themeBoot: vi.fn(),
  },
}))
vi.mock('../../api/client', () => ({ api }))

beforeEach(() => {
  vi.clearAllMocks()
  api.themes.mockResolvedValue({ themes: [] })
  api.themeDetail.mockResolvedValue({})
  api.themeBoot.mockResolvedValue({})
  api.patchConfig.mockResolvedValue({ ok: true })
})

const theSwitch = () => screen.getByRole('switch')

describe('RepeatedFailuresNoticeToggle', () => {
  it.each([
    ['no stored value', {}],
    ['a quoted "true"', { memory: { repeated_failures_notice: 'true' } }],
  ])('shows off for %s, the reading the gateway makes', async (_case, config) => {
    api.kirocrewConfig.mockResolvedValue(config)
    renderWithProviders(<RepeatedFailuresNoticeToggle />)

    await waitFor(() => expect(theSwitch()).not.toBeDisabled())
    expect(theSwitch()).toHaveAttribute('aria-checked', 'false')
  })

  it('turns the weekly note on through the config PATCH', async () => {
    api.kirocrewConfig.mockResolvedValue({ memory: {} })
    renderWithProviders(<RepeatedFailuresNoticeToggle />)
    await waitFor(() => expect(theSwitch()).not.toBeDisabled())

    await userEvent.click(theSwitch())

    expect(api.patchConfig).toHaveBeenCalledWith(REPEATED_FAILURES_NOTICE_KEY, true)
  })

  it('says so when the save is refused', async () => {
    api.kirocrewConfig.mockResolvedValue({ memory: {} })
    api.patchConfig.mockRejectedValue(new Error('denied'))
    renderWithProviders(<RepeatedFailuresNoticeToggle />)
    await waitFor(() => expect(theSwitch()).not.toBeDisabled())

    await userEvent.click(theSwitch())

    expect(await screen.findByRole('alert')).toHaveTextContent('denied')
  })
})
