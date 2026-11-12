/* Thin ABI over the unchanged upstream Colibri tier.h. */
#include "tier.h"
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
API int ac_tier_abi(void) { return 1; }
API int ac_tier_promote(uint32_t hot, uint32_t cold) { return tier_should_promote(hot, cold); }
API void ac_tier_decay(uint32_t *heat, int count) { tier_decay(heat, count); }
API int ac_tier_pick(const uint32_t *heat, const uint32_t *last, uint32_t clock,
                    int count, const int *pinned, int npin, int *slot, int *expert) {
    long gain = 0;
    return tier_pick_lfru(heat, last, clock, count, pinned, npin, slot, expert, &gain);
}
