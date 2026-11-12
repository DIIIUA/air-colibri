/* The tier-only DLL uses no CRT or Windows APIs. Exact-width MSVC types let
 * this pure-integer bridge build without a Windows SDK installation. */
#ifndef AIRCOLIBRI_FREESTANDING_STDINT_H
#define AIRCOLIBRI_FREESTANDING_STDINT_H
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#endif
