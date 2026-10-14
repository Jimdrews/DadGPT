## What and why

<!-- What this changes and the reason for it. Link any related issue. -->

## Checklist

- [ ] Tests pass locally on a GPU (`ctest --preset ...`). CI can't run them.
- [ ] Formatted with clang-format 18 (`--target format`)
- [ ] Ran `gpu-memcheck` / `gpu-racecheck` if this touches CUDA kernels
