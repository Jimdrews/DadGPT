# Contributing to DadGPT

Thanks for helping out. Bug reports, fixes and small focused features are all welcome.

## Setup

Follow [Requirements](README.md#requirements) and [Build](README.md#build) in the README. In
short, you need an NVIDIA GPU, the CUDA Toolkit, CMake, Ninja and vcpkg.

## Before opening a pull request

1. **Run the tests on a GPU.** CI only checks formatting and compiles the code, because GitHub's
   runners have no GPU. So please run the tests yourself:

   ```sh
   ctest --preset windows            # Linux: default
   ```

   If you change a kernel, also run `gpu-memcheck` and `gpu-racecheck`:

   ```sh
   cmake --build --preset default --target gpu-racecheck
   ```

2. **Format the code** with clang-format 18 (CI checks this):

   ```sh
   cmake --build --preset default --target format
   ```

3. **Keep commits focused.** Write messages in the imperative mood and say what changed and why,
   e.g. `Fix softmax/cross-entropy race and tied-embedding gradient overwrite`.

## Reporting bugs

Open an issue with your GPU, OS, CUDA version and compiler, the exact command you ran, and its
output. The bug report template asks for all of these.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
