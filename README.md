# MaReCo

MaReCo is a library of **configurable algorithmic skeletons** for GPU programming in Elixir, built on top of [PolyHok](https://lups.inf.ufpel.edu.br/polyhok). Instead of exposing a different abstraction for each execution pattern, MaReCo provides a small set of *Map* and *Reduce* skeletons whose behavior is customized through configuration parameters, while specialized GPU implementations are selected and generated internally.

<!-- TODO: uncomment once the logo is added to the repository
<div align="center">
  <a href="https://github.com/hpfred/MaReCo">
    <img src="imgs/mareco-logo.png" alt="MaReCo Logo" width="300"/>
  </a>
</div>
-->

MaReCo is developed at the [LUPS Research Group](https://lups.inf.ufpel.edu.br) at Universidade Federal de Pelotas (UFPel), Brazil, on top of the PolyHok DSL. To know more about PolyHok, its documentation and features, please visit the [PolyHok website](https://lups.inf.ufpel.edu.br/polyhok).

## Overview

Algorithmic skeletons encapsulate recurring patterns of parallel computation as reusable higher-order functions, allowing programmers to express *what* should be computed while the skeleton decides *how* it is executed in parallel. MaReCo applies this idea to GPU programming in Elixir:

- A single **Map** abstraction covers execution patterns that are usually exposed as distinct skeletons in other libraries, such as coordinate-aware mappings, in-place updates and one- or two-dimensional data. These behaviors are selected through configuration parameters, not through different skeleton abstractions.
- **Reduce** performs a parallel aggregation over a GNX, combining partial results hierarchically in shared memory.
- The skeleton is **specialized automatically**: Elixir's pattern matching selects a function clause according to the shape of the call, and the configuration parameters determine which specialized GPU implementation is launched. This selection is transparent to the programmer.

The name MaReCo refers to the *Map* and *Reduce* skeletons it provides, in their configurable form: rather than fixed abstractions, both skeletons accept parameters that determine how the computation is executed.

On the host side, computations are expressed over GNX structures, which represent arrays allocated in GPU memory. Kernel generation, type specialization, JIT compilation and device memory management are delegated to PolyHok and remain transparent to the programmer.

## Usage

MaReCo depends on PolyHok, Nx and the CUDA backend, and declares them in its own `mix.exs`. Therefore, adding MaReCo to a project is enough to fetch everything it needs.

The installation requirements (Erlang/OTP, Elixir, CUDA Toolkit and an NVIDIA GPU) are the same as PolyHok's and are described in the [PolyHok documentation](https://lups.inf.ufpel.edu.br/polyhok).

To use MaReCo in an Elixir project, you need to do two things:

1. Add MaReCo as a dependency in your `mix.exs` file:

    ```elixir
    defp deps do
      [
        {:mareco, git: "https://github.com/hpfred/MaReCo.git"}
      ]
    end
    ```

    PolyHok, Nx and the PolyHok CUDA backend are fetched transitively and resolved to a single copy each, so there is no need to declare them. If you prefer to keep them explicit — because you use Nx or PolyHok directly in your own code, for example — declare them with exactly the same specification used by MaReCo:

    ```elixir
    defp deps do
      [
        {:nx, "~> 0.9"},
        {:poly_hok, git: "https://github.com/lups-ufpel/poly_hok.git", sparse: "poly_hok"},
        {:cuda_backend, git: "https://github.com/lups-ufpel/poly_hok.git", sparse: "backends/cuda_backend"},
        {:mareco, git: "https://github.com/hpfred/MaReCo.git"}
      ]
    end
    ```

    Mix identifies a dependency by its application name: identical specifications are resolved into a single checkout under `deps/`, while divergent ones (a different repository, ref or `sparse` path) are reported as a conflict. In that case, keep a single declaration and mark it with `override: true` in your project.

2. Configure the backend in your application. PolyHok needs to know which backend to use at runtime. You can configure the backend in your `config/runtime.exs` file. For example, to use MaReCo with the CUDA backend, your `config/runtime.exs` file should look like this:

    ```elixir
    import Config

    config :poly_hok, backend: CudaBackend
    ```

Then fetch the dependencies and compile the project:

```bash
mix deps.get
mix compile
```

PolyHok also supports an OpenCL backend, which can be added with `{:opencl_backend, git: "https://github.com/lups-ufpel/poly_hok.git", sparse: "backends/opencl_backend"}`. Note that MaReCo's compatibility with the OpenCL backend has not been evaluated, and it is developed and evaluated with the CUDA backend.

## Writing GPU Skeletons

The module `Ske` provides the programming interface of MaReCo. User-defined GPU functions are created with the `PolyHok.phok` macro, which converts the supplied function into an intermediate representation that PolyHok later incorporates into a generated CUDA kernel. The `new_gnx` and `get_gnx` primitives are responsible for transferring data between the host and GPU memory.

The following example squares each element of a vector with the *Map* skeleton:

```elixir
require PolyHok
use Ske

arr = Nx.tensor([1, 2, 3, 4])
fun = PolyHok.phok fn y -> y * y end

a = PolyHok.new_gnx(arr)

result = Ske.map(a, fun, [])
         |> PolyHok.get_gnx()
```

The general form of the *Map* skeleton is

$$\texttt{map(<gnx...>, <function> [, params...] [, options])} \rightarrow \texttt{gnx}$$

where the invocation receives one to three GNX structures, the function to be executed, an optional list of up to three auxiliary parameters, and a list of configuration options. The auxiliary parameters are forwarded unchanged to every invocation of the user-defined function, which allows computations of different arities to be expressed through the same skeleton.

The *Reduce* skeleton aggregates a single GNX according to

$$\texttt{reduce(<gnx>, <initial>, <function>)} \rightarrow \texttt{value}$$

where `initial` is the initial accumulator value and the function must be associative, so that different evaluation orders produce the same result.

### Map configuration options

$$[\texttt{coord: true|\underline{false}},\ \texttt{return: \underline{true}|false},\ \texttt{dim: \underline{:one}|:two}]$$

| Option | Description |
| --- | --- |
| `coord` | Passes the coordinates of the processed element as additional arguments to the function, allowing position-aware computations. |
| `return` | Specifies whether the function produces a value stored in a newly allocated GNX, or whether no output GNX is allocated and the function handles the result directly through the provided GNX structures. |
| `dim` | Specifies whether the computation is interpreted over one- or two-dimensional data. |

Omitted options assume the underlined default values. The combination of these options with the number of input structures and auxiliary parameters determines which specialized implementation MaReCo dispatches internally, so that a single `Ske.map` call expresses patterns that would otherwise require multiple skeleton abstractions.

<details>
<summary><strong>Note: structured data</strong></summary>

Each GNX carries a *step* attribute describing how many consecutive positions in the underlying one-dimensional buffer correspond to a single logical element. When set, the skeleton advances through the array according to this value and passes the grouped values as separate arguments to the user-defined function. This makes it possible to represent structured data, such as particles with multiple attributes, while preserving a contiguous memory layout suitable for GPU execution. This mechanism is what allows, for instance, the N-Body and Ray Tracer benchmarks to operate on multi-attribute elements.

</details>

## Repository Structure

Below we list the components of the MaReCo repository:

- `lib/poly_hok/`: The MaReCo library itself. `ske.ex` implements the `Ske` module, exposing the configurable *Map* and *Reduce* skeletons and the specialization mechanism that selects and launches the corresponding GPU kernel.
- `ph_includes/`: C headers that are injected into the GPU code generated by PolyHok.
- `benchmarks/ske_lib/`: The benchmark applications written with the MaReCo skeletons, listed below.
- `config/runtime.exs`: Runtime configuration of the PolyHok backend used by MaReCo.
- `mix.exs`: Project definition. Declares PolyHok, Nx and the CUDA backend as dependencies.
- `run_all.sh`: Script that runs the benchmark suite in sequence.

## Running the Benchmarks

The benchmarks are already included in the `benchmarks/` directory, so no extra setup is required beyond fetching the dependencies and compiling the project:

```bash
mix deps.get
mix compile
```

Each benchmark receives the problem size as a command-line argument. For example, to run the dot product benchmark with vectors of 10,000 elements:

```bash
mix run benchmarks/ske_lib/dot_product.ex 10000
```

The benchmarks evaluated in the accompanying publication are:

| Benchmark | File | Skeletons | Configuration Used |
| --- | --- | --- | --- |
| Dot Product | `dot_product.ex` | Map, Reduce | Default (`return: true`, `coord: false`, `dim: :one`) |
| N-Body | `nbodies.ex` | Map | `return: false`, step = 6 |
| Ray Tracer | `raytracer.ex` | Map | `dim: :two`, `return: false`, `coord: true`, step = 4 |
| Nearest Neighbor | `nearest_neighbor.ex` | Map, Reduce | `return: false` |
| SAXPY | `saxpy_rts.ex` | Map | Default (`return: true`, `coord: false`, `dim: :one`) |

The directory also contains a Julia set benchmark (`julia.ex`) and a matrix multiplication benchmark (`mm.ex`), both written with the same skeletons but not included in the evaluation reported in the publication. Note that the benchmarks may eventually be moved to a separate repository.

## Publication

MaReCo is described in the paper *"MaReCo: Configurable Map and Reduce Algorithmic Skeletons with Closure Support for GPUs in Elixir"*, by Frederico Peixoto Antunes, André Rauber Du Bois and Gerson Geraldo Homrich Cavalheiro, to be published in the proceedings of SSCAD.

<!-- TODO: add the link to the proceedings once the event takes place. -->

## License

<!-- TODO: uncomment once the LICENSE file is added to the repository
MaReCo is licensed under the [MIT License](LICENSE).
-->

A license file has not been added to this repository yet.
