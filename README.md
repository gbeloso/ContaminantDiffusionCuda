# ContaminantDiffusionCuda
<h1 align="center" style="font-weight: bold;">Contaminant Diffusion using Cuda💻</h1>

<p align="center">
 <a href="#tech">Technologies</a> • 
 <a href="#started">Getting Started</a> •
</p>

<p align="center">
    <b>Create a simulation to model the diffusion of contaminants in a body of water (such as a lake or river), applying parallelism concepts to accelerate calculations and observe the behavior of pollutants over time. The project will investigate the impact of OpenMP, CUDA, and MPI on execution time and model accuracy. </b>
</p>

<h2 id="technologies">💻 Technologies</h2>

- CUDA
- C
- Jupyter-Notebook
- Anaconda

<h2 id="started">🚀 Getting started</h2>

<h3>Prerequisites</h3>

- A GPU
- NVCC
- GCC
- Jupyter-notebook

<h3>Cloning</h3>

How to clone your project

```bash
git clone https://github.com/gbeloso/ContaminantDiffusionCuda
```

<h3>Starting</h3>
After cloning, in the makefile change the architecture to match your GPU, this link can help: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list

```bash
cd ContaminantDiffusionCuda
make
./tests/test_cuda.sh
./tests/test_seq.sh
```
to run again first you need to execute:

```bash
make clean
```

<h3>Documentations that might help</h3>

[📝 How to create a Pull Request](https://www.atlassian.com/br/git/tutorials/making-a-pull-request)

[💾 Commit pattern](https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716)
