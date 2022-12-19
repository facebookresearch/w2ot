# Wasserstein-2 Optimal Transport (`w2ot`)

`w2ot` is JAX software by [Brandon Amos](http://bamos.github.io)
for estimating Wasserstein-2 optimal transport maps between
continuous measures in Euclidean space.
This is the official source code behind the paper
[on amortizing convex conjugates for optimal transport](https://arxiv.org/abs/2210.12153),
which also unifies and implements the dual potential training from
[Makkuva et al.](https://arxiv.org/abs/1908.10962),
[Korotin et al.](https://arxiv.org/abs/1909.13082) (Wasserstein-2 Generative Networks),
and [Taghvaei and Jalali](https://arxiv.org/abs/1902.07197).

![alg](https://user-images.githubusercontent.com/707462/197440788-3d11cc74-606b-4dc1-9e09-151b49b78c25.png)

# Getting started

## Setup

```bash
pip install -r requirements.txt
python3 setup.py develop
```

## Code structure


```bash
config # hydra config for the training setup
├── amortization
├── conjugate_solver
├── data # measures (or data) to couple
├── dual_trainer # main trainer object with model specifications
└── train.yaml # main entry point for running the code
scripts
├── analyze_2d_results.py # summarizes sweeps over 2d datasets
├── analyze_benchmark_results.py # summarizes sweeps over the W2 benchmarks
├── eval-conj-solver-benchmarks.py # evaluates the conj solver on the benchmarks
├── eval-conj-solver-lbfgs.py # ablates the LBFGS conj solvers
├── eval-conj-solver.py # evaluates the conj solver used for an experiment
├── prof-conj.py # profiles the conj solver
├── vis-2d-grid-warp.py # visualizes the grid warping by the OT map
└── vis-2d-transport.py # visualizes the transport map
w2ot # the main module
├── amortization.py # amortization choices
├── conjugate_solver.py # wrappers around conjugate solvers
├── data.py # connects all data into the same interface
├── dual_trainer.py # the main trainer for optimizing the W2 dual
├── external # Modified external code
├── models
│   ├── icnn.py # Input-convex neural network potential
│   ├── init_nn.py # An MLP amortization model
│   ├── potential_conv.py # A non-convex convolutional potential model
│   └── potential_nn.py # A non-convex MLP potential model
├── run_train.py # executable file for starting the training run
```

## Running the 2d examples

A training run can be launched with [w2ot/run_train.py](w2ot/run_train.py), which
specifies the dataset along with the choices for the models,
amortization type, and conjugate solver.
See the [config](./config) directory for all of the available
configuration options.

```bash
$ ./w2ot/run_train.py data=gauss8 dual_trainer=icnn amortization=regression conjugate_solver=lbfgs
```

This will write out the expermiental results to a local workspace
directory `<exp_dir>` that saves the latest and best models and logged metrics
about the progress.

[scripts/vis-2d-transport.py](./scripts/vis-2d-transport.py) produces
additional visualizations about the learned transport potentials and
the estimated optimal transport map:

```bash
$ ./scripts/vis-2d-transport.py <exp_dir>
```


![transport-gauss8](https://user-images.githubusercontent.com/707462/197360329-f5b406ea-93ac-4b7a-b040-e99ae02f3a18.gif)


[scripts/vis-2d-grid-warp.py](./scripts/vis-2d-grid-warp.py) provides
another visualization of how the transport warps a grid:

```bash
$ ./scripts/vis-2d-grid-warp.py <exp_dir>
```

![grid](https://user-images.githubusercontent.com/707462/197359233-2ecdef57-cb1e-4609-b244-4b46703f1ea6.gif)


Results in other 2d settings can be obtained similarly:

```bash
$ ./w2ot/run_train.py data=gauss_sq dual_trainer=icnn amortization=regression conjugate_solver=lbfgs
$ ./scripts/vis-2d-grid-warp.py <exp_dir>
```

![transport-gauss-sq](https://user-images.githubusercontent.com/707462/197360434-faf6d1f5-358f-4356-bf94-663351a77d16.gif)


## Results on settings from [Rout et al.](https://arxiv.org/abs/2110.02999)

These are the `circles`, `moons`, `s_curve`, and `swiss` datasets.

![rout](https://user-images.githubusercontent.com/707462/197428151-3f0e4c23-43e1-41fe-8dd6-7696d9ec9e06.gif)

## Results on settings from [Huang et al.](https://arxiv.org/abs/2012.05942)

These are the `maf_moon`, `rings`, and `moon_to_rings` datasets.

![huang](https://user-images.githubusercontent.com/707462/197429888-069d2eb2-9517-4af2-9a02-4bfea9c88461.gif)

## Results on image interpolations

The [image data loader](https://github.com/facebookresearch/w2ot/blob/main/config/data/images.yaml)
allows images to be used to give samples from 2-dimensional measures.
Training on samples between
[this image](https://user-images.githubusercontent.com/707462/208446984-0f0294cc-9ca2-40ba-a82a-402d51910f3c.png)
and
[this image](https://user-images.githubusercontent.com/707462/208447009-09eba823-cfca-444f-bda9-10649c6e86e9.png) gives:

```bash
./w2ot/run_train.py data=images dual_trainer=nn amortization=regression conjugate_solver=lbfgs_high_precision dual_trainer.D.dim_hidden='[512,512]' dual_trainer.D.act='leaky_relu_0.01'
```

![transport-samples-bi (1)](https://user-images.githubusercontent.com/707462/208447040-47daa776-58a1-4637-ad37-78fbc213fbc5.gif)

# Evaluating on the Wasserstein 2 benchmark

The software in this repository attains state-of-the-art performance
on the [Wasserstein-2 benchmark](https://arxiv.org/abs/2106.01954)
([code](https://github.com/iamalexkorotin/Wasserstein2Benchmark)),
which consists of two experimental settings that seek to recover known
transport maps between measures.

![w2-benchmark](https://user-images.githubusercontent.com/707462/197450170-7a11b634-cc2a-4bba-946f-1b588e8247d0.png)

## Running a single instance

The configuration and code for these experiments can be specifed
through hydra as before. To train an NN potential on the
256-dimensional HD benchmark with regression-based amortization
and an LBFGS conjugate solver, run:

```bash
$ ./w2ot/run_train.py data=benchmark_hd dual_trainer=nn_hd_benchmark amortization=regression data.input_dim=256 conjugate_solver=lbfgs
```

A single run for the CelebA part of the benchmark can similarly
be run with:

```bash
$ ./w2ot/run_train.py data=benchmark_images dual_trainer=image_benchmark data.which=Early amortization=regression conjugate_solver=lbfgs
```

## Running the main sweep
**All** of the experimental results can be obtained by launching
a sweep with hydra's multirun option.

```bash
$ ./w2ot/run_train.py -m seed=$(seq -s, 10) data=benchmark_images dual_trainer=image_benchmark data.which=Early,Mid,Late amortization=objective,objective_finetune,regression,w2gn,w2gn_finetune
$ ./train.py -m seed=$(seq -s, 10) data=benchmark_hd dual_trainer=icnn_hd_benchmark,nn_hd_benchmark amortization=objective,objective_finetune,regression,w2gn,w2gn_finetune data.input_dim=2,4,8,16,32,64,128,256
```

The following code synthesizes the results from these runs and
outputs the LaTeX source code for the tables that appear in the paper:

```bash
./analyze_benchmark_results.py <exp_root> # Output main tables.
```

# Extending this software
I have written this code to make it easy to add new
measures, dual training methods, and conjugate solvers.

## Adding new measures and data
Add a new config entry to [config/data](./config/data) pointing to
the samplers for the measures, which you can add to [w2ot/data.py](./w2ot/data.py).

## Adding a new training method
If your new method is a variant of the dual potential-based
approach, you may be able to add the right new config options
and implementations to [w2ot/dual_trainer.py](https://github.com/facebookresearch/w2ot/blob/main/w2ot/dual_trainer.py).
Otherwise, it may be simpler to copy this and create another trainer
with a similar interface.

## Adding a new conjugate solver
Add a new config entry to [config/conjugate_solver](./config/conjugate_solver)
pointing to your conjugate solver,
which should follow the same interface as the ones
in [w2ot/conjugate_solver.py](./w2ot/conjugate_solver.py).

# Licensing
Unless otherwise stated, the source code in this repository is
licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt).
The code in [w2ot/external](./w2ot/external) contains
modified external software from
[jax](https://github.com/google/jax),
[jaxopt](https://github.com/google/jaxopt),
[Wasserstein2Benchmark](https://github.com/iamalexkorotin/Wasserstein2Benchmark),
and [ott](https://github.com/ott-jax/ott)
that remain under the original license.

