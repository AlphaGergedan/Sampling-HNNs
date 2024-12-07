<div align="center">
  <h1>
    Training Hamiltonian neural networks without backpropagation
  </h1>

  <h4>
    Source code for our <a href="https://arxiv.org/abs/2411.17511">paper</a> on approximating Hamiltonian with sampled neural networks
  </h4>
</div>

<details open>
  <summary>
    <h2>Models</h2>
  </summary>

  Available models are:
  - **MLP**: ODE-Net, directly approximates `q_dot` and `p_dot`. <a href="https://arxiv.org/abs/1806.07366">paper</a>
  - **HNN**: Hamiltonian neural network approximates `H`, then `q_dot` and `p_dot` are recovered
  using automatic differentiation and Hamilton's equations. <a href="https://arxiv.org/abs/1906.01563v2">paper</a>

  All the models are available in sampled form. Sampled models have the **S-** prefix, which stands
  for **Sampled**. In this case, the model's hidden layer parameters are sampled, and the network's
  last layer is set using the least-squares solution. Different sampling options and resampling
  using approximate values are available for the SWIM method. <a href="https://arxiv.org/abs/2306.16830">paper</a>

  Here are the sampled ODE-Net (S-MLP) and HNN (S-MLP) models with their architecture illustrations for comparison:
  <div align="center">
    <img src="/assets/smlp-shnn.png" />
  </div>
</details>

---

<details close>
  <summary>
    <h2>Setup</h2>
  </summary>

  Create the conda environment:
  ```sh
  conda env create --file=environments.yml
  ```
  Then activate it with `conda activate s-hnn`.

  ### Examples
  After setting up the conda environment, you can use the bash script `main` located at the root of the
  project.
  - Run `./main --help` for usage.
  - Training a traditional network: `./main --target single_pendulum --model {MLP,HNN}`
  - Sampling a network: `./main --target single_pendulum --model {S-MLP,S-HNN}`

  Here is an example to quickly train a Sampled-HNN for single pendulum:
  ```sh
  python src/main.py --target single_pendulum --model S-HNN
  ```

  First-order error correction example:
  ```sh
  python src/main_limited_data.py --target single_pendulum --model S-HNN
  ```

  For details you can refer to our paper <a href="https://arxiv.org/abs/2411.17511">paper</a>.
</details>

---

<details close>
  <summary>
    <h2>Paper experiments</h2>
  </summary>

  - All the experiment results listed in our paper, including all the trained models, are stored under `/experiments` as pickle files and categorized.
  - In order to reproduce the experiments, refer to the scripts `/src/*experiment.py`.
  - In order to analyze the results we prepared notebooks located at the root of the project `/analyze-*.ipynb`.
  - The scripts `/batch*.sh` are used to conduct the experiments listed in our paper in a cluster environment.
  - The notebook `error_correction_demonstration.ipynb` contains error correction experiments.
</details>

---

<details open>
  <summary>
    <h2>Citation</h2>
  </summary>

  If you use Sampled-HNNs in your research, please cite our <a href="https://arxiv.org/abs/2411.17511">paper</a>.

</details>
