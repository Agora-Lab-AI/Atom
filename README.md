[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Python Package Template
A easy, reliable, fluid template for python packages complete with docs, testing suites, readme's, github workflows, linting and much much more


## Installation

You can install the package using pip

```python
git clone https://github.com/jquesnelle/yarn
cd Atom
pip install -e .
```

### Training

To train the models, run `accelerate config` and enable DeepSpeed acceleration. `deepspeed/zero3.json` was the configuration file used for training.

```sh
# ./train.sh
```

The tokenized training data is available on [Hugging Face](https://huggingface.co/datasets/emozilla/pg_books-tokenized-bos-eos-chunked-65536) and was derived from the [pg19](https://huggingface.co/datasets/emozilla/pg19) dataset.

### Evaluation

To reproduce the evaluations, install [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with `pip install git+https://github.com/EleutherAI/lm-evaluation-harness` and then run the two provided scripts.

```sh
# ./eval.sh
# ./eval-harness.sh
```

### Citation

```
@misc{peng2023yarn,
      title={YaRN: Efficient Context Window Extension of Large Language Models}, 
      author={Bowen Peng and Jeffrey Quesnelle and Honglu Fan and Enrico Shippole},
      year={2023},
      eprint={2309.00071},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```