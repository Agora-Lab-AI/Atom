[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Atom
Atom is a finetuned fork of YARN LLAMA to create better LLMS through Pytorch Data!




## Installation

You can install the package using pip

```bash
pip install atom-torch

#Or

git clone https://github.com/jquesnelle/yarn
cd Atom
pip install -e .
```

### Training

To train the models, run `accelerate config` and enable DeepSpeed acceleration. `deepspeed/zero3.json` was the configuration file used for training.

```sh
./run.sh
```

The tokenized training data is available on [Hugging Face](https://huggingface.co/datasets/emozilla/pg_books-tokenized-bos-eos-chunked-65536) and was derived from the [pg19](https://huggingface.co/datasets/emozilla/pg19) dataset.

### Evaluation

To reproduce the evaluations, install [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with `pip install git+https://github.com/EleutherAI/lm-evaluation-harness` and then run the two provided scripts.

```sh
# ./scripts/eval.sh
# ./scripts/eval-harness.sh
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
