[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Atom
Atom is a finetuned fork of YARN LLAMA to create better LLMS through Pytorch Data!


## Installation

You can install the package using pip

```bash
pip install atom-torch
```

### Training

- To train the models, run `accelerate config` and enable DeepSpeed acceleration. `deepspeed/zero3.json` was the configuration file used for training.0
- Then run accelerate launch `accelerate launch finetune.py`


```sh
./run.sh
```

- [Dataset](kye/all-lucidrain-code-python-tokenized-65536-1)
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
