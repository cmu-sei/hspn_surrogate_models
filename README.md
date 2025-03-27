# HyperSPIN project code 


## Install

```sh
url="url_here"
git clone $url/hspn_surrogate_models
cd hspn_surrogate_models
pip install -e .
```

## Quickstart

## Preprocess Dataset
Used to preprocess data and create an H5 dataset for use by the models.


```sh
hspn-prepare data_dir=./data branch_files=[f_total.npy] trunk_files=[xyz.npy] output_files=[y_total.npy] output_path=./data/don_dataset.h5
```

> Note: There are more options, use `--cfg=job` to see them and read the the CLI documentation below to learn how to use this CLI.

Corresponds to the structure:

```sh
data/
| f_total.npy
| xyz.npy
| y_total.npy
| don_dataset.h5 # created
```

## Train Model

TODO: this section

```sh
hspn-train 
```

> Note: There are more options, use `--cfg=job` to see them and read the the CLI documentation below to learn how to use this CLI.


## General CLI Usage

The following applies to all CLI applications in hspn.

To see all available options:

```sh
# hspn-cli is a stand-in for any hspn cli invocation
hspn-<train/prepare/etc> ---help
hspn-<train/prepare/etc> --cfg=job # or --cfg=all
```

It is recommended to check the final config the job will execute with before running:
```sh
hspn-<train/prepare/etc> --cfg=job # or --cfg=all for verbose information
hspn-<train/prepare/etc> --cfg=job --resolve # causes variable references in the config to be resolved (resolving is always done at runtime, so this shows the final resolved config the job will use)
```

## Additional CLI Features

Each hspn CLI application can be invoked three ways. Using the `prepare` application as an example:
1. Directly: `python src/hspn/prepare.py` Cons: need the exact filepath so it depends on what your current working directory is. Pros: support shell completion so good for interactive experimentation, see below.
2. Module: `python -m hspn.prepare` Cons: No shell completion. Pros: Can be run anywhere as long as `hspn` is installed.
3. Shortcut: `hspn-prepare` this is an alias for option (2) and is installed by pip in `$HOME/.local/bin/`. Cons: No shell completion, might not be optimal in containers where `$HOME/.local/bin` is not in `$PATH`. Pros: Can be run anywhere as long as `hspn` is installed, easy to discover `hspn` commands via `hspn-<TAB><TAB>`.

### Shell autocompletion

For interactive experimentation it is recommended to use option (1) above and take advantage of shell completion which can be installed with:

```sh
hspn-<train/prepare/etc> --shell-completion install=<bash/zsh/fish>
# for a useful shorthand version:
hspn-<train/prepare/etc> -sc install=$(basename $SHELL)
```

To install train and prepare (could be placed in `~/.zshrc`/`~/.bashrc`/etc):

```sh
hspn-train -sc install=$(basename $SHELL)
hspn-prepare -sc install=$(basename $SHELL)
```

Now, you can get autocomplete while setting configuration options! Remember that you must specify the path to the file for autocomplete to work. Try:

```sh
python src/hspn/train.py model.<TAB><TAB>
```

> Note: depending on your machine completion may lag a bit.

