## Use Conda-package to Ease the Environment Setup

We also provide a Pre-generated Conda Package to ease the perparation of the libraries above. 
 
Conda package download link: [py38_nsmae](https://drive.google.com/file/d/1inS_9ZUgf-WFKHGRdmJwxzAnEdcZzC8h/view?usp=sharing) (about 3.1G)

Unzip the conda package to your anaconda environment path, like so:
```bash
tar -xzvf py38_nsmae.tar.gz -C [prefix_link_of_your_anaconda]/anaconda3/envs/py38_nsmae
```

> Tip: We create this depolyable conda package with [Conda-Pack](https://github.com/conda/conda-pack). More details please refer to this repo.

Then you can check whether the environemnt is transfered successfully via the following command.

```bash
conda info -e
```

As the conda package can not be packed with editable packages, e.g., mmcv, [mmcv-full 1.4.0](https://github.com/open-mmlab/mmcv/tree/v1.4.0) requires additional setup.

After installing these dependencies, please run this command to install the codebase of this repo:

```bash
python setup.py develop
```

> Tip: [Ninja](https://github.com/ninja-build/ninja) is recommended to use before run the command above to compile the project faster!



