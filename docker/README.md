## Use Dockerfile to Ease the Environment Setup
You can also refer to the [Dockerfile](docker/Dockerfile) to ease the partial environment setup. To get started with docker, please make sure that `nvidia-docker` is installed on your machine. After that, please execute the following command to build the docker image:

```bash
cd docker && docker build . -t nsmae
```

We can then run the docker with the following command:

```bash
nvidia-docker run -it -v `pwd`/../data:/dataset --shm-size 16g nsmae /bin/bash
```

We recommend the users to run data preparation (instructions are available in the next section) outside the docker if possible. Note that the dataset directory should be an absolute path. Within the docker, please run the following command to clone our repo and install custom CUDA extensions:

```bash
cd home && git clone https://github.com/JerryX1110/NeRF-Supervised-MAE/nsmae && cd nsmae
python setup.py develop
```

Tip: [Ninja](https://github.com/ninja-build/ninja) is recommended to use before run the command above to compile the project faster!

You can then create a symbolic link `data` to the `/dataset` directory in the docker.
