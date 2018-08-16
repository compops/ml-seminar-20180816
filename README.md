# Material for presentation "Software for Machine learning"

The presentation slides are found in `ml-software.pdf` and the simple code for simulating a Gaussian process is found in `python_syntax.py`. The folder `deep-learning` contains two examples: `first_mnist.py` and `second_imagenet.py` containing the code for learning a deep neural network to classify handwritten digits and pictures respectively. The weights for the latter are downloaded from the internet so no training is required.

To install Anaconda, visit https://www.anaconda.com/download/ to download and install most packages. For the deep learning code you need to install `tensorflow` and `keras`. Probably by running
``` bash
pip install --upgrade tensorflow keras
```

However, it is good practice to create a seperate environment for this, see https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands.
Then activate the created environment with
``` bash
activate myenv
```
for Windows and
``` bash
source activate myenv
```

for MacOS and Linux. Here, myenv is the name of your environment for deep learning. Then execute the pip command as above. After this the code should run just fine but you might have to adjust the search path in `second_imagenet.py` to point at the correct images.
