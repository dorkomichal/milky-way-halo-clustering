# Pre-requisites
Extreme Deconvolution library is needed by the project.\
If using [Docker image](#using-docker) setup is included in the image build. 

If not using Docker please see instructions on how to setup here:
https://github.com/jobovy/extreme-deconvolution

After setting up extreme-deconvolution also remember to add the ``extreme-deconvolution/py`` directory to ``PYTHONPATH`` to enable Python to find XD.
# Environment setup
## Using pip
Install required packages:
```
pip install --no-cache-dir -r requirements.txt
```
## Using Conda
Create virtual environment:
```
conda env create -n <name> -f environment.yaml
```
Activate environment:
```
conda activate <name>
```
## Using Docker
To build container from Dockerfile in the root of the repository run:
```
docker build -t galaxy_halo_image .
```
After build completes start up the container using following command to start-up the container:
```
docker run --rm -d -t --name=galaxy_halo_solutions galaxy_halo_image
```
# Running the code
The code is designed as python module with ``halo_clustering/__main__.py`` file as entry point. 
Executable expects two required arguments ``--galah`` and ``--apogee`` which specify the path to the datasets. Datasets are available in the ``datasets`` directory.
Additional optional argument ``--multiprocess`` can be specified which runs GMM fitting across multiple processes which is more suitable for HPC with more resources.

## Outside Docker
To run the executable simply execute from the root of the repository:
```
python -m halo_clustering --galah=datasets/galah_gaia_data_subset.csv --apogee=datasets/apogee_gaia_data_subset.csv
```
To enable multiprocess fitting specify ``--multiprocess`` flag such as:
```
python -m halo_clustering --galah=datasets/galah_gaia_data_subset.csv --apogee=datasets/apogee_gaia_data_subset.csv --multiprocess
```
## Inside Docker
To run the executable inside the Docker container invoke python through ```docker exec``` as:
```
docker exec -ti galaxy_halo_solutions python -m halo_clustering --galah=datasets/galah_gaia_data_subset.csv --apogee=datasets/apogee_gaia_data_subset.csv
```
With multiprocessing:
```
docker exec -ti galaxy_halo_solutions python -m halo_clustering --galah=datasets/galah_gaia_data_subset.csv --apogee=datasets/apogee_gaia_data_subset.csv --multiprocess
```

# Output plots
Plots are saved into ```output``` directory in the root of the repository (this directory is created by corresponding scripts and doesn't exist by default).

### When using Docker
When running solutions inside the Docker container users have two options to view output figures
1. Connecting to container using interactive shell and viewing plots inside ```output``` directory:
```
docker exec -t -i galaxy_halo_solutions /bin/bash
```
2. Copying contents of the ```output``` directory from inside the container into local filesystem (and then opening them on local filesystem):
```
docker cp galaxy_halo_solutions:/home/docker_user/md2018/output <destination_local_filesystem> 
```

