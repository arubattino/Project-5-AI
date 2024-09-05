# Final-project-5 - Alexander Rubattino

Details and conclusions of the project are detailed in the file "conclusions.ipynb".

In the folder "api-final-project-main" is the complete work of the team. Front and tests is the merit of José Luis Cabrera.

- **CPU:**

```bash
$ docker build -t final_project_ar -f docker/Dockerfile .
```

- **GPU:**

```bash
$ docker build -t final_project_ar -f docker/Dockerfile_gpu .        ## Only once on the server.
```

### Run Docker

```bash
$ docker run --rm --net host --gpus all -it \       ## Remove --gpus all to run local.
    -v $(pwd):/home/app/src \
    --workdir /home/app/src \
    final_project_ar \
    bash
```
