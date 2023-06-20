
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11:latest

RUN pip install pip install tf-models-official==2.11.0 tensorflow-text==2.11.0

WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

RUN apt update && apt -y install nvtop

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]
