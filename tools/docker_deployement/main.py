import glob
import os
from pathlib import Path

import docker
import pickle




def main():
    # Path to the docker files.
    docker_file_path = Path() / "docker_image_files"
    # The name under which the docker image is stored.
    image_tag = "j0rd1smit/detectron2"
    # the path to the image files
    images_path = Path() / "images"
    volumes = [f"{images_path.absolute()}:/app/images"]
    # Activate docker.
    client = docker.from_env()

    # Check if the docker image exist if not build it.
    image = client.images.build(
        path=str(docker_file_path.absolute()),
        rm=True,
        tag=image_tag,
        quiet=False,
    )

    # Run the container to make the predictions
    client.containers.run(
        image_tag,
        volumes=volumes,
    )

    # fetch the prediction for disk.
    data = []
    for path in glob.glob("images/*.p"):
        # read the binary results from disk.
        with open(path, "rb") as f:
            prediction = pickle.load(f)
            data.append(prediction)

        # clean up the binary results from disk
        os.remove(path)

    # Output the predictions or do something else with it.
    for prediction in data:
        print(prediction["file_name"], prediction["num_instances"], prediction["scores"])



if __name__ == '__main__':
    main()
