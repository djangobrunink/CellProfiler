import glob
import json
from pathlib import Path

def main():
    for old_annotations in glob.glob("./**/*.json"):
        old_annotations = Path(old_annotations)
        assert "project" not in old_annotations.name
        old_annotations.unlink()



    with open(Path("./project.json")) as f:
        data = json.load(f)

    annotations = data["_via_img_metadata"]
    annotations2 = dict()
    for v in annotations.values():
        k = v["filename"]
        assert k not in annotations2
        annotations2[k] = v

    annotations = annotations2

    for image_path in glob.glob("./**/*.jpg"):
        image_path = Path(image_path)
        assert image_path.name in annotations, f"name = {image_path.name} {image_path.absolute()}"

        meta_data = annotations[image_path.name]

        stem = image_path.stem
        folder = image_path.parents[0]
        output = folder / f"{stem}.json"

        with open(output, "w") as f:
            json.dump(meta_data, f, indent=1)




if __name__ == '__main__':
    main()