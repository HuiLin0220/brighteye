import json
import tempfile
from pathlib import Path
from pprint import pprint

import SimpleITK as sitk
from PIL import Image

DEFAULT_GLAUCOMATOUS_FEATURES = {
    "appearance neuroretinal rim superiorly": None,
    "appearance neuroretinal rim inferiorly": None,
    "retinal nerve fiber layer defect superiorly": None,
    "retinal nerve fiber layer defect inferiorly": None,
    "baring of the circumlinear vessel superiorly": None,
    "baring of the circumlinear vessel inferiorly": None,
    "nasalization of the vessel trunk": None,
    "disc hemorrhages": None,
    "laminar dots": None,
    "large cup": None,
}


def inference_tasks():
    input_files = [x for x in Path("./test/input/").rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)

    is_referable_glaucoma_stacked = []
    is_referable_glaucoma_likelihood_stacked = []
    glaucomatous_features_stacked = []

    def save_prediction(
            is_referable_glaucoma,
            likelihood_referable_glaucoma,
            glaucomatous_features=None,
    ):
        is_referable_glaucoma_stacked.append(is_referable_glaucoma)
        is_referable_glaucoma_likelihood_stacked.append(likelihood_referable_glaucoma)
        if glaucomatous_features is not None:
            glaucomatous_features_stacked.append({**DEFAULT_GLAUCOMATOUS_FEATURES, **glaucomatous_features})
        else:
            glaucomatous_features_stacked.append(DEFAULT_GLAUCOMATOUS_FEATURES)

    for file_path in input_files:
        if file_path.suffix == ".mha":  # A single image
            yield from single_file_inference(image_file=file_path, callback=save_prediction)
        elif file_path.suffix == ".tiff":  # A stack of images
            yield from stack_inference(stack=file_path, callback=save_prediction)

    write_referable_glaucoma_decision(is_referable_glaucoma_stacked)
    write_referable_glaucoma_decision_likelihood(
        is_referable_glaucoma_likelihood_stacked
    )
    write_glaucomatous_features(glaucomatous_features_stacked)


def single_file_inference(image_file, callback):
    with tempfile.TemporaryDirectory() as temp_dir:
        image = sitk.ReadImage(image_file)

        # Define the output file path
        output_path = Path(temp_dir) / "image.jpg"

        # Save the 2D slice as a JPG file
        sitk.WriteImage(image, str(output_path))

        # Call back that saves the result
        def save_prediction(
            is_referable_glaucoma,
            likelihood_referable_glaucoma,
            glaucomatous_features=None,
        ):
            glaucomatous_features = (
                glaucomatous_features or DEFAULT_GLAUCOMATOUS_FEATURES
            )
            write_referable_glaucoma_decision([is_referable_glaucoma])
            write_referable_glaucoma_decision_likelihood(
                [likelihood_referable_glaucoma]
            )
            write_glaucomatous_features(
                [{**DEFAULT_GLAUCOMATOUS_FEATURES, **glaucomatous_features}]
            )

        yield output_path, callback


def stack_inference(stack, callback):
    de_stacked_images = []

    # Unpack the stack
    with tempfile.TemporaryDirectory() as temp_dir:
        with Image.open(stack) as tiff_image:

            # Iterate through all pages
            for page_num in range(tiff_image.n_frames):
                # Select the current page
                tiff_image.seek(page_num)

                # Define the output file path
                #Path(temp_dir)
                output_path =  './data/'+ f"image_{page_num + 1}.jpg"
                tiff_image.save(output_path, "JPEG")

                de_stacked_images.append(output_path)

                print(f"De-Stacked {output_path}")

        # Loop over the images, and generate the actual tasks
        for index, image in enumerate(de_stacked_images):
            # Call back that saves the result
            yield image, callback


def write_referable_glaucoma_decision(result):
    print(result)
    with open(f"./test/output/multiple-referable-glaucoma-binary.json", "w") as f:
        f.write(json.dumps(result))


def write_referable_glaucoma_decision_likelihood(result):
    with open(f"./test/output/multiple-referable-glaucoma-likelihoods.json", "w") as f:
        f.write(json.dumps(result))


def write_glaucomatous_features(result):
    with open(f"./test/output/stacked-referable-glaucomatous-features.json", "w") as f:
        f.write(json.dumps(result))
