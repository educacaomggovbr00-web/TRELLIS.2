# TRELLIS.2 image -> GLB workflow

This fork contains a mobile-friendly GitHub Actions workflow that sends an image
to the public Microsoft TRELLIS.2 Hugging Face Space and saves the returned GLB
as a GitHub Actions artifact.

The heavy inference does **not** run on the normal GitHub runner. TRELLIS.2
requires a large NVIDIA GPU, so the workflow uses the public Space as the remote
GPU service.

## Included Naruto test input

The current test image is:

`workflow_inputs/naruto_reference.jpg`

It is a tight crop made from the reference image supplied in the ChatGPT
conversation so the foreground character occupies most of the frame.

## Run it from a phone

Open the repository on GitHub, go to **Actions**, choose
**Generate GLB with TRELLIS.2**, tap **Run workflow**, and leave the defaults.

When the run finishes, open the run and download the
`naruto-trellis-glb` artifact. It contains:

- `naruto_trellis.glb`
- `space_api.json` for debugging API changes

The first push that adds the workflow is configured to trigger one automatic
test run.

## Hugging Face token

The public Space may work anonymously, but ZeroGPU can impose queues or quota
limits. If needed, add a repository Actions secret named `HF_TOKEN` with a
Hugging Face token. The script uses it only when present.

The client keeps one Gradio session alive and uses the public
`preprocess_image`, `image_to_3d`, and `extract_glb` endpoints.
