import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_piper_example() -> dict:
    """Creates a random input example for the Piper policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


_PIPER_MILLIDEG_PER_RAD = 57295.780490


def _maybe_convert_piper_state_units(state: np.ndarray) -> np.ndarray:
    """Heuristically convert Piper joint state units.

    Some Piper stacks/logged datasets store joint positions in a milli-degree-scaled unit
    (~rad * 57295.78). Others use radians. Our checkpoints' norm stats may expect either.

    We only ever convert the first 6 dims (joints). The last dim (gripper) is left as-is
    because datasets vary widely in gripper representation.
    """
    state = np.asarray(state)
    if state.shape[-1] < 6:
        return state

    joints = state[..., :6]
    # If joints look like radians (|q| < ~50), convert to milli-degree-scaled units.
    # We match the same threshold used in PiperOutputs for action unit detection.
    if np.max(np.abs(joints)) <= 50.0:
        state = state.astype(np.float64, copy=False)
        state[..., :6] = joints * _PIPER_MILLIDEG_PER_RAD
    return state


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        # Piper LeRobot configs repack raw dataset keys (e.g. "observation.images.wrist") into
        # standardized keys (e.g. "observation/image"). This transform should consume the
        # standardized keys.
        # NOTE: avoid printing in the hot path; use logging outside if needed.
        base_image = _parse_image(data["observation/image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": _maybe_convert_piper_state_units(data["observation/state"]),
            # "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]
        elif "action" in data:
            inputs["actions"] = data["action"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        # actions = np.asarray(data["actions"][:, :7])

        # # Piper hardware stack commonly represents joints in milli-degree-scaled units
        # # (roughly multiplied by ~57295.78). If we detect that, convert back to radians.
        # joints = actions[:, :6]
        # if np.max(np.abs(joints)) > 50.0:
        #     actions = actions.astype(np.float64, copy=False)
        #     actions[:, :6] = joints / 57295.780490

        # # Keep gripper within the expected physical range.
        # actions[:, 6] = np.clip(actions[:, 6], 0.0, 0.08)
        # return {"actions": actions}

        return {"actions": np.asarray(data["actions"][:, :7])}
