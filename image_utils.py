import numpy as np

def uint16_to_float_image(image):
  """
  Converts a uint16 image to a normalized float image.

  Args:
    image: A numpy array representing the uint16 image.

  Returns:
    A numpy array representing the normalized float image.
  """

  # Check if the input is a uint16 image
  if not image.dtype == np.uint16:
    raise ValueError("Input image must be of type uint16.")

  # Normalize the image to the range [0, 1]
  max_value = np.iinfo(image.dtype).max  # Get the maximum value for uint16
  float_image = image.astype(np.float32) / max_value

  return float_image