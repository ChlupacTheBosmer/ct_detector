from IPython.display import Image, display

def display_file_image(image_path, width=None, height=None):
    """
    A callback function to display an image in a Jupyter notebook.
    Use as the on_plot argument of the plot method of a <class 'ultralytics.utils.metrics.ConfusionMatrix'> instance.
    Display the saved image in the Jupyter notebook with custom size.
    Args:
        image_path (str): Path to the image file
        width (int, optional): Width of the displayed image in pixels
        height (int, optional): Height of the displayed image in pixels
    """
    display(Image(filename=str(image_path), width=width, height=height))


def display_pil_image(image, width=None, height=None):
    """
    A callback function to display a PIL image in a Jupyter notebook.
    Display the passed image in the Jupyter notebook with custom size.

    Args:
        image (PIL.Image): The PIL image to be displayed
        width (int, optional): Width of the displayed image in pixels
        height (int, optional): Height of the displayed image in pixels
    """
    display(image.resize((width, height)) if width and height else image)