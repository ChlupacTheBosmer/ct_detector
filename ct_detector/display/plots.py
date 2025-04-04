from IPython.display import Image, display

def display_confusion_matrix(plot_fname, width=None, height=None):
    """
    A callback function to display the confusion matrix plot in a Jupyter notebook.
    Use as the on_plot argument of the plot method of a <class 'ultralytics.utils.metrics.ConfusionMatrix'> instance.
    Display the saved image in the Jupyter notebook with custom size.

    Args:
        plot_fname (str or Path): Path to the image file
        width (int, optional): Width of the displayed image in pixels
        height (int, optional): Height of the displayed image in pixels
    """
    display(Image(filename=str(plot_fname), width=width, height=height))