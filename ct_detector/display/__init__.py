from ct_detector.display.plots import display_file_image, display_pil_image
import cv2

COLOR_CONVERSIONS = {
                "BGR2BGRA": cv2.COLOR_BGR2BGRA,
                "BGRA2BGR": cv2.COLOR_BGRA2BGR,
                "BGR2RGBA": cv2.COLOR_BGR2RGBA,
                "RGBA2BGR": cv2.COLOR_RGBA2BGR,
                "BGRA2RGBA": cv2.COLOR_BGRA2RGBA,
                "RGBA2BGRA": cv2.COLOR_RGBA2BGRA,

                "BGR2RGB": cv2.COLOR_BGR2RGB,
                "RGB2BGR": cv2.COLOR_RGB2BGR,
                "BGR2GRAY": cv2.COLOR_BGR2GRAY,
                "RGB2GRAY": cv2.COLOR_RGB2GRAY,
                "GRAY2BGR": cv2.COLOR_GRAY2BGR,
                "GRAY2RGB": cv2.COLOR_GRAY2RGB,
                "GRAY2BGRA": cv2.COLOR_GRAY2BGRA,
                "GRAY2RGBA": cv2.COLOR_GRAY2RGBA,
                "BGRA2GRAY": cv2.COLOR_BGRA2GRAY,
                "RGBA2GRAY": cv2.COLOR_RGBA2GRAY,

                "BGR2HSV": cv2.COLOR_BGR2HSV,
                "RGB2HSV": cv2.COLOR_RGB2HSV,
                "HSV2BGR": cv2.COLOR_HSV2BGR,
                "HSV2RGB": cv2.COLOR_HSV2RGB,

                "BGR2HLS": cv2.COLOR_BGR2HLS,
                "RGB2HLS": cv2.COLOR_RGB2HLS,
                "HLS2BGR": cv2.COLOR_HLS2BGR,
                "HLS2RGB": cv2.COLOR_HLS2RGB,

                "BGR2Lab": cv2.COLOR_BGR2Lab,
                "RGB2Lab": cv2.COLOR_RGB2Lab,
                "Lab2BGR": cv2.COLOR_Lab2BGR,
                "Lab2RGB": cv2.COLOR_Lab2RGB,

                "BGR2LUV": cv2.COLOR_BGR2LUV,
                "RGB2LUV": cv2.COLOR_RGB2LUV,
                "LUV2BGR": cv2.COLOR_LUV2BGR,
                "LUV2RGB": cv2.COLOR_LUV2RGB,

                "BGR2YCrCb": cv2.COLOR_BGR2YCrCb,
                "RGB2YCrCb": cv2.COLOR_RGB2YCrCb,
                "YCrCb2BGR": cv2.COLOR_YCrCb2BGR,
                "YCrCb2RGB": cv2.COLOR_YCrCb2RGB,

                "BGR2XYZ": cv2.COLOR_BGR2XYZ,
                "RGB2XYZ": cv2.COLOR_RGB2XYZ,
                "XYZ2BGR": cv2.COLOR_XYZ2BGR,
                "XYZ2RGB": cv2.COLOR_XYZ2RGB,

                "BGR2YUV": cv2.COLOR_BGR2YUV,
                "RGB2YUV": cv2.COLOR_RGB2YUV,
                "YUV2BGR": cv2.COLOR_YUV2BGR,
                "YUV2RGB": cv2.COLOR_YUV2RGB,

                "NONE": None  # special case to skip conversion
            }

DEFAULT_CONVERSION = "NONE"

__all__ = [
    "display_file_image",
    "display_pil_image",
    "COLOR_CONVERSIONS",
    "DEFAULT_CONVERSION"
]