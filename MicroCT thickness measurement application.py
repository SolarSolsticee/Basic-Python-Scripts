import numpy as np
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.morphology import disk
from skimage import io, color, exposure
from skimage.feature import canny
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import PySimpleGUI as sg
import pandas as pd

# Configure logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_synthetic_2d_object(shape=(200, 200), object_type='ring'):
    """
    Creates a synthetic 2D binary image to simulate a segmented object in a single slice.
    This function is a placeholder for your actual 2D image loading and segmentation.

    Args:
        shape (tuple): The (height, width) dimensions of the 2D array.
        object_type (str): Type of object to create ('circle', 'square', 'ring').

    Returns:
        np.array: A 2D binary numpy array (True for object, False for background).
    """
    logging.info(f"Creating synthetic 2d object of type: {object_type} with shape: {shape}")
    img = np.zeros(shape, dtype=bool)
    center_y, center_x = shape[0] // 2, shape[1] // 2

    if object_type == 'circle':
        radius = min(shape) // 4
        for y in range(shape[0]):
            for x in range(shape[1]):
                if (x - center_x) ** 2 + (y - center_y) ** 2 < radius ** 2:
                    img[y, x] = True
    elif object_type == 'square':
        side = min(shape) // 3
        img[center_y - side // 2: center_y + side // 2,
        center_x - side // 2: center_x + side // 2] = True
    elif object_type == 'ring':
        outer_radius = min(shape) // 3
        inner_radius = min(shape) // 5
        for y in range(shape[0]):
            for x in range(shape[1]):
                dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
                if inner_radius ** 2 < dist_sq < outer_radius ** 2:
                    img[y, x] = True
    else:
        logging.warning(f"Unknown object type: {object_type}. Creating a ring by default.")
        return create_synthetic_2d_object(shape, 'ring')

    logging.info("Synthetic 2d object created.")
    return img


def measure_thickness_edt(binary_image):
    """
    Measures the local thickness (width/diameter) of a 2D binary image
    using the Euclidean Distance Transform (EDT).
    For each foreground pixel, its thickness is estimated as twice the
    distance to the nearest background pixel. This represents the diameter
    of the largest inscribed circle at that pixel.

    Args:
        binary_image (np.array): A 2D binary numpy array (True for object, False for background).

    Returns:
        np.array: A 2D numpy array where each foreground pixel's value represents its
                  estimated local thickness. Background pixels will have a value of 0.
    """
    if not isinstance(binary_image, np.ndarray) or binary_image.ndim != 2:
        logging.error("Input must be a 2D numpy array.")
        raise ValueError("Input must be a 2D numpy array.")

    logging.info("Starting thickness measurement using Euclidean Distance Transform for 2D image.")

    # Calculate the Euclidean Distance Transform (EDT) for foreground pixels
    # This gives the distance from each foreground pixel to the nearest background pixel.
    distance_map = distance_transform_edt(binary_image)

    # The local thickness in 2D is approximately twice the distance to the nearest background.
    # This represents the diameter of the largest inscribed circle.
    thickness_map = 2 * distance_map

    logging.info("Thickness measurement completed.")
    return thickness_map


def visualize_thickness_2d(thickness_map, original_image_for_overlay=None, unit_label="pixels",
                           roi_data=None, distance_per_pixel_for_drawing=1.0, plot_basename="2D Thickness Map",
                           original_filename="", canny_params=None, exposure_params=None, save_path=None):
    """
    Visualizes the 2D thickness map. Optionally overlays the original image and ROI boxes.
    Saves the plot to save_path if provided, otherwise displays it.

    Args:
        thickness_map (np.array): The 2D array of thickness values.
        original_image_for_overlay (np.array, optional): The original grayscale image
                                                        to overlay for context.
        unit_label (str): The label for the units (e.g., "pixels", "mm").
        roi_data (list, optional): A list of dictionaries, each containing ROI coords, stats, and custom label.
                                   Expected to contain 'value', 'std', 'max_t_start_plot_xy', 'max_t_end_plot_xy', 'metric_type'.
        distance_per_pixel_for_drawing (float): The pixels/mm value used to convert max_t (in mm) back to pixels for drawing.
        plot_basename (str): The user-defined basename for the plot title.
        original_filename (str): The filename of the original image.
        canny_params (dict, optional): Dictionary containing 'sigma', 'low_threshold', 'high_threshold'.
        exposure_params (dict, optional): Dictionary containing 'method' and 'gamma' (if method is gamma).
        save_path (str, optional): Full path to save the plot. If None, the plot is displayed.
    """
    if thickness_map.ndim != 2:
        logging.error("Thickness map must be 2D for visualization.")
        return

    logging.info(f"Visualizing 2D thickness map in {unit_label}.")

    # Set figure size for 2000x1000 resolution at 100 dpi
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Apply vertical flip to both images for correct orientation
    flipped_thickness_map = np.flipud(thickness_map)

    if original_image_for_overlay is not None and original_image_for_overlay.shape == thickness_map.shape:
        flipped_original_image = np.flipud(original_image_for_overlay)
        # Overlay original image as background for context
        ax.imshow(flipped_original_image, cmap='gray', alpha=0.5, origin='lower')
        im = ax.imshow(flipped_thickness_map, cmap='viridis', alpha=0.7, origin='lower',
                       vmin=0, vmax=np.max(flipped_thickness_map[flipped_thickness_map > 0]) * 1.1)
    else:
        im = ax.imshow(flipped_thickness_map, cmap='viridis', origin='lower')

    # Draw ROI boxes and max distance lines/text if provided
    if roi_data:
        height_img = thickness_map.shape[0]  # Get original image height for y-flip
        for i, roi_info in enumerate(roi_data):
            x_min, y_min, width, height = roi_info['coords']
            metric_value = roi_info['value']  # This will be max_t or min_t
            std_t = roi_info['std']
            custom_label = roi_info.get('custom_label', '')
            metric_type = roi_info.get('metric_type', 'MAX')  # 'MAX' or 'MIN'

            # Retrieve the specific start and end points for the max thickness line
            max_t_start_plot_xy = roi_info.get('max_t_start_plot_xy')
            max_t_end_plot_xy = roi_info.get('max_t_end_plot_xy')
            # For MIN, we need min_t_start_plot_xy, min_t_end_plot_xy
            min_t_start_plot_xy = roi_info.get('min_t_start_plot_xy')
            min_t_end_plot_xy = roi_info.get('min_t_end_plot_xy')

            # Adjust y_min for origin='lower' plotting
            rect_y_min_flipped = height_img - (y_min + height)

            # Draw ROI box
            rect = patches.Rectangle((x_min, rect_y_min_flipped), width, height,
                                     linewidth=1, edgecolor='red', facecolor='red', alpha=0.2, linestyle='-')
            ax.add_patch(rect)

            # Add ROI number label to the top-left corner
            text_x_roi_num = x_min + 5
            text_y_roi_num = rect_y_min_flipped + height - 5
            ax.text(text_x_roi_num, text_y_roi_num, str(i + 1), color='white', fontsize=8, ha='left', va='top',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

            # Draw the appropriate line (MAX or MIN)
            line_color = 'yellow' if metric_type == 'MAX' else 'cyan'
            line_style = ':'
            line_width = 1.5
            arrow_style = '<->'

            if metric_type == 'MAX' and max_t_start_plot_xy and max_t_end_plot_xy:
                ax.annotate('', xy=max_t_end_plot_xy, xytext=max_t_start_plot_xy,
                            arrowprops=dict(arrowstyle=arrow_style, color=line_color, lw=line_width, ls=line_style))
            elif metric_type == 'MIN' and min_t_start_plot_xy and min_t_end_plot_xy:
                ax.annotate('', xy=min_t_end_plot_xy, xytext=min_t_start_plot_xy,
                            arrowprops=dict(arrowstyle=arrow_style, color=line_color, lw=line_width, ls=line_style))
            else:
                logging.warning(
                    f"No specific {metric_type} thickness coordinates found for ROI {i + 1}. Skipping arrow drawing.")

            # Write custom label with smart positioning
            if custom_label:
                # Adjust label positioning based on ROI aspect ratio, not arrow direction
                if width > height:  # Horizontal ROI, horizontal arrow, label above ROI
                    label_x = x_min + width / 2
                    label_y = rect_y_min_flipped + height + 5
                    ax.text(label_x, label_y, custom_label,
                            color='blue', fontsize=8, ha='center', va='bottom',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))
                else:  # Vertical ROI, vertical arrow, label to the right of ROI
                    label_x = x_min + width + 5
                    label_y = rect_y_min_flipped + height / 2
                    ax.text(label_x, label_y, custom_label,
                            color='blue', fontsize=8, ha='left', va='center',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))

            # Write Metric Value and StdDev values
            text_x_stats = x_min + width / 2
            text_y_value = rect_y_min_flipped - 10  # Offset below the box
            text_y_stddev = rect_y_min_flipped - 25  # Further offset for StdDev

            if not np.isnan(metric_value):
                ax.text(text_x_stats, text_y_value, f'{metric_type}: {metric_value:.2f}{unit_label}',
                        color='black', fontsize=7, ha='center', va='top',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))
            if not np.isnan(std_t):
                ax.text(text_x_stats, text_y_stddev, f'Std: {std_t:.2f}{unit_label}',
                        color='black', fontsize=7, ha='center', va='top',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))

    # --- Remove the color bar ---
    # cbar = fig.colorbar(im, ax=ax, label=f'Local Thickness ({unit_label})') # Removed this line

    # Set X and Y axis labels to mm
    img_height, img_width = thickness_map.shape

    # For X-axis
    x_ticks_px = np.linspace(0, img_width - 1, num=5, dtype=int)  # Example: 5 ticks
    x_tick_labels_mm = [f"{p / distance_per_pixel_for_drawing:.1f}" for p in x_ticks_px]
    ax.set_xticks(x_ticks_px)
    ax.set_xticklabels(x_tick_labels_mm)
    ax.set_xlabel(f'X-axis ({unit_label})')

    # For Y-axis
    y_ticks_px = np.linspace(0, img_height - 1, num=5, dtype=int)  # Example: 5 ticks
    y_tick_labels_mm = [f"{p / distance_per_pixel_for_drawing:.1f}" for p in y_ticks_px]
    ax.set_yticks(y_ticks_px)
    ax.set_yticklabels(y_tick_labels_mm)
    ax.set_ylabel(f'Y-axis ({unit_label})')

    # Set the plot title with basename and filename
    title_text = f"{plot_basename}"
    if original_filename:
        title_text += f" ({os.path.basename(original_filename)})"
    title_text += f" Radiography thickness dist. analysis"
    ax.set_title(title_text)

    # Add Canny parameters box to top-right corner
    if canny_params:
        param_text = (f"Canny Parameters:\n"
                      f"  Sigma: {canny_params['sigma']:.2f}\n"
                      f"  Low Thresh: {canny_params['low_threshold']:.2f}\n"
                      f"  High Thresh: {canny_params['high_threshold']:.2f}")

        ax.text(0.98, 0.98, param_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7, ec='gray'))

    # Add Exposure parameters box (top-left for now, adjust as needed)
    if exposure_params and exposure_params['method'] != 'None':
        exposure_text = f"Exposure: {exposure_params['method']}"
        if exposure_params['method'] == 'Gamma':
            exposure_text += f" (Gamma: {exposure_params['gamma']:.2f})"
        elif exposure_params['method'] == 'CLAHE':
            exposure_text += f" (Clip Limit: {exposure_params.get('clip_limit', 'Default'):.2f})"

        ax.text(0.02, 0.98, exposure_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7, ec='gray'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        logging.info(f"Plot saved to {save_path}")
    else:
        plt.show()
        logging.info("2D thickness map visualization complete.")


def adjust_image_exposure(image_np, method='None', gamma=1.0, clip_limit=0.01):
    """
    Adjusts the exposure (brightness/contrast) of a grayscale image.

    Args:
        image_np (np.array): The input grayscale image (float type, range 0-1 preferred).
        method (str): Exposure adjustment method ('None', 'Histogram Equalization', 'Gamma', 'CLAHE').
        gamma (float): Gamma value for gamma correction.
        clip_limit (float): Clip limit for CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Returns:
        np.array: The adjusted grayscale image.
    """
    logging.info(f"Applying exposure adjustment method: {method}")

    if method == 'Histogram Equalization':
        # Ensure image is float between 0 and 1 for skimage.exposure functions
        if image_np.dtype != np.float64:
            image_np = image_np.astype(np.float64) / image_np.max()
        return exposure.equalize_hist(image_np)
    elif method == 'Gamma':
        if image_np.dtype != np.float64:
            image_np = image_np.astype(np.float64) / image_np.max()
        return exposure.adjust_gamma(image_np, gamma)
    elif method == 'CLAHE':
        if image_np.dtype != np.float64:
            image_np = image_np.astype(np.float64) / image_np.max()
        return exposure.equalize_adapthist(image_np, clip_limit=clip_limit)
    elif method == 'None':
        return image_np
    else:
        logging.warning(f"Unknown exposure method: {method}. No adjustment applied.")
        return image_np


def extract_rois_from_image(roi_image_path, target_shape):
    """
    Loads an image, detects red boxes, and extracts their bounding box coordinates.

    Args:
        roi_image_path (str): Path to the image file containing the red boxes.
        target_shape (tuple): (height, width) of the main image, for scaling/checking.

    Returns:
        list: A list of (x_min, y_min, width, height) tuples for each detected ROI.
              Returns an empty list if no ROIs are found or on error.
    """
    logging.info(f"Extracting ROIs from image: {roi_image_path}")
    try:
        roi_img = io.imread(roi_image_path)

        if roi_img.ndim == 3 and roi_img.shape[2] == 4:
            rgb_img = roi_img[:, :, :3]
            alpha_channel = roi_img[:, :, 3]
        elif roi_img.ndim == 3 and roi_img.shape[2] == 3:
            rgb_img = roi_img
            alpha_channel = None
        else:
            sg.popup_ok("ROI Image Warning", "ROI image is not RGB or RGBA. Cannot detect red boxes.")
            logging.warning("ROI image is not RGB/RGBA. Cannot detect red boxes.")
            return []

        if rgb_img.dtype == np.uint8:
            rgb_img = rgb_img / 255.0

        red_threshold_low = 0.7
        green_threshold_high = 0.3
        blue_threshold_high = 0.3

        red_mask = (rgb_img[:, :, 0] > red_threshold_low) & \
                   (rgb_img[:, :, 1] < green_threshold_high) & \
                   (rgb_img[:, :, 2] < blue_threshold_high)

        if not np.any(red_mask):
            sg.popup_ok("ROI Detection Failed",
                        "No red pixels detected in the ROI image. Please check the image content and color thresholds.")
            logging.warning("No red pixels detected in ROI image.")
            return []

        labeled_rois = label(red_mask)
        props = regionprops(labeled_rois)

        if not props:
            sg.popup_ok("ROI Detection Failed", "No distinct red regions (ROIs) found in the image.")
            logging.warning("No distinct red regions found in ROI image.")
            return []

        extracted_rois = []
        for prop in props:
            min_row, min_col, max_row, max_col = prop.bbox
            x_min = min_col
            y_min = min_row
            width = max_col - min_col
            height = max_row - min_row
            extracted_rois.append((x_min, y_min, width, height))

        extracted_rois.sort(key=lambda r: (r[1], r[0]))

        logging.info(f"Found {len(extracted_rois)} ROIs in the image.")
        return extracted_rois

    except Exception as e:
        sg.popup_error("ROI Image Error", f"Failed to load or process ROI image: {e}")
        logging.error(f"Error extracting ROIs from image: {e}", exc_info=True)
        return []


# Helper function to find contiguous segments and their positions
def find_contiguous_segments_with_positions(bool_array):
    """
    Finds lengths and start/end positions of contiguous True segments in a 1D boolean array.
    Returns: list of (length, (start_idx, end_idx)) tuples for each segment.
    start_idx and end_idx are inclusive and relative to the input bool_array.
    """
    if not np.any(bool_array):
        return []

    # Pad with False at ends to easily detect starts/ends of segments
    padded_array = np.pad(bool_array, (1, 1), 'constant', constant_values=False)

    starts = np.where(np.diff(padded_array.astype(int)) == 1)[0]
    ends = np.where(np.diff(padded_array.astype(int)) == -1)[0]

    segments = []
    for i in range(len(starts)):
        length = ends[i] - starts[i]
        start_idx = starts[i] - 1  # Adjust for the leading padding
        end_idx = start_idx + length - 1  # Adjust for 0-indexing to get inclusive end
        segments.append((length, (start_idx, end_idx)))
    return segments


def main():
    """
    Main function to run the 2D image thickness measurement application.
    """
    logging.info("Starting 2D Image Thickness Measurement Application.")

    # Initialize variables that might be used later to prevent UnboundLocalError
    image_np_original = None
    image_path = None  # Will be set in the loop
    distance_per_pixel = None

    default_distance_per_pixel = 28.089
    default_canny_sigma = 1.0
    default_canny_low_threshold = 0.1
    default_canny_high_threshold = 0.2
    default_exposure_method = 'None'
    default_gamma_value = 1.0
    default_clahe_clip_limit = 0.01

    canny_params = {}
    exposure_params = {}
    global_rois = []  # Initialized here
    global_custom_labels = []  # Initialized here
    roi_metric_choices = {}  # Initialized here

    V1_TRANSPARENT_LABELS = [
        'T4', 'T11', 'T3', 'T4.1', 'T10', 'T12', 'T2', 'T8', 'T1',
        'T12.1', 'T9', 'T4.3', 'T12.2', 'T4.4', 'T12.3', 'T12.4', 'T13', 'T5'
    ]

    root = tk.Tk()
    root.withdraw()

    sg.theme('LightBlue3')

    layout = [
        [sg.Text('Initial Setup Parameters', font=('Helvetica', 14, 'bold'))],
        [sg.Frame('Plot and Calibration', [
            [sg.Text('Plot Basename:'), sg.InputText(default_text='Untitled', key='-PLOT_BASENAME-')],
            [sg.Text('Pixels per mm:'),
             sg.InputText(default_text=str(default_distance_per_pixel), key='-PIXELS_PER_MM-')],
        ])],
        [sg.Frame('Canny Edge Detection Parameters', [
            [sg.Text('Sigma:'), sg.InputText(default_text=str(default_canny_sigma), key='-CANNY_SIGMA-')],
            [sg.Text('Low Threshold:'),
             sg.InputText(default_text=str(default_canny_low_threshold), key='-CANNY_LOW_THRESH-')],
            [sg.Text('High Threshold:'),
             sg.InputText(default_text=str(default_canny_high_threshold), key='-CANNY_HIGH_THRESH-')],
        ])],
        [sg.Frame('Image Exposure Enhancement', [
            [sg.Text('Method:'), sg.Combo(['None', 'Histogram Equalization', 'Gamma', 'CLAHE'],
                                          default_value=default_exposure_method, key='-EXPOSURE_METHOD-',
                                          enable_events=True)],
            [sg.Text('Gamma Value:'), sg.InputText(default_text=str(default_gamma_value), key='-GAMMA_VALUE-')],
            [sg.Text('CLAHE Clip Limit:'),
             sg.InputText(default_text=str(default_clahe_clip_limit), key='-CLAHE_CLIP_LIMIT-')],
            [sg.Text('Note: Gamma is for "Gamma" method, Clip Limit for "CLAHE".')],
        ])],
        [sg.Button('Start Processing', size=(15, 1)), sg.Button('Cancel', size=(15, 1))]
    ]

    window = sg.Window('MicroCT Thickness App Setup', layout, finalize=True)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel' or event == 'Start Processing':
            break

        if event == '-EXPOSURE_METHOD-':
            method = values['-EXPOSURE_METHOD-']
            window['-GAMMA_VALUE-'].update(disabled=(method != 'Gamma'))
            window['-CLAHE_CLIP_LIMIT-'].update(disabled=(method != 'CLAHE'))

    window.close()

    if event == sg.WIN_CLOSED or event == 'Cancel':
        sg.popup_ok("Setup Cancelled", "Application setup cancelled. Exiting.")
        root.destroy()
        return

    plot_basename = values['-PLOT_BASENAME-'] if values['-PLOT_BASENAME-'] else "Untitled"

    try:
        distance_per_pixel = float(values['-PIXELS_PER_MM-'])
    except ValueError:
        sg.popup_ok("Input Error",
                    f"Invalid Pixels per mm '{values['-PIXELS_PER_MM-']}'. Using default: {default_distance_per_pixel}")
        distance_per_pixel = default_distance_per_pixel

    try:
        canny_sigma = float(values['-CANNY_SIGMA-'])
    except ValueError:
        sg.popup_ok("Input Error",
                    f"Invalid Canny Sigma '{values['-CANNY_SIGMA-']}'. Using default: {default_canny_sigma}")
        canny_sigma = default_canny_sigma

    try:
        canny_low_threshold = float(values['-CANNY_LOW_THRESH-'])
    except ValueError:
        sg.popup_ok("Input Error",
                    f"Invalid Canny Low Threshold '{values['-CANNY_LOW_THRESH-']}'. Using default: {default_canny_low_threshold}")
        canny_low_threshold = default_canny_low_threshold

    try:
        canny_high_threshold = float(values['-CANNY_HIGH_THRESH-'])
    except ValueError:
        sg.popup_ok("Input Error",
                    f"Invalid Canny High Threshold '{values['-CANNY_HIGH_THRESH-']}'. Using default: {default_canny_high_threshold}")
        canny_high_threshold = default_canny_high_threshold

    exposure_params['method'] = values['-EXPOSURE_METHOD-']
    try:
        exposure_params['gamma'] = float(values['-GAMMA_VALUE-'])
    except ValueError:
        sg.popup_ok("Input Error",
                    f"Invalid Gamma Value '{values['-GAMMA_VALUE-']}'. Using default: {default_gamma_value}")
        exposure_params['gamma'] = default_gamma_value
    try:
        exposure_params['clip_limit'] = float(values['-CLAHE_CLIP_LIMIT-'])
    except ValueError:
        sg.popup_ok("Input Error",
                    f"Invalid CLAHE Clip Limit '{values['-CLAHE_CLIP_LIMIT-']}'. Using default: {default_clahe_clip_limit}")
        exposure_params['clip_limit'] = default_clahe_clip_limit

    unit_label = "mm" if distance_per_pixel != 1.0 else "pixels"
    logging.info(f"Plot Basename: {plot_basename}")
    logging.info(f"Distance per pixel: {distance_per_pixel} {unit_label}")
    canny_params = {
        'sigma': canny_sigma,
        'low_threshold': canny_low_threshold,
        'high_threshold': canny_high_threshold
    }
    logging.info(f"Final Canny parameters for processing: {canny_params}")
    logging.info(f"Final Exposure parameters for processing: {exposure_params}")

    try:
        image_paths = filedialog.askopenfilenames(
            title="Select .bmp image file(s) for analysis",
            filetypes=[("BMP files", "*.bmp"), ("All files", "*.*")],
            parent=root
        )
    except Exception as e:
        sg.popup_error("File Selection Error", f"An error occurred during file selection: {e}")
        logging.error(f"Error during file dialog: {e}", exc_info=True)
        root.destroy()
        return

    if not image_paths:
        sg.popup_ok("No Images Selected", "No image files were selected. Exiting application.")
        root.destroy()
        return

    output_directory = filedialog.askdirectory(
        title="Select Output Directory for Processed Images and Data",
        parent=root
    )
    if not output_directory:
        sg.popup_ok("No Output Directory", "No output directory selected. Exiting application.")
        root.destroy()
        return

    processed_images_dir = os.path.join(output_directory, 'processed-images')
    os.makedirs(processed_images_dir, exist_ok=True)
    logging.info(f"Processed images will be saved to: {processed_images_dir}")

    root.destroy()

    # --- ROI Selection and Metric Choice Logic ---
    # These are initialized globally at the top of main, but populated here.
    # Ensure they are reset for each run if needed, but for a single run, global initialization is fine.

    first_image_path_for_roi_ref = image_paths[0]  # Use a distinct name for clarity
    first_binary_image_for_roi_ref = None  # This will be used only for shape for ROI detection

    try:
        temp_image_np = io.imread(first_image_path_for_roi_ref)
        if temp_image_np.ndim == 3:
            temp_image_np = color.rgb2gray(temp_image_np)

        temp_image_np_adjusted = adjust_image_exposure(temp_image_np.copy(), **exposure_params)

        # This binary image is used only to get the shape for extract_rois_from_image,
        # not for the actual thickness calculations within the main loop.
        first_binary_image_for_roi_ref = binary_fill_holes(canny(temp_image_np_adjusted, sigma=canny_params['sigma'],
                                                                 low_threshold=canny_params['low_threshold'],
                                                                 high_threshold=canny_params['high_threshold']))
        if not np.any(first_binary_image_for_roi_ref):
            sg.popup_ok("First Image Warning",
                        "No object detected in the first image for ROI reference. ROIs will be skipped for all images.")
            first_binary_image_for_roi_ref = None
    except Exception as e:
        sg.popup_error("First Image Load Error",
                       f"Failed to load first image for ROI reference: {e}\nROIs will be skipped for all images.")
        first_binary_image_for_roi_ref = None

    if first_binary_image_for_roi_ref is not None:
        roi_selection_layout = [
            [sg.Text('Select ROI Image', font=('Helvetica', 14, 'bold'))],
            [sg.Text(
                'Please select ONE transparent image (e.g., PNG) containing red boxes to define Regions of Interest.')],
            [sg.Input(key='-ROI_FILE_PATH-', enable_events=True, visible=True, disabled=True),
             sg.FileBrowse('Browse for ROI Image',
                           file_types=(("PNG Files", "*.png"), ("BMP Files", "*.bmp"), ("All Files", "*.*")))],
            [sg.Button('Confirm Selection', size=(18, 1)), sg.Button('Skip ROIs', size=(15, 1))]
        ]
        roi_selection_window = sg.Window('ROI Image Selection', roi_selection_layout, finalize=True)

        roi_image_path = None
        while True:
            roi_event, roi_values = roi_selection_window.read()
            if roi_event == sg.WIN_CLOSED or roi_event == 'Skip ROIs':
                sg.popup_ok("ROI Selection Skipped",
                            "No ROI image selected. Proceeding without specific ROI analysis for all images.")
                break
            elif roi_event == 'Confirm Selection':
                selected_path = roi_values['-ROI_FILE_PATH-']
                if selected_path:
                    roi_image_path = selected_path
                    break
                else:
                    sg.popup_ok("No File Selected", "Please select an ROI image or click 'Skip ROIs'.")
        roi_selection_window.close()

        if roi_image_path:
            global_rois = extract_rois_from_image(roi_image_path, first_binary_image_for_roi_ref.shape)
            if not global_rois:
                sg.popup_ok("ROI Warning",
                            f"No valid ROIs extracted from '{os.path.basename(roi_image_path)}'. Proceeding without specific ROI analysis for all images.")
            else:
                use_hardcoded = sg.popup_yes_no(
                    "Custom ROI Labels",
                    "Do you want to use the 'v1 transparent' hardcoded labels?\n"
                    "Click 'Yes' for hardcoded, 'No' to enter custom labels."
                ) == 'Yes'

                if use_hardcoded:
                    global_custom_labels = V1_TRANSPARENT_LABELS
                    if len(global_custom_labels) < len(global_rois):
                        logging.warning(
                            "Hardcoded labels are fewer than detected ROIs. Some ROIs will have default labels.")
                        global_custom_labels.extend(
                            [f"ROI_{i + 1}" for i in range(len(global_custom_labels), len(global_rois))])
                    global_custom_labels = global_custom_labels[:len(global_rois)]  # Ensure labels match ROI count

                    logging.info(f"Using 'v1 transparent' hardcoded labels: {global_custom_labels}")
                else:
                    # Changed variable name here: 'label_tag_input' instead of 'label_input'
                    label_tag_input = sg.popup_get_text(
                        "Custom ROI Labels",
                        f"Enter {len(global_rois)} custom labels for your ROIs, separated by commas.\n"
                        f"Example: Top, Left, Bottom, Right, ...",
                        default_text=""
                    )
                    if label_tag_input:  # Check 'label_tag_input'
                        global_custom_labels = [tag.strip() for tag in label_tag_input.split(',')]  # Use 'tag'
                        if len(global_custom_labels) != len(global_rois):
                            sg.popup_warning("Label Count Mismatch",
                                             f"You provided {len(global_custom_labels)} labels but {len(global_rois)} ROIs were detected. "
                                             "Some ROIs will use default numbering if labels don't match.")
                            if len(global_custom_labels) < len(global_rois):
                                global_custom_labels.extend(
                                    [f"ROI_{i + 1}" for i in range(len(global_custom_labels), len(global_rois))])
                            else:
                                global_custom_labels = global_custom_labels[:len(global_rois)]

                        logging.info(f"Custom labels entered: {global_custom_labels}")
                    else:
                        logging.info("No custom labels entered, using default ROI numbering.")
                        global_custom_labels = [f"ROI_{i + 1}" for i in range(len(global_rois))]

                roi_choice_layout = [
                    [sg.Text('Select Measurement Metric for Each ROI', font=('Helvetica', 14, 'bold'))],
                    [sg.Text(
                        'Choose whether to calculate MAX (full extent) or MIN (largest internal segment) thickness for each detected ROI.')],
                    [sg.Column([
                        # Changed 'label' to 'label_tag' in f-string
                        [sg.Text(f"ROI {idx + 1}: {label_tag}"),
                         sg.Radio('MAX', group_id=f'roi_metric_{idx}', key=f'-ROI_MAX_{idx}-',
                                  default=(idx + 1 != 8 and idx + 1 != 10)),
                         sg.Radio('MIN', group_id=f'roi_metric_{idx}', key=f'-ROI_MIN_{idx}-',
                                  default=(idx + 1 == 8 or idx + 1 == 10))]
                        for idx, label_tag in enumerate(global_custom_labels)  # Changed 'label' to 'label_tag'
                    ], scrollable=True, vertical_scroll_only=True, size=(400, 300))],
                    [sg.Button('Confirm Choices', size=(15, 1)), sg.Button('Cancel', size=(15, 1))]
                ]

                roi_choice_window = sg.Window('ROI Metric Selection', roi_choice_layout, finalize=True)

                # Default roi_metric_choices to MAX for all, in case of cancellation
                roi_metric_choices = {tag: 'MAX' for tag in global_custom_labels}  # Changed 'label' to 'tag'

                while True:
                    event_roi_choice, values_roi_choice = roi_choice_window.read()
                    if event_roi_choice == sg.WIN_CLOSED or event_roi_choice == 'Cancel':
                        sg.popup_ok("ROI Metric Selection Cancelled", "Proceeding with default MAX for all ROIs.")
                        break
                    if event_roi_choice == 'Confirm Choices':
                        for idx, label_tag in enumerate(global_custom_labels):  # Changed 'label' to 'label_tag'
                            if values_roi_choice[f'-ROI_MAX_{idx}-']:
                                roi_metric_choices[label_tag] = 'MAX'  # Changed 'label' to 'label_tag'
                            elif values_roi_choice[f'-ROI_MIN_{idx}-']:
                                roi_metric_choices[label_tag] = 'MIN'  # Changed 'label' to 'label_tag'
                        logging.info(f"ROI Metric Choices: {roi_metric_choices}")
                        break
                roi_choice_window.close()
        # No else needed here, global_rois and roi_metric_choices retain their (possibly empty) state.

    all_samples_excel_data = []
    all_unique_roi_labels = set()

    # Define arrow offset in pixels
    # This offset will be applied in the Matplotlib (Y-up) coordinate system.
    # Positive for X (moves right), Negative for Y (moves up, on plot)
    arrow_offset_pixels = 0  # Revert to 0 for now as the original working code did not have an explicit offset.

    for sample_index, image_path_current in enumerate(image_paths):  # Renamed to avoid shadowing
        logging.info(f"\n--- Processing Sample {sample_index + 1}: {os.path.basename(image_path_current)} ---")

        # Define specific binary images for MAX and MIN calculation paths
        image_for_max_calc = None
        image_for_min_calc = None
        current_image_np_original = None  # Renamed for clarity within loop
        thickness_map_for_viz = None  # Initialize here

        try:
            logging.info(f"Attempting to load main image from: {image_path_current}")
            image_np = io.imread(image_path_current)

            if image_np.ndim == 3:
                logging.info("Converting RGB image to grayscale.")
                image_np = color.rgb2gray(image_np)
            elif image_np.ndim != 2:
                sg.popup_error("Image Dimension Error",
                               f"Unsupported image dimensions: {image_np.ndim}. Expected 2D or 3D (RGB). Skipping {os.path.basename(image_path_current)}.")
                logging.error(
                    f"Unsupported image dimensions: {os.path.ndim}. Skipping {os.path.basename(image_path_current)}.")
                continue

            current_image_np_original = image_np  # Store original grayscale for overlay

            logging.info(f"Applying exposure enhancement: {exposure_params['method']}")
            image_np_enhanced = adjust_image_exposure(image_np.copy(),
                                                      method=exposure_params['method'],
                                                      gamma=exposure_params['gamma'],
                                                      clip_limit=exposure_params['clip_limit'])

            # --- Pre-calculate binary images for both MAX and MIN interpretations ---
            edges = canny(image_np_enhanced, sigma=canny_params['sigma'],
                          low_threshold=canny_params['low_threshold'],
                          high_threshold=canny_params['high_threshold'])

            # Binary image for MAX: Apply Canny then fill holes (object extent)
            image_for_max_calc = binary_fill_holes(edges).astype(bool)
            logging.info("Binary image (for MAX) generated by Canny and filling holes.")

            # Binary image for MIN: Apply Canny WITHOUT filling holes (to see individual lines/features)
            image_for_min_calc = edges.astype(bool)  # Directly use edges for MIN
            logging.info("Binary image (for MIN) generated by Canny edges (no hole filling).")

            # Determine the `thickness_map` that will be visualized (it's often based on the 'filled' image)
            thickness_map_for_viz = measure_thickness_edt(image_for_max_calc)


        except Exception as e:
            sg.popup_error("Main Image Loading/Processing Error",
                           f"Failed to load or process main image '{os.path.basename(image_path_current)}': {e}\nSkipping this image.")
            logging.error(f"Failed to load or process main image '{os.path.basename(image_path_current)}': {e}",
                          exc_info=True)
            continue

        try:
            current_sample_excel_row = {'Sample #': sample_index + 1}  # Initialize dictionary for current row

            if global_rois:  # This block only runs if ROIs were successfully loaded/extracted and global_rois is not empty
                logging.info(f"\n--- Thickness Statistics for {len(global_rois)} Image-Defined Rectangular ROIs ---")

                roi_stats_for_viz = []

                for i, (x_min, y_min, width, height) in enumerate(global_rois):
                    current_label_tag = global_custom_labels[i]
                    all_unique_roi_labels.add(current_label_tag)  # Add current label to the set of all labels

                    logging.info(
                        f"  Processing ROI {i + 1} ({current_label_tag}) (x:{x_min}, y:{y_min}, w:{width}, h:{height}):")

                    current_x_min = max(0, x_min)
                    current_y_min = max(0, y_min)
                    current_x_max = min(current_image_np_original.shape[1], x_min + width)
                    current_y_max = min(current_image_np_original.shape[0], y_min + height)

                    current_width = current_x_max - current_x_min
                    current_height = current_y_max - current_y_min

                    if current_width <= 0 or current_height <= 0:
                        logging.warning(
                            f"    ROI {i + 1} ({current_label_tag}) is outside or too small for current image. Skipping.")
                        roi_stats_for_viz.append({'coords': (x_min, y_min, width, height),
                                                  'mean': np.nan, 'median': np.nan, 'std': np.nan, 'value': np.nan,
                                                  'custom_label': current_label_tag,
                                                  'metric_type': 'N/A'})
                        current_sample_excel_row[current_label_tag] = np.nan  # Ensure NaN is recorded for Excel
                        continue

                    metric_type = roi_metric_choices.get(current_label_tag, 'MAX')

                    directional_measurements_for_roi = []
                    raw_line_start_xy_roi = None
                    raw_line_end_xy_roi = None

                    chosen_line_length_pixels = -1 if metric_type == 'MAX' else float('inf')

                    scan_axis_is_row = (current_width > current_height)

                    if metric_type == 'MAX':
                        roi_binary_segment = image_for_max_calc[current_y_min: current_y_max,
                                             current_x_min: current_x_max]

                        if scan_axis_is_row:
                            for r_idx in range(roi_binary_segment.shape[0]):
                                line_pixels = roi_binary_segment[r_idx, :]
                                true_coords_in_line = np.where(line_pixels)[0]

                                if true_coords_in_line.size > 0:
                                    current_line_length = true_coords_in_line.max() - true_coords_in_line.min() + 1
                                    directional_measurements_for_roi.append(current_line_length)

                                    if current_line_length > chosen_line_length_pixels:
                                        chosen_line_length_pixels = current_line_length
                                        raw_line_start_xy_roi = (true_coords_in_line.min(), r_idx)
                                        raw_line_end_xy_roi = (true_coords_in_line.max(), r_idx)
                        else:
                            for c_idx in range(roi_binary_segment.shape[1]):
                                line_pixels = roi_binary_segment[:, c_idx]
                                true_coords_in_line = np.where(line_pixels)[0]

                                if true_coords_in_line.size > 0:
                                    current_line_length = true_coords_in_line.max() - true_coords_in_line.min() + 1
                                    directional_measurements_for_roi.append(current_line_length)

                                    if current_line_length > chosen_line_length_pixels:
                                        chosen_line_length_pixels = current_line_length
                                        raw_line_start_xy_roi = (c_idx, true_coords_in_line.min())
                                        raw_line_end_xy_roi = (c_idx, true_coords_in_line.max())

                    else:  # metric_type == 'MIN'
                        roi_edges_segment = image_for_min_calc[current_y_min: current_y_max,
                                            current_x_min: current_x_max]

                        if scan_axis_is_row:
                            for r_idx in range(roi_edges_segment.shape[0]):
                                line_pixels = roi_edges_segment[r_idx, :]
                                segments = find_contiguous_segments_with_positions(line_pixels)
                                true_coords_in_line = np.where(line_pixels)[0]

                                if len(segments) >= 3:
                                    segments.sort(key=lambda x: x[1][0])

                                    segment_start_coords = [s[1][0] for s in segments]
                                    segment_end_coords = [s[1][1] for s in segments]

                                    gap1_start = segment_end_coords[0] + 1
                                    gap1_end = segment_start_coords[1]
                                    gap1_length = gap1_end - gap1_start

                                    gap2_start = segment_end_coords[1] + 1
                                    gap2_end = segment_start_coords[2]
                                    gap2_length = gap2_end - gap2_start

                                    current_line_measure = max(gap1_length, gap2_length)

                                    if current_line_measure > 0:
                                        directional_measurements_for_roi.append(current_line_measure)

                                        if gap1_length >= gap2_length:
                                            temp_start_coord = gap1_start
                                            temp_end_coord = gap1_end - 1
                                        else:
                                            temp_start_coord = gap2_start
                                            temp_end_coord = gap2_end - 1

                                        if current_line_measure < chosen_line_length_pixels:
                                            chosen_line_length_pixels = current_line_measure
                                            raw_line_start_xy_roi = (temp_start_coord, r_idx)
                                            raw_line_end_xy_roi = (temp_end_coord, r_idx)
                                    else:
                                        logging.debug(
                                            f"ROI {current_label_tag} (H-scan, r_idx {r_idx}): Calculated gap(s) not positive. Skipping.")
                                else:  # Fallback to MAX case: <= 2 segments, measure overall span
                                    if true_coords_in_line.size > 0:
                                        current_line_measure = true_coords_in_line.max() - true_coords_in_line.min() + 1
                                        directional_measurements_for_roi.append(current_line_measure)
                                        logging.debug(
                                            f"ROI {current_label_tag} (H-scan, r_idx {r_idx}): Falling back to MAX logic (span={current_line_measure}).")

                                        if current_line_measure < chosen_line_length_pixels:
                                            chosen_line_length_pixels = current_line_measure
                                            raw_line_start_xy_roi = (true_coords_in_line.min(), r_idx)
                                            raw_line_end_xy_roi = (true_coords_in_line.max(), r_idx)
                                    else:
                                        logging.debug(
                                            f"ROI {current_label_tag} (H-scan, r_idx {r_idx}): No measurable features for MIN or fallback. Skipping.")


                        else:  # Vertical ROI
                            for c_idx in range(roi_edges_segment.shape[1]):
                                line_pixels = roi_edges_segment[:, c_idx]
                                segments = find_contiguous_segments_with_positions(line_pixels)
                                true_coords_in_line = np.where(line_pixels)[0]

                                if len(segments) >= 3:
                                    segments.sort(key=lambda x: x[1][0])  # Sort by start_idx (row)

                                    segment_start_coords = [s[1][0] for s in segments]
                                    segment_end_coords = [s[1][1] for s in segments]

                                    gap1_start = segment_end_coords[0] + 1
                                    gap1_end = segment_start_coords[1]
                                    gap1_length = gap1_end - gap1_start

                                    gap2_start = segment_end_coords[1] + 1
                                    gap2_end = segment_start_coords[2]
                                    gap2_length = gap2_end - gap2_start

                                    current_line_measure = max(gap1_length, gap2_length)

                                    if current_line_measure > 0:
                                        directional_measurements_for_roi.append(current_line_measure)

                                        if gap1_length >= gap2_length:
                                            temp_start_coord = gap1_start
                                            temp_end_coord = gap1_end - 1
                                        else:
                                            temp_start_coord = gap2_start
                                            temp_end_coord = gap2_end - 1

                                        if current_line_measure < chosen_line_length_pixels:
                                            chosen_line_length_pixels = current_line_measure
                                            raw_line_start_xy_roi = (c_idx, temp_start_coord)
                                            raw_line_end_xy_roi = (c_idx, temp_end_coord)
                                    else:
                                        logging.debug(
                                            f"ROI {current_label_tag} (V-scan, c_idx {c_idx}): Calculated gap(s) not positive. Skipping.")
                                else:  # Fallback to MAX case: <= 2 segments, measure overall span
                                    if true_coords_in_line.size > 0:
                                        current_line_measure = true_coords_in_line.max() - true_coords_in_line.min() + 1
                                        directional_measurements_for_roi.append(current_line_measure)
                                        logging.debug(
                                            f"ROI {current_label_tag} (V-scan, c_idx {c_idx}): Falling back to MAX logic (span={current_line_measure}).")

                                        if current_line_measure < chosen_line_length_pixels:
                                            chosen_line_length_pixels = current_line_measure
                                            raw_line_start_xy_roi = (c_idx, true_coords_in_line.min())
                                            raw_line_end_xy_roi = (c_idx, true_coords_in_line.max())
                                    else:
                                        logging.debug(
                                            f"ROI {current_label_tag} (V-scan, c_idx {c_idx}): No measurable features for MIN or fallback. Skipping.")

                    if directional_measurements_for_roi:
                        calibrated_measurements = np.array(directional_measurements_for_roi) / distance_per_pixel

                        if calibrated_measurements.size > 0:
                            mean_t = np.mean(calibrated_measurements)
                            median_t = np.median(calibrated_measurements)
                            std_t = np.std(calibrated_measurements)

                            if metric_type == 'MAX':
                                chosen_metric_value = np.max(calibrated_measurements)
                            else:  # MIN (based on identified gaps or fallback spans)
                                chosen_metric_value = np.min(calibrated_measurements)

                            logging.info(
                                f"    Min of measured values: {np.min(calibrated_measurements):.2f} {unit_label}, Max of measured values: {np.max(calibrated_measurements):.2f} {unit_label}, Mean: {mean_t:.2f} {unit_label}, Median: {median_t:.2f} {unit_label}, StdDev: {std_t:.2f} {unit_label}")
                            logging.info(f"    Chosen Metric ({metric_type}): {chosen_metric_value:.2f} {unit_label}")

                            plot_start_xy = None
                            plot_end_xy = None
                            # Apply global ROI offset and Y-flip first
                            if raw_line_start_xy_roi and raw_line_end_xy_roi:
                                img_height_total = current_image_np_original.shape[0]

                                x1_global_raw = current_x_min + raw_line_start_xy_roi[0]
                                y1_global_raw = current_y_min + raw_line_start_xy_roi[1]
                                x2_global_raw = current_x_min + raw_line_end_xy_roi[0]
                                y2_global_raw = current_y_min + raw_line_end_xy_roi[1]

                                # Convert to Matplotlib plot coordinates (Y-axis inverted)
                                plot_x1_final = x1_global_raw
                                plot_y1_final = img_height_total - 1 - y1_global_raw
                                plot_x2_final = x2_global_raw
                                plot_y2_final = img_height_total - 1 - y2_global_raw

                                # No explicit arrow_offset_pixels applied here, as per your request
                                plot_start_xy = (plot_x1_final, plot_y1_final)
                                plot_end_xy = (plot_x2_final, plot_y2_final)

                            roi_stats_for_viz.append({'coords': (x_min, y_min, width, height),
                                                      'mean': mean_t, 'median': median_t, 'std': std_t,
                                                      'value': chosen_metric_value,
                                                      'custom_label': current_label_tag,
                                                      'metric_type': metric_type,
                                                      'max_t_start_plot_xy': plot_start_xy if metric_type == 'MAX' else None,
                                                      'max_t_end_plot_xy': plot_end_xy if metric_type == 'MAX' else None,
                                                      'min_t_start_plot_xy': plot_start_xy if metric_type == 'MIN' else None,
                                                      'min_t_end_plot_xy': plot_end_xy if metric_type == 'MIN' else None
                                                      })
                            # This is the line that records the value for Excel
                            current_sample_excel_row[current_label_tag] = chosen_metric_value
                        else:
                            logging.info(
                                f"    ROI {i + 1} ({current_label_tag}): No valid measurements after calibration or filtering. Skipping.")
                            roi_stats_for_viz.append({'coords': (x_min, y_min, width, height),
                                                      'mean': np.nan, 'median': np.nan, 'std': np.nan, 'value': np.nan,
                                                      'custom_label': current_label_tag,
                                                      'metric_type': 'N/A',
                                                      'max_t_start_plot_xy': None, 'max_t_end_plot_xy': None,
                                                      'min_t_start_plot_xy': None, 'min_t_end_plot_xy': None})
                            current_sample_excel_row[current_label_tag] = np.nan
                    else:
                        logging.info(
                            f"    ROI {i + 1} ({current_label_tag}): No valid directional measurements found. Skipping.")
                        roi_stats_for_viz.append({'coords': (x_min, y_min, width, height),
                                                  'mean': np.nan, 'median': np.nan, 'std': np.nan, 'value': np.nan,
                                                  'custom_label': current_label_tag,
                                                  'metric_type': 'N/A',
                                                  'max_t_start_plot_xy': None, 'max_t_end_plot_xy': None,
                                                  'min_t_start_plot_xy': None, 'min_t_end_plot_xy': None})
                        current_sample_excel_row[current_label_tag] = np.nan

                all_samples_excel_data.append(current_sample_excel_row)

            else:
                thickness_map_calibrated_overall = measure_thickness_edt(image_for_max_calc) / distance_per_pixel
                foreground_thickness_values = thickness_map_calibrated_overall[image_for_max_calc]

                overall_max = np.nan
                if foreground_thickness_values.size > 0:
                    overall_max = np.max(foreground_thickness_values)
                    logging.info(
                        f"Overall Minimum thickness (2D): {np.min(foreground_thickness_values):.2f} {unit_label}")
                    logging.info(f"Overall Maximum thickness (2D): {overall_max:.2f} {unit_label}")
                    logging.info(
                        f"Overall Average thickness (2D): {np.mean(foreground_thickness_values):.2f} {unit_label}")
                    logging.info(
                        f"Overall Median thickness (2D): {np.median(foreground_thickness_values):.2f} {unit_label}")
                    logging.info(
                        f"Overall Standard deviation of thickness (2D): {np.std(foreground_thickness_values):.2f} {unit_label}")
                else:
                    logging.warning("No foreground pixels detected for overall statistics.")

                current_sample_excel_row['Overall_Max'] = overall_max
                all_unique_roi_labels.add('Overall_Max')
                all_samples_excel_data.append(current_sample_excel_row)

            output_filename = os.path.splitext(os.path.basename(image_path_current))[0] + "-processed.png"
            plot_save_path = os.path.join(processed_images_dir, output_filename)

            visualize_thickness_2d(thickness_map_for_viz / distance_per_pixel,
                                   original_image_for_overlay=current_image_np_original,
                                   unit_label=unit_label, roi_data=roi_stats_for_viz,
                                   distance_per_pixel_for_drawing=distance_per_pixel,
                                   plot_basename=plot_basename,
                                   original_filename=image_path_current,
                                   canny_params=canny_params,
                                   exposure_params=exposure_params,
                                   save_path=plot_save_path)

        except Exception as e:
            sg.popup_error("Processing Error",
                           f"An error occurred during thickness measurement or visualization for '{os.path.basename(image_path_current)}': {e}\nSkipping this image.")
            logging.error(
                f"An error occurred during thickness measurement or visualization for '{os.path.basename(image_path_current)}': {e}",
                exc_info=True)
            continue

    if all_samples_excel_data:
        # Ensure all ROI labels are present in each row, filling with NaN if missing
        sorted_roi_labels = sorted(list(all_unique_roi_labels))
        # Ensure 'Sample #' is the first column
        excel_columns = ['Sample #'] + sorted_roi_labels

        df = pd.DataFrame(all_samples_excel_data, columns=excel_columns)

        excel_output_path = os.path.join(output_directory, f"{plot_basename}_thickness_data.xlsx")
        try:
            df.to_excel(excel_output_path, index=False)
            sg.popup_ok("Processing Complete", f"All images processed and data saved to:\n{excel_output_path}\n"
                                               f"Processed images saved to:\n{processed_images_dir}")
            logging.info(f"Excel data saved to: {excel_output_path}")
        except Exception as e:
            sg.popup_error("Excel Save Error", f"Failed to save Excel file: {e}")
            logging.error(f"Error saving Excel file: {e}", exc_info=True)
    else:
        sg.popup_ok("No Data Processed", "No valid image data was processed to compile for Excel.")
        logging.warning("No data collected for Excel output.")

    logging.info("2D Image Thickness Measurement Application finished.")


if __name__ == "__main__":
    main()
