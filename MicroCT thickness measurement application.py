import numpy as np
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.morphology import disk
from skimage import io, color
from skimage.feature import canny
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

# Removed pandas import as it's no longer needed in this single-image version

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
                           original_filename="", canny_params=None):
    """
    Visualizes the 2D thickness map. Optionally overlays the original image and ROI boxes.

    Args:
        thickness_map (np.array): The 2D array of thickness values.
        original_image_for_overlay (np.array, optional): The original grayscale image
                                                        to overlay for context.
        unit_label (str): The label for the units (e.g., "pixels", "mm").
        roi_data (list, optional): A list of dictionaries, each containing ROI coords, stats, and custom label.
        distance_per_pixel_for_drawing (float): The pixels/mm value used to convert mean_t (in mm) back to pixels for drawing.
        plot_basename (str): The user-defined basename for the plot title.
        original_filename (str): The filename of the original image.
        canny_params (dict, optional): Dictionary containing 'sigma', 'low_threshold', 'high_threshold'.
    """
    if thickness_map.ndim != 2:
        logging.error("Thickness map must be 2D for visualization.")
        return

    logging.info(f"Visualizing 2D thickness map in {unit_label}.")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

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

    # Draw ROI boxes and mean lines/text if provided
    if roi_data:
        height_img = thickness_map.shape[0]  # Get original image height for y-flip
        for i, roi_info in enumerate(roi_data):
            x_min, y_min, width, height = roi_info['coords']
            mean_t = roi_info['mean']
            median_t = roi_info['median']
            std_t = roi_info['std']
            custom_label = roi_info.get('custom_label', '')  # Get custom label, default to empty string

            # Adjust y_min for origin='lower' plotting
            rect_y_min_flipped = height_img - (y_min + height)

            # Draw ROI box
            rect = patches.Rectangle((x_min, rect_y_min_flipped), width, height,
                                     linewidth=1, edgecolor='red', facecolor='red', alpha=0.2, linestyle='-')
            ax.add_patch(rect)

            # Add ROI number label to the top-left corner
            text_x_roi_num = x_min + 5  # Small offset from left edge
            text_y_roi_num = rect_y_min_flipped + height - 5  # Small offset from top edge
            ax.text(text_x_roi_num, text_y_roi_num, str(i + 1), color='white', fontsize=8, ha='left', va='top',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

            # Draw mean distance line with arrows
            line_color = 'yellow'
            line_style = ':'
            line_width = 1.5
            arrow_style = '<->'

            # Convert mean_t (in mm) back to pixels for drawing on the image
            mean_t_pixels = mean_t * distance_per_pixel_for_drawing

            if width > height:  # Horizontal ROI, draw horizontal line
                line_y = rect_y_min_flipped + height / 2
                line_start_x = x_min + (width / 2) - (mean_t_pixels / 2)
                line_end_x = x_min + (width / 2) + (mean_t_pixels / 2)
                ax.annotate('', xy=(line_end_x, line_y), xytext=(line_start_x, line_y),
                            arrowprops=dict(arrowstyle=arrow_style, color=line_color, lw=line_width, ls=line_style))
            else:  # Vertical ROI, draw vertical line
                line_x = x_min + width / 2
                line_start_y = rect_y_min_flipped + (height / 2) - (mean_t_pixels / 2)
                line_end_y = rect_y_min_flipped + (height / 2) + (mean_t_pixels / 2)
                ax.annotate('', xy=(line_x, line_end_y), xytext=(line_x, line_start_y),
                            arrowprops=dict(arrowstyle=arrow_style, color=line_color, lw=line_width, ls=line_style))

            # Write custom label with smart positioning
            if custom_label:
                if width > height:  # Horizontal ROI, label to the right
                    label_x = x_min + width + 5  # Offset to the right
                    label_y = rect_y_min_flipped + height / 2
                    ax.text(label_x, label_y, custom_label,
                            color='blue', fontsize=8, ha='left', va='center',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))
                else:  # Vertical ROI, label above
                    label_x = x_min + width / 2
                    label_y = rect_y_min_flipped + height + 5  # Offset above
                    ax.text(label_x, label_y, custom_label,
                            color='blue', fontsize=8, ha='center', va='bottom',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))

            # Write Median and StdDev values
            text_x_stats = x_min + width / 2
            text_y_median = rect_y_min_flipped - 10  # Offset below the box
            text_y_stddev = rect_y_min_flipped - 25  # Further offset for StdDev

            if not np.isnan(median_t):
                ax.text(text_x_stats, text_y_median, f'Med: {median_t:.2f}{unit_label}',
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
    ax.set_xlabel(f'X-axis ({unit_label})')  # Changed 'X-coordinate' to 'X-axis'

    # For Y-axis
    y_ticks_px = np.linspace(0, img_height - 1, num=5, dtype=int)  # Example: 5 ticks
    y_tick_labels_mm = [f"{p / distance_per_pixel_for_drawing:.1f}" for p in y_ticks_px]
    ax.set_yticks(y_ticks_px)
    ax.set_yticklabels(y_tick_labels_mm)
    ax.set_ylabel(f'Y-axis ({unit_label})')  # Changed 'Y-coordinate' to 'Y-axis'

    # Set the plot title with basename and filename
    title_text = f"{plot_basename}"
    if original_filename:
        title_text += f" ({os.path.basename(original_filename)})"
    title_text += f" Radiography thickness dist. analysis"  # Changed '2D Thickness Map'
    ax.set_title(title_text)

    # Add Canny parameters box to top-right corner
    if canny_params:
        param_text = (f"Canny Parameters:\n"
                      f"  Sigma: {canny_params['sigma']:.2f}\n"
                      f"  Low Thresh: {canny_params['low_threshold']:.2f}\n"
                      f"  High Thresh: {canny_params['high_threshold']:.2f}")

        # Position the text box in the top-right corner
        # `transform=ax.transAxes` means coordinates are relative to the axes (0,0 to 1,1)
        ax.text(0.98, 0.98, param_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7, ec='gray'))

    plt.tight_layout()
    # Removed save_path logic, now always displays
    plt.show()
    logging.info("2D thickness map visualization complete.")


def get_float_input(title, prompt, default_value, parent):
    """Helper function to get float input with error handling."""
    while True:
        value_str = simpledialog.askstring(title, prompt, initialvalue=str(default_value), parent=parent)
        if value_str is None:  # User cancelled
            logging.info(f"User cancelled input for '{title}'. Using default value: {default_value}")
            return default_value  # Return default value if cancelled
        try:
            float_value = float(value_str)
            logging.info(f"User entered '{float_value}' for '{title}'.")
            return float_value
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")


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

        # If image has an alpha channel (RGBA), separate it.
        # If it's RGB, it will be (H, W, 3).
        if roi_img.ndim == 3 and roi_img.shape[2] == 4:  # RGBA
            rgb_img = roi_img[:, :, :3]
            alpha_channel = roi_img[:, :, 3]
        elif roi_img.ndim == 3 and roi_img.shape[2] == 3:  # RGB
            rgb_img = roi_img
            alpha_channel = None  # No explicit alpha
        else:
            messagebox.showwarning("ROI Image Warning", "ROI image is not RGB or RGBA. Cannot detect red boxes.")
            logging.warning("ROI image is not RGB/RGBA. Cannot detect red boxes.")
            return []

        # Convert to float and scale to 0-1 if not already (skimage.io.imread often does this)
        if rgb_img.dtype == np.uint8:
            rgb_img = rgb_img / 255.0

        # Define a color range for "red"
        # These thresholds might need tuning based on the exact shade of red in your ROI image
        red_threshold_low = 0.7  # Minimum red intensity
        green_threshold_high = 0.3  # Maximum green intensity
        blue_threshold_high = 0.3  # Maximum blue intensity

        # Create a binary mask where red pixels are True
        red_mask = (rgb_img[:, :, 0] > red_threshold_low) & \
                   (rgb_img[:, :, 1] < green_threshold_high) & \
                   (rgb_img[:, :, 2] < blue_threshold_high)

        if not np.any(red_mask):
            messagebox.showwarning("ROI Detection Failed",
                                   "No red pixels detected in the ROI image. Please check the image content and color thresholds.")
            logging.warning("No red pixels detected in ROI image.")
            return []

        # Label connected components of red pixels
        labeled_rois = label(red_mask)
        props = regionprops(labeled_rois)

        if not props:
            messagebox.showwarning("ROI Detection Failed", "No distinct red regions (ROIs) found in the image.")
            logging.warning("No distinct red regions found in ROI image.")
            return []

        extracted_rois = []
        for prop in props:
            # Bounding box (min_row, min_col, max_row, max_col)
            min_row, min_col, max_row, max_col = prop.bbox
            x_min = min_col
            y_min = min_row  # y_min corresponds to the top of the box
            width = max_col - min_col
            height = max_row - min_row
            extracted_rois.append((x_min, y_min, width, height))

        # Sort ROIs for consistent numbering (e.g., top-left to bottom-right)
        extracted_rois.sort(key=lambda r: (r[1], r[0]))  # Sort by y_min then x_min

        logging.info(f"Found {len(extracted_rois)} ROIs in the image.")
        return extracted_rois

    except Exception as e:
        messagebox.showerror("ROI Image Error", f"Failed to load or process ROI image: {e}")
        logging.error(f"Error extracting ROIs from image: {e}", exc_info=True)
        return []


def main():
    """
    Main function to run the 2D image thickness measurement application.
    """
    logging.info("Starting 2D Image Thickness Measurement Application.")

    binary_image = None
    image_np_original = None  # Store original grayscale for potential overlay
    image_path = None
    roi_image_path = None  # Path to the ROI image
    distance_per_pixel = None
    default_distance_per_pixel = 28.089  # Default value for distance per pixel

    # Canny edge detection default parameters
    default_canny_sigma = 1.0
    default_canny_low_threshold = 0.1
    default_canny_high_threshold = 0.2

    # Store Canny parameters in a dictionary to pass to visualize_thickness_2d
    canny_params = {}

    # Predefined labels for 'v1 transparent'
    V1_TRANSPARENT_LABELS = [
        'T4', 'T11', 'T3', 'T4.1', 'T10', 'T12', 'T2', 'T8', 'T1',
        'T12.1', 'T9', 'T4.3', 'T12.2', 'T4.4', 'T12.3', 'T12.4', 'T13', 'T5'
    ]

    # Create a Tkinter root window but hide it
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # --- Get Basename for Plot Title ---
    plot_basename = simpledialog.askstring(
        "Plot Title Basename",
        "Enter a basename for the plot title (e.g., 'Sample A-1'):",
        parent=root
    )
    if plot_basename is None:  # User cancelled
        plot_basename = "Untitled"
        messagebox.showinfo("Plot Title", "No basename entered. Using 'Untitled'.")

    # --- Step 1: Get Distance Per Pixel from User ---
    try:
        distance_per_pixel = get_float_input(
            "Input Calibration",
            f"Enter the pixels per mm (e.g., pixels/mm).\n"
            f"Default value: {default_distance_per_pixel}",
            default_distance_per_pixel,
            root
        )

        if distance_per_pixel is None:  # User clicked Cancel
            messagebox.showinfo("Calibration Cancelled",
                                "No distance per pixel entered. Calculations will be in pixels.")
            distance_per_pixel = 1.0
            unit_label = "pixels"
        else:
            unit_label = "mm"  # Assuming mm as the desired output unit
            logging.info(f"Distance per pixel set to: {distance_per_pixel} {unit_label}")

    except Exception as e:
        messagebox.showerror("Input Error", f"An error occurred while getting distance per pixel: {e}")
        logging.error(f"Error getting distance per pixel: {e}", exc_info=True)
        distance_per_pixel = 1.0
        unit_label = "pixels"

    # --- Step 2: Get Canny Edge Detection Parameters from User ---
    canny_sigma = get_float_input(
        "Canny Parameters",
        f"Enter the Sigma value for Canny edge detection (smoothing).\n"
        f"Default: {default_canny_sigma}",
        default_canny_sigma,
        root
    )

    canny_low_threshold = get_float_input(
        "Canny Parameters",
        f"Enter the Low Threshold for Canny edge detection (hysteresis).\n"
        f"Default: {default_canny_low_threshold}",
        default_canny_low_threshold,
        root
    )

    canny_high_threshold = get_float_input(
        "Canny Parameters",
        f"Enter the High Threshold for Canny edge detection (hysteresis).\n"
        f"Default: {default_canny_high_threshold}",
        default_canny_high_threshold,
        root
    )

    canny_params = {
        'sigma': canny_sigma,
        'low_threshold': canny_low_threshold,
        'high_threshold': canny_high_threshold
    }
    logging.info(f"Final Canny parameters for processing: {canny_params}")

    # --- Step 3: Select Main Image for Analysis (single image) ---
    try:
        image_path = filedialog.askopenfilename(  # Reverted to single file selection
            title="Select the .bmp image file for analysis",
            filetypes=[("BMP files", "*.bmp"), ("All files", "*.*")],
            parent=root
        )
    except Exception as e:
        messagebox.showerror("File Selection Error", f"An error occurred during file selection: {e}")
        logging.error(f"Error during file dialog: {e}", exc_info=True)
        root.destroy()
        return

    if not image_path:  # If user cancelled file selection
        messagebox.showinfo("No Image Selected", "No image file was selected. Exiting application.")
        root.destroy()
        return

    # --- Step 4: Select ROI Image ---
    # This prompt is now outside the batch loop and will be asked once.
    roi_image_path = None  # Initialize outside the try block
    try:
        roi_image_path = filedialog.askopenfilename(
            title="Select a transparent image with red boxes for ROIs (e.g., .png) or cancel to skip",
            filetypes=[("PNG files", "*.png"), ("BMP files", "*.bmp"), ("All files", "*.*")],
            parent=root
        )
    except Exception as e:
        messagebox.showerror("ROI Image Selection Error", f"An error occurred during ROI image selection: {e}")
        logging.error(f"Error during ROI image dialog: {e}", exc_info=True)
        # roi_image_path remains None if an error occurs

    # Destroy the Tkinter root window after all dialogs
    root.destroy()

    # --- Load and Process Main Image ---
    binary_image = None
    image_np_original = None

    if not image_path.lower().endswith('.bmp'):
        messagebox.showwarning("File Type Warning",
                               f"File '{os.path.basename(image_path)}' is not a .bmp file. Attempting to load anyway.")
        logging.warning(
            f"Warning: File '{image_path}' is not a .bmp file. Attempting to load anyway, but .bmp is expected.")
    try:
        logging.info(f"Attempting to load main image from: {image_path}")
        image_np = io.imread(image_path)

        # Convert to grayscale if it's an RGB image
        if image_np.ndim == 3:
            logging.info("Converting RGB image to grayscale.")
            image_np = color.rgb2gray(image_np)
        elif image_np.ndim != 2:
            messagebox.showerror("Image Dimension Error",
                                 f"Unsupported image dimensions: {image_np.ndim}. Expected 2D or 3D (RGB). Using synthetic object.")
            logging.error(
                f"Unsupported image dimensions: {image_np.ndim}. Expected 2D or 3D (RGB). Using synthetic object.")
            binary_image = create_synthetic_2d_object(shape=(200, 200), object_type='ring').astype(bool)
            image_np_original = binary_image  # Use synthetic as original for overlay
            image_path = "synthetic_object.bmp"  # Update path for title

        if binary_image is None:  # Only proceed if no error or synthetic object already created
            image_np_original = image_np  # Store original grayscale for overlay

            # --- Segmentation: Edge Detection and Hole Filling ---
            logging.info("Applying Canny edge detection.")
            edges = canny(image_np, sigma=canny_params['sigma'], low_threshold=canny_params['low_threshold'],
                          high_threshold=canny_params['high_threshold'])

            logging.info("Filling holes to create binary object from edges.")
            binary_image = binary_fill_holes(edges)

            # Check if the binary image is empty after processing
            if not np.any(binary_image):
                messagebox.showwarning("No Object Detected",
                                       "No object was detected after edge detection and hole filling. "
                                       "Please adjust Canny parameters or check your image. Using synthetic object.")
                logging.warning("Binary image is empty after processing. Using synthetic object.")
                binary_image = create_synthetic_2d_object(shape=(200, 200), object_type='ring').astype(bool)
                image_np_original = binary_image  # Use synthetic as original for overlay
                image_path = "synthetic_object.bmp"  # Update path for title
            else:
                logging.info(f"Main image loaded, edge-detected, and binarized. Shape: {binary_image.shape}")

    except Exception as e:
        messagebox.showerror("Main Image Loading/Processing Error",
                             f"Failed to load or process main image '{os.path.basename(image_path)}': {e}\nUsing synthetic object instead.")
        logging.error(f"Failed to load or process main image '{os.path.basename(image_path)}': {e}", exc_info=True)
        binary_image = create_synthetic_2d_object(shape=(200, 200), object_type='ring').astype(bool)
        image_np_original = binary_image  # Use synthetic as original for overlay
        image_path = "synthetic_object.bmp"  # Update path for title

    # --- Extract ROIs from the selected ROI image ---
    rois = []
    if roi_image_path:
        rois = extract_rois_from_image(roi_image_path, binary_image.shape)
        if not rois:
            messagebox.showwarning("ROI Warning",
                                   "No valid ROIs extracted from the provided ROI image. Proceeding without specific ROI analysis.")
            logging.warning("No valid ROIs extracted from ROI image.")

    if not rois:  # If no ROIs are found, provide a fallback message
        messagebox.showinfo("No ROIs Defined",
                            "No custom ROI image provided or valid ROIs found. Proceeding with overall thickness statistics only.")
        logging.info("No ROIs defined, proceeding with overall thickness statistics.")

    # --- Get Custom ROI Labels (if ROIs were found) ---
    custom_labels = []
    if rois:
        # Use messagebox.askyesno to act as a toggle, then simpledialog for input
        use_hardcoded = messagebox.askyesno(
            "Custom ROI Labels",
            "Do you want to use the 'v1 transparent' hardcoded labels?\n"
            "Click 'Yes' for hardcoded, 'No' to enter custom labels."
        )

        if use_hardcoded:
            custom_labels = V1_TRANSPARENT_LABELS
            logging.info(f"Using 'v1 transparent' hardcoded labels: {custom_labels}")
        else:
            label_input = simpledialog.askstring(
                "Custom ROI Labels",
                f"Enter {len(rois)} custom labels for your ROIs, separated by commas.\n"
                f"Example: Top, Left, Bottom, Right, ...",
                parent=None
            )
            if label_input:
                custom_labels = [label.strip() for label in label_input.split(',')]
                logging.info(f"Custom labels entered: {custom_labels}")
            else:
                logging.info("No custom labels entered.")

    try:
        # Ensure the binary image is boolean type for EDT
        binary_image = binary_image.astype(bool)

        # --- Measure Thickness (2D Width/Diameter) ---
        thickness_map = measure_thickness_edt(binary_image)

        # --- Analyze and Collect Results ---
        # For single image, we don't need all_samples_excel_data or all_unique_roi_labels directly
        # But we still need roi_stats_for_viz for plotting

        if rois:
            logging.info(f"\n--- Thickness Statistics for {len(rois)} Image-Defined Rectangular ROIs ---")

            roi_stats_for_viz = []  # Data for visualization

            for i, (x_min, y_min, width, height) in enumerate(rois):
                logging.info(f"  Processing ROI {i + 1} (x:{x_min}, y:{y_min}, w:{width}, h:{height}):")

                roi_binary_segment = binary_image[y_min: y_min + height, x_min: x_min + width]

                current_custom_label = custom_labels[i] if i < len(custom_labels) else f"ROI_{i + 1}"

                if not np.any(roi_binary_segment):
                    logging.info(f"    ROI {i + 1}: No object pixels in this region. Skipping directional measurement.")
                    roi_stats_for_viz.append({'coords': (x_min, y_min, width, height),
                                              'mean': np.nan, 'median': np.nan, 'std': np.nan,
                                              'custom_label': current_custom_label})
                    continue

                directional_measurements = []
                if width > height:  # Horizontal ROI, measure X distances
                    for r_idx in range(roi_binary_segment.shape[0]):
                        row_pixels = roi_binary_segment[r_idx, :]
                        true_cols = np.where(row_pixels)[0]
                        if true_cols.size > 0:
                            distance = true_cols.max() - true_cols.min() + 1
                            directional_measurements.append(distance)
                else:  # Vertical ROI (height >= width), measure Y distances
                    for c_idx in range(roi_binary_segment.shape[1]):
                        col_pixels = roi_binary_segment[:, c_idx]
                        true_rows = np.where(col_pixels)[0]
                        if true_rows.size > 0:
                            distance = true_rows.max() - true_rows.min() + 1
                            directional_measurements.append(distance)

                if directional_measurements:
                    calibrated_measurements = np.array(directional_measurements) / distance_per_pixel
                    mean_t = np.mean(calibrated_measurements)
                    median_t = np.median(calibrated_measurements)
                    std_t = np.std(calibrated_measurements)

                    logging.info(
                        f"    Min: {np.min(calibrated_measurements):.2f} {unit_label}, Max: {np.max(calibrated_measurements):.2f} {unit_label}, Mean: {mean_t:.2f} {unit_label}, Median: {median_t:.2f} {unit_label}, StdDev: {std_t:.2f} {unit_label}")

                    roi_stats_for_viz.append({'coords': (x_min, y_min, width, height),
                                              'mean': mean_t, 'median': median_t, 'std': std_t,
                                              'custom_label': current_custom_label})
                else:
                    logging.info(f"    ROI {i + 1}: No valid directional measurements found. Skipping.")
                    roi_stats_for_viz.append({'coords': (x_min, y_min, width, height),
                                              'mean': np.nan, 'median': np.nan, 'std': np.nan,
                                              'custom_label': current_custom_label})

        else:  # Fallback to overall EDT statistics if no ROIs were defined
            logging.info("\n--- Overall Thickness Statistics (No specific ROIs defined) ---")
            thickness_map_calibrated_overall = thickness_map / distance_per_pixel
            foreground_thickness_values = thickness_map_calibrated_overall[binary_image]

            if foreground_thickness_values.size > 0:
                logging.info(f"Overall Minimum thickness (2D): {np.min(foreground_thickness_values):.2f} {unit_label}")
                logging.info(f"Overall Maximum thickness (2D): {np.max(foreground_thickness_values):.2f} {unit_label}")
                logging.info(f"Overall Average thickness (2D): {np.mean(foreground_thickness_values):.2f} {unit_label}")
                logging.info(
                    f"Overall Median thickness (2D): {np.median(foreground_thickness_values):.2f} {unit_label}")
                logging.info(
                    f"Overall Standard deviation of thickness (2D): {np.std(foreground_thickness_values):.2f} {unit_label}")
            else:
                logging.warning("No foreground pixels detected for overall statistics.")

            roi_stats_for_viz = None  # No specific ROIs to visualize

        # --- Visualize the processed image plot ---
        visualize_thickness_2d(thickness_map / distance_per_pixel,  # Pass calibrated map for visualization
                               original_image_for_overlay=image_np_original,
                               unit_label=unit_label, roi_data=roi_stats_for_viz,
                               distance_per_pixel_for_drawing=distance_per_pixel,
                               plot_basename=plot_basename,
                               original_filename=image_path,
                               canny_params=canny_params)

    except Exception as e:
        messagebox.showerror("Processing Error",
                             f"An error occurred during thickness measurement or visualization: {e}")
        logging.error(f"An error occurred during thickness measurement or visualization: {e}", exc_info=True)

    logging.info("2D Image Thickness Measurement Application finished.")


if __name__ == "__main__":
    main()
