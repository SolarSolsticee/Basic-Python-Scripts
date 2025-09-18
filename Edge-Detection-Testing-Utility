import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, color
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import argparse


class EdgeDetectionTester:
    def __init__(self):
        self.image = None
        self.image_path = None
        self.processed_image = None

    def load_image(self, path=None):
        """Load an image from file path or prompt user to select one"""
        if path is None:
            # Hide the main tkinter window
            root = tk.Tk()
            root.withdraw()

            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                    ("All files", "*.*")
                ]
            )

            if not file_path:
                print("No image selected.")
                return False

            self.image_path = file_path
        else:
            self.image_path = path

        try:
            # Load image using OpenCV
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                print(f"Error: Could not load image from {self.image_path}")
                return False

            # Convert BGR to RGB for display
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            print(f"Image loaded successfully: {self.image_path}")
            print(f"Image shape: {self.image.shape}")
            return True

        except Exception as e:
            print(f"Error loading image: {e}")
            return False

    def display_menu(self):
        """Display the main menu for selecting edge detection algorithms"""
        print("\n" + "=" * 50)
        print("EDGE DETECTION TESTING UTILITY")
        print("=" * 50)
        print("Available Edge Detection Algorithms:")
        print("1. Canny Edge Detection (OpenCV)")
        print("2. Sobel Edge Detection (OpenCV)")
        print("3. Laplacian Edge Detection (OpenCV)")
        print("4. Prewitt Edge Detection (scikit-image)")
        print("5. Roberts Edge Detection (scikit-image)")
        print("6. Scharr Edge Detection (OpenCV)")
        print("7. Gaussian Edge Detection (scikit-image)")
        print("8. LoG (Laplacian of Gaussian) (scikit-image)")
        print("9. Compare All Methods")
        print("0. Exit")
        print("=" * 50)

        try:
            choice = int(input("Select an algorithm (0-9): "))
            return choice
        except ValueError:
            print("Invalid input. Please enter a number.")
            return -1

    def get_canny_parameters(self):
        """Get parameters for Canny edge detection"""
        print("\nCanny Edge Detection Parameters:")
        print("Default values: threshold1=30, threshold2=80, sigma=1.0")
        print("(Lower thresholds recommended for radiographs/medical images)")

        try:
            threshold1 = input("Enter lower threshold (default 30): ")
            threshold1 = int(threshold1) if threshold1 else 30

            threshold2 = input("Enter upper threshold (default 80): ")
            threshold2 = int(threshold2) if threshold2 else 80

            sigma = input("Enter sigma for Gaussian smoothing (default 1.0): ")
            sigma = float(sigma) if sigma else 1.0

            return {'threshold1': threshold1, 'threshold2': threshold2, 'sigma': sigma}
        except ValueError:
            print("Invalid input. Using default values.")
            return {'threshold1': 30, 'threshold2': 80, 'sigma': 1.0}

    def get_sobel_parameters(self):
        """Get parameters for Sobel edge detection"""
        print("\nSobel Edge Detection Parameters:")
        print("Default values: ksize=3 (kernel size)")

        try:
            ksize = input("Enter kernel size (1, 3, 5, 7, default 3): ")
            ksize = int(ksize) if ksize else 3
            if ksize not in [1, 3, 5, 7]:
                ksize = 3

            return {'ksize': ksize}
        except ValueError:
            print("Invalid input. Using default values.")
            return {'ksize': 3}

    def get_laplacian_parameters(self):
        """Get parameters for Laplacian edge detection"""
        print("\nLaplacian Edge Detection Parameters:")
        print("Default values: ksize=3")

        try:
            ksize = input("Enter kernel size (1, 3, 5, 7, default 3): ")
            ksize = int(ksize) if ksize else 3
            if ksize not in [1, 3, 5, 7]:
                ksize = 3

            return {'ksize': ksize}
        except ValueError:
            print("Invalid input. Using default values.")
            return {'ksize': 3}

    def get_gaussian_parameters(self):
        """Get parameters for Gaussian edge detection"""
        print("\nGaussian Edge Detection Parameters:")
        print("Default values: sigma=1.0")

        try:
            sigma = input("Enter sigma value (default 1.0): ")
            sigma = float(sigma) if sigma else 1.0

            return {'sigma': sigma}
        except ValueError:
            print("Invalid input. Using default values.")
            return {'sigma': 1.0}

    def apply_canny(self, params):
        """Apply Canny edge detection with Gaussian smoothing"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur first if sigma is specified
        if params.get('sigma', 1.0) != 1.0:
            gray = cv2.GaussianBlur(gray, (0, 0), params['sigma'])

        edges = cv2.Canny(gray, params['threshold1'], params['threshold2'])
        return edges

    def apply_sobel(self, params):
        """Apply Sobel edge detection"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # Apply Sobel in X and Y directions
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=params['ksize'])
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=params['ksize'])

        # Combine the two gradients
        sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Normalize to 0-255 range
        sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

        return sobel_combined

    def apply_laplacian(self, params):
        """Apply Laplacian edge detection"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=params['ksize'])

        # Convert to absolute values and normalize
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(np.clip(laplacian, 0, 255))

        return laplacian

    def apply_prewitt(self, params=None):
        """Apply Prewitt edge detection using scikit-image"""
        gray = color.rgb2gray(self.image)
        edges = filters.prewitt(gray)

        # Convert to 0-255 range
        edges = np.uint8(edges * 255)

        return edges

    def apply_roberts(self, params=None):
        """Apply Roberts edge detection using scikit-image"""
        gray = color.rgb2gray(self.image)
        edges = filters.roberts(gray)

        # Convert to 0-255 range
        edges = np.uint8(edges * 255)

        return edges

    def apply_scharr(self, params=None):
        """Apply Scharr edge detection"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # Apply Scharr in X and Y directions
        scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

        # Combine the two gradients
        scharr_combined = np.sqrt(scharr_x ** 2 + scharr_y ** 2)

        # Normalize to 0-255 range
        scharr_combined = np.uint8(np.clip(scharr_combined, 0, 255))

        return scharr_combined

    def apply_gaussian(self, params):
        """Apply Gaussian edge detection using scikit-image"""
        gray = color.rgb2gray(self.image)
        edges = feature.canny(gray, sigma=params['sigma'])

        # Convert to 0-255 range
        edges = np.uint8(edges * 255)

        return edges

    def apply_log(self, params=None):
        """Apply Laplacian of Gaussian edge detection"""
        gray = color.rgb2gray(self.image)
        edges = filters.laplace(filters.gaussian(gray, sigma=1))

        # Convert to absolute values and normalize
        edges = np.absolute(edges)
        edges = np.uint8((edges / edges.max()) * 255)

        return edges

    def display_results(self, edges, title="Edge Detection Result", params=None):
        """Display the original image and edge detection result with parameters"""
        plt.figure(figsize=(14, 7))

        # Get image filename
        image_name = os.path.basename(self.image_path) if self.image_path else "Unknown"

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(self.image)
        plt.title(f"Original Image\n{image_name}")
        plt.axis('off')

        # Create detailed title with parameters
        detailed_title = title
        if params:
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            detailed_title += f"\nParameters: {param_str}"
        detailed_title += f"\nImage: {image_name}"

        # Edge detection result
        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap='gray')
        plt.title(detailed_title)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def compare_all_methods(self):
        """Compare all edge detection methods"""
        methods = [
            ("Canny", self.apply_canny, {'threshold1': 30, 'threshold2': 80, 'sigma': 1.0}),
            ("Sobel", self.apply_sobel, {'ksize': 3}),
            ("Laplacian", self.apply_laplacian, {'ksize': 3}),
            ("Prewitt", self.apply_prewitt, None),
            ("Roberts", self.apply_roberts, None),
            ("Scharr", self.apply_scharr, None),
            ("Gaussian", self.apply_gaussian, {'sigma': 1.0}),
            ("LoG", self.apply_log, None)
        ]

        # Get image filename
        image_name = os.path.basename(self.image_path) if self.image_path else "Unknown"

        plt.figure(figsize=(16, 10))

        # Original image
        plt.subplot(3, 3, 1)
        plt.imshow(self.image)
        plt.title(f"Original Image\n{image_name}")
        plt.axis('off')

        # Apply each method
        for i, (name, method, params) in enumerate(methods, 2):
            try:
                if params:
                    edges = method(params)
                else:
                    edges = method()

                plt.subplot(3, 3, i)
                plt.imshow(edges, cmap='gray')

                # Create title with parameters
                method_title = f"{name} Edge Detection"
                if params:
                    param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                    method_title += f"\n{param_str}"

                plt.title(method_title)
                plt.axis('off')

            except Exception as e:
                print(f"Error applying {name}: {e}")
                plt.subplot(3, 3, i)
                plt.text(0.5, 0.5, f"Error: {name}", ha='center', va='center')
                plt.title(f"{name} Edge Detection\nError")
                plt.axis('off')

        plt.suptitle(f"Edge Detection Comparison - {image_name}", fontsize=16)
        plt.tight_layout()
        plt.show()

    def run(self):
        """Main program loop"""
        print("Welcome to the Edge Detection Testing Utility!")

        # Load image
        if not self.load_image():
            return

        while True:
            choice = self.display_menu()

            if choice == 0:
                print("Exiting...")
                break
            elif choice == 1:
                params = self.get_canny_parameters()
                edges = self.apply_canny(params)
                self.display_results(edges, "Canny Edge Detection", params)
            elif choice == 2:
                params = self.get_sobel_parameters()
                edges = self.apply_sobel(params)
                self.display_results(edges, "Sobel Edge Detection", params)
            elif choice == 3:
                params = self.get_laplacian_parameters()
                edges = self.apply_laplacian(params)
                self.display_results(edges, "Laplacian Edge Detection", params)
            elif choice == 4:
                edges = self.apply_prewitt()
                self.display_results(edges, "Prewitt Edge Detection", None)
            elif choice == 5:
                edges = self.apply_roberts()
                self.display_results(edges, "Roberts Edge Detection", None)
            elif choice == 6:
                edges = self.apply_scharr()
                self.display_results(edges, "Scharr Edge Detection", None)
            elif choice == 7:
                params = self.get_gaussian_parameters()
                edges = self.apply_gaussian(params)
                self.display_results(edges, "Gaussian Edge Detection", params)
            elif choice == 8:
                edges = self.apply_log()
                self.display_results(edges, "LoG Edge Detection", None)
            elif choice == 9:
                self.compare_all_methods()
            else:
                print("Invalid choice. Please try again.")

            # Ask if user wants to continue
            continue_choice = input("\nDo you want to try another algorithm? (y/n): ").lower()
            if continue_choice != 'y':
                # Ask if user wants to load a new image
                new_image = input("Do you want to load a new image? (y/n): ").lower()
                if new_image == 'y':
                    if not self.load_image():
                        break
                else:
                    break


def main():
    """Main function to run the edge detection utility"""
    parser = argparse.ArgumentParser(description="Edge Detection Testing Utility")
    parser.add_argument("--image", "-i", type=str, help="Path to input image")
    args = parser.parse_args()

    tester = EdgeDetectionTester()

    if args.image:
        if not tester.load_image(args.image):
            print("Failed to load specified image. Exiting.")
            return

    tester.run()


if __name__ == "__main__":
    main()
