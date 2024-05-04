import cv2
import numpy as np


def calculate_sharpness(image):
    """Calculate the sharpness of an image using the gradient magnitude."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.mean(gradient_magnitude)[0]


def normalize(data):
    """ Normalize data to [0, 1] range. """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def calculate_laplacian_variance(image):
    """ Calculate sharpness using the Laplacian variance method. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def calculate_sobel_edges(image):
    """ Calculate sharpness based on edge information using Sobel operators. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(grad_x, grad_y)
    return np.mean(sobel_edges)


def calculate_fourier_transform(image):
    """ Calculate sharpness based on the high-frequency content using Fourier Transform. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    high_freq = np.sum(magnitude_spectrum > np.mean(magnitude_spectrum))
    return high_freq


def calculate_composite_sharpness(image, weights=(0.5, 0.25, 0.25)):
    """ Calculate a composite sharpness score based on multiple techniques. """
    laplacian_score = calculate_laplacian_variance(image)
    sobel_score = calculate_sobel_edges(image)
    fourier_score = calculate_fourier_transform(image)

    # Normalize scores
    scores = np.array([laplacian_score, sobel_score, fourier_score])
    normalized_scores = normalize(scores)

    # Weighted average
    composite_score = np.dot(weights, normalized_scores)
    return composite_score


def detect_changes(prev_frame, current_frame, threshold, num_of_diff_pixels):
    """Detect significant changes between two frames."""
    diff = cv2.absdiff(prev_frame, current_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    changes = []
    for contour in contours:
        if cv2.contourArea(contour) > num_of_diff_pixels:
            x, y, w, h = cv2.boundingRect(contour)
            roi = current_frame[y:y + h, x:x + w]
            changes.append((roi, x, y, w, h))
    return changes


def initialize_background_subtractor():
    """ Initialize the background subtractor model. """
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    return back_sub






