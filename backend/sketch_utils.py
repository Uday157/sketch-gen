import cv2

def convert_to_sketch(image_path):
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert image
    inverted = cv2.bitwise_not(gray)

    # Blur image
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)

    # Invert blurred image
    inverted_blur = cv2.bitwise_not(blurred)

    # Final sketch
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)

    return sketch
