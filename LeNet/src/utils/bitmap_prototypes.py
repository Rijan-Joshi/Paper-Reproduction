from src.utils.cupy_numpy import np
from PIL import Image

# LeNet-5 original 7x12 bitmaps for digits 0-9
# Based on the original LeNet-5 paper by Yann LeCun et al.
# 1 = foreground (black), -1 = background (white)
# These are normalized to [-1, 1] as used in LeNet-5

lenet5_bitmaps = {
    0: [
        [-1, -1,  1,  1,  1, -1, -1],
        [-1,  1,  1,  1,  1,  1, -1],
        [ 1,  1,  1, -1,  1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1,  1, -1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1, -1],
        [-1, -1,  1,  1,  1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ],
    1: [
        [-1, -1, -1,  1,  1, -1, -1],
        [-1, -1,  1,  1,  1, -1, -1],
        [-1,  1,  1,  1,  1, -1, -1],
        [-1, -1, -1,  1,  1, -1, -1],
        [-1, -1, -1,  1,  1, -1, -1],
        [-1, -1, -1,  1,  1, -1, -1],
        [-1, -1, -1,  1,  1, -1, -1],
        [-1, -1, -1,  1,  1, -1, -1],
        [-1, -1, -1,  1,  1, -1, -1],
        [-1,  1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1,  1],
        [-1, -1, -1, -1, -1, -1, -1],
    ],
    2: [
        [-1,  1,  1,  1,  1,  1, -1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [-1, -1, -1, -1, -1,  1,  1],
        [-1, -1, -1, -1,  1,  1,  1],
        [-1, -1,  1,  1,  1,  1, -1],
        [-1,  1,  1,  1,  1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1],
        [ 1,  1, -1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [-1, -1, -1, -1, -1, -1, -1],
    ],
    3: [
        [-1,  1,  1,  1,  1,  1, -1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [-1, -1, -1, -1, -1,  1,  1],
        [-1, -1, -1,  1,  1,  1, -1],
        [-1, -1, -1,  1,  1,  1, -1],
        [-1, -1, -1, -1, -1,  1,  1],
        [-1, -1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ],
    4: [
        [-1, -1, -1, -1,  1,  1, -1],
        [-1, -1, -1,  1,  1,  1, -1],
        [-1, -1,  1,  1,  1,  1, -1],
        [-1,  1,  1, -1,  1,  1, -1],
        [ 1,  1, -1, -1,  1,  1, -1],
        [ 1,  1, -1, -1,  1,  1, -1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [-1, -1, -1, -1,  1,  1, -1],
        [-1, -1, -1, -1,  1,  1, -1],
        [-1, -1, -1, -1,  1,  1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ],
    5: [
        [ 1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [ 1,  1, -1, -1, -1, -1, -1],
        [ 1,  1, -1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1,  1, -1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [-1, -1, -1, -1, -1,  1,  1],
        [-1, -1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ],
    6: [
        [-1, -1,  1,  1,  1,  1, -1],
        [-1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1,  1, -1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1,  1, -1, -1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ],
    7: [
        [ 1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [-1, -1, -1, -1,  1,  1,  1],
        [-1, -1, -1, -1,  1,  1, -1],
        [-1, -1, -1,  1,  1,  1, -1],
        [-1, -1, -1,  1,  1, -1, -1],
        [-1, -1,  1,  1,  1, -1, -1],
        [-1, -1,  1,  1, -1, -1, -1],
        [-1, -1,  1,  1, -1, -1, -1],
        [-1, -1,  1,  1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ],
    8: [
        [-1,  1,  1,  1,  1,  1, -1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [-1,  1,  1,  1,  1,  1, -1],
        [-1,  1,  1,  1,  1,  1, -1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ],
    9: [
        [-1,  1,  1,  1,  1,  1, -1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1, -1,  1,  1],
        [ 1,  1,  1, -1, -1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1,  1],
        [-1, -1, -1, -1, -1,  1,  1],
        [ 1,  1, -1, -1,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1, -1],
        [-1,  1,  1,  1,  1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ],
}

def get_lenet5_digit(digit, normalize=True):
    """
    Get LeNet-5 bitmap for a digit
    
    Args:
        digit: int 0-9
        normalize: if True, returns values in [-1, 1] range (LeNet-5 original)
                   if False, returns values in [0, 1] range
    
    Returns:
        numpy array of shape (12, 7)
    """
    bitmap = np.array(lenet5_bitmaps[digit], dtype=np.float32)
    
    if not normalize:
        # Convert from [-1, 1] to [0, 1]
        bitmap = (bitmap + 1) / 2
    
    return bitmap

def save_lenet5_bitmaps(scale=20):
    """Save each digit as a separate BMP file"""
    for digit in range(10):
        # Get bitmap in [0, 1] range
        bitmap = get_lenet5_digit(digit, normalize=False)
        
        # Convert to [0, 255] for image
        arr = (bitmap * 255).astype(np.uint8)
        
        # Create image (mode 'L' for grayscale)
        img = Image.fromarray(arr, mode='L')
        
        # Scale up for visibility
        img = img.resize((7*scale, 12*scale), Image.NEAREST)
        
        # Save as BMP
        img.save(f'lenet5_digit_{digit}.bmp')
        print(f"Saved lenet5_digit_{digit}.bmp")

def create_lenet5_digit_sheet(scale=20):
    """Create a single image with all LeNet-5 digits"""
    spacing = 2
    width = 10 * 7 + 9 * spacing  # 10 digits with spacing
    height = 12
    
    # Create blank canvas
    canvas = np.zeros((height, width), dtype=np.float32)
    
    # Place each digit
    for i in range(10):
        bitmap = get_lenet5_digit(i, normalize=False)
        x_offset = i * (7 + spacing)
        canvas[:, x_offset:x_offset+7] = bitmap
    
    # Convert to [0, 255]
    canvas = (canvas * 255).astype(np.uint8)
    
    # Create and scale image
    img = Image.fromarray(canvas, mode='L')
    img = img.resize((width*scale, height*scale), Image.NEAREST)
    img.save('lenet5_all_digits.bmp')
    print("Saved lenet5_all_digits.bmp")
    
    return img

def visualize_digit_ascii(digit):
    """Print ASCII representation of a digit"""
    bitmap = get_lenet5_digit(digit, normalize=True)
    print(f"\nDigit {digit} (7x12):")
    print("=" * 9)
    for row in bitmap:
        line = ""
        for val in row:
            if val > 0:  # foreground
                line += "██"
            else:  # background
                line += "  "
        print(line)
    print("=" * 9)

def get_all_digits_as_array():
    """
    Get all digits as a single numpy array
    
    Returns:
        numpy array of shape (10, 12, 7) normalized to [-1, 1]
    """
    digits = np.zeros((10, 12, 7), dtype=np.float32)
    for i in range(10):
        digits[i] = get_lenet5_digit(i, normalize=True)
    return digits

# Example usage
if __name__ == "__main__":
    # Save individual digit bitmaps
    print("Saving individual LeNet-5 digit bitmaps...")
    save_lenet5_bitmaps(scale=20)
    
    # Create a sheet with all digits
    print("\nCreating digit sheet...")
    create_lenet5_digit_sheet(scale=20)
    
    # Visualize digits in ASCII
    for digit in range(10):
        visualize_digit_ascii(digit)
    
    # Get all digits as array (useful for training)
    all_digits = get_all_digits_as_array()
    print(f"\nAll digits array shape: {all_digits.shape}")
    print(f"Value range: [{all_digits.min()}, {all_digits.max()}]")
    
    # Example: Access specific digit
    digit_5 = get_lenet5_digit(5, normalize=True)
    print(f"\nDigit 5 shape: {digit_5.shape}")
    print(f"Digit 5 values:\n{digit_5}")