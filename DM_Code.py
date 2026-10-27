import numpy as np
from PIL import Image

def calculate_dm_code_version(text_bytes_length):
    """
    Dynamically calculates the version and grid size based on text size.
    Reserved pixels: 4 pixels (2x2 header) + 5 pixels (White 'L' frame) + 3 vertex markers = 12 reserved pixels.
    """
    versions = [
        (1, 21, (21 * 21) - 12),  # Version 1: 21x21
        (2, 25, (25 * 25) - 12),  # Version 2: 25x25
        (3, 29, (29 * 29) - 12),  # Version 3: 29x29
        (4, 33, (33 * 33) - 12),  # Version 4: 33x33
        (5, 37, (37 * 37) - 12)   # Version 5: 37x37
    ]
    
    for version, grid_size, capacity in versions:
        if text_bytes_length <= capacity:
            return version, grid_size
            
    return 5, 37, (37 * 37) - 12

def generate_optimized_dm_code(text, mask_id=1, bits_mode=256, pixel_scale=20):
    text_bytes = text.encode('utf-8')
    bytes_length = len(text_bytes)
    
    # Auto-sizing based on the updated reserved limit
    version, grid_size = calculate_dm_code_version(bytes_length)
    print(f"Text: {bytes_length} bytes | Version: v{version} ({grid_size}x{grid_size}) | Mask: {mask_id}")
    
    # Base matrix filled with neutral background tone (128)
    matrix = np.full((grid_size, grid_size), 128, dtype=np.uint8)
    
    # 1. OPTIMIZED 2x2 HEADER (Top-Left Corner)
    matrix[0, 0] = 255                  # (0,0) White Anchor
    matrix[0, 1] = version * 50         # (0,1) Grid Version
    matrix[1, 0] = mask_id * 80         # (1,0) Mask ID
    matrix[1, 1] = 255 if bits_mode == 256 else 128  # (1,1) Data Depth / Bits Mode
    
    # 2. Drawing the White Inverted "L" Frame around the 2x2 header for visual isolation:
    for i in range(3):
        if i < grid_size:
            matrix[2, i] = 255          # Horizontal leg of the white L
            matrix[i, 2] = 255          # Vertical leg of the white L

    # 3. Secondary markers at the vertices
    matrix[0, grid_size - 1] = 0              # Top-right corner (Black)
    matrix[grid_size - 1, 0] = 0              # Bottom-left corner (Black)
    matrix[grid_size - 1, grid_size - 1] = 255  # Bottom-right corner (White)

    # Helper function to check if a coordinate belongs to any protected structural zone
    def is_protected_zone(y, x):
        is_top_left_block = (y < 3 and x < 3) # Covers 2x2 header + 3x3 white 'L' frame area
        is_top_right = (y == 0 and x == grid_size - 1)
        is_bottom_left = (y == grid_size - 1 and x == 0)
        is_bottom_right = (y == grid_size - 1 and x == grid_size - 1)
        return is_top_left_block or is_top_right or is_bottom_left or is_bottom_right

    # 4. LTR data insertion (strictly ignoring all protected zones)
    idx = 0
    for y in range(grid_size):
        for x in range(grid_size):
            if is_protected_zone(y, x):
                continue
                
            if idx < bytes_length:
                matrix[y, x] = int(text_bytes[idx])
                idx += 1

    # 5. MASK APPLICATION WITH STRICT PROTECTED ZONES
    masked_matrix = matrix.copy()
    for y in range(grid_size):
        for x in range(grid_size):
            if is_protected_zone(y, x):
                continue
            
            if mask_id == 1:
                if (x + y) % 3 == 0:
                    masked_matrix[y, x] = min(255, int(masked_matrix[y, x]) + 25)
            elif mask_id == 2:
                if (x * y) % 5 == 0:
                    masked_matrix[y, x] = max(0, int(masked_matrix[y, x]) - 35)
            elif mask_id == 3:
                if abs(x - y) % 4 == 0:
                    masked_matrix[y, x] = min(255, int(masked_matrix[y, x]) + 30)

    # 6. Final rendering with upscaling
    base_image = Image.fromarray(masked_matrix, mode='L')
    enlarged_image = base_image.resize(
        (grid_size * pixel_scale, grid_size * pixel_scale), 
        resample=Image.NEAREST
    )
    return enlarged_image

# Testing the optimized DM Code with the white "L" frame
test_message = "DM Code featuring the protected white L-frame layout!"
optimized_dm = generate_optimized_dm_code(test_message, mask_id=3)
optimized_dm.save("dm_code_optimized.png")
print("Optimized DM Code with L-frame generated successfully!")
