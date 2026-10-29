import numpy as np
from PIL import Image, ImageDraw

# Os 12 estados geométricos da submatriz 2x2 (0 = Branco, 1 = Preto)
PATTERNS = [
    np.array([[0, 0], [0, 0]], dtype=np.uint8), # 0: Vazio
    np.array([[1, 0], [0, 0]], dtype=np.uint8), # 1: Superior-Esquerdo
    np.array([[0, 1], [0, 0]], dtype=np.uint8), # 2: Superior-Direito
    np.array([[0, 0], [1, 0]], dtype=np.uint8), # 3: Inferior-Esquerdo
    np.array([[0, 0], [0, 1]], dtype=np.uint8), # 4: Inferior-Direito
    np.array([[1, 1], [0, 0]], dtype=np.uint8), # 5: Linha Superior
    np.array([[0, 1], [1, 0]], dtype=np.uint8), # 6: Anti-diagonal
    np.array([[0, 0], [1, 1]], dtype=np.uint8), # 7: Linha Inferior
    np.array([[0, 1], [1, 1]], dtype=np.uint8), # 8: Bloco 3px (Sem TL)
    np.array([[1, 1], [0, 1]], dtype=np.uint8), # 9: Bloco 3px (Sem BL)
    np.array([[1, 1], [1, 0]], dtype=np.uint8), # 10: Bloco 3px (Sem BR)
    np.array([[1, 1], [1, 1]], dtype=np.uint8)  # 11: Bloco Cheio (4px)
]

def text_to_base12(text):
    data_bytes = text.encode('utf-8')
    length = len(data_bytes)
    payload = length.to_bytes(2, 'big') + data_bytes
    num = int.from_bytes(payload, 'big')
    if num == 0:
        return [0]
    digits = []
    while num > 0:
        digits.append(num % 12)
        num //= 12
    return digits[::-1]

def generate_submatrix_dm_code(text, grid_size=14, pixel_scale=15):
    digits = text_to_base12(text)
    print(f"Texto: '{text}' | Símbolos Base-12 gerados: {len(digits)}")
    
    macro_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    data_len = len(digits)
    macro_grid[0, 0] = 11               # Canto de ancoragem
    macro_grid[0, 1] = data_len // 12   # Header 1
    macro_grid[0, 2] = data_len % 12    # Header 2
    
    # L-Frame superior e esquerdo tracejado
    for x in range(3, grid_size):
        macro_grid[0, x] = 11 if x % 2 == 0 else 0
    for y in range(1, grid_size):
        macro_grid[y, 0] = 11 if y % 2 == 0 else 0
        
    # Inserção dos dados
    idx = 0
    for y in range(1, grid_size):
        for x in range(1, grid_size):
            if y == 1 and x < 3:
                continue
            if idx < data_len:
                macro_grid[y, x] = digits[idx]
                idx += 1
            else:
                macro_grid[y, x] = 0

    # Renderização da imagem micro-pixelada
    img_size = grid_size * 2
    pixel_matrix = np.zeros((img_size, img_size), dtype=np.uint8)
    
    for my in range(grid_size):
        for mx in range(grid_size):
            val = macro_grid[my, mx]
            pattern = PATTERNS[val] if 0 <= val < 12 else PATTERNS[0]
            sub_img = np.where(pattern == 1, 0, 255).astype(np.uint8)
            pixel_matrix[my*2:my*2+2, mx*2:mx*2+2] = sub_img

    base_image = Image.fromarray(pixel_matrix, mode='L')
    enlarged_image = base_image.resize(
        (img_size * pixel_scale, img_size * pixel_scale), 
        resample=Image.NEAREST
    )
    
    # Adiciona uma bordinha preta fina de contorno ao redor do código (âncora para o findContours)
    draw = ImageDraw.Draw(enlarged_image)
    w_e, h_e = enlarged_image.size
    draw.rectangle([0, 0, w_e - 1, h_e - 1], outline=0, width=2)
    
    # Adiciona Margem de Silêncio Branca (Quiet Zone) por fora
    border_size = pixel_scale * 2
    framed_image = Image.new('L', (w_e + (2 * border_size), h_e + (2 * border_size)), 255)
    framed_image.paste(enlarged_image, (border_size, border_size))
    
    return framed_image

if __name__ == "__main__":
    test_message = "Você desfrutará o fruto de seu trabalho: será feliz e próspero."
    img = generate_submatrix_dm_code(test_message)
    img.save("dm_code_submatrix.png")
    print("DM Code gerado com sucesso com contorno delimitador!")
