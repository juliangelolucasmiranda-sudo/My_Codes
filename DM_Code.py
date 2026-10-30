import numpy as np
from PIL import Image, ImageDraw

def byte_to_submatrix(byte_val):
    p0 = (byte_val >> 6) & 0x03
    p1 = (byte_val >> 4) & 0x03
    p2 = (byte_val >> 2) & 0x03
    p3 = byte_val & 0x03
    return np.array([[p0, p1], [p2, p3]], dtype=np.uint8)

# 0 = Branco Puro (255), 1 = Cinza Claro (175), 2 = Cinza Escuro (85), 3 = Preto Puro (0)
LEVEL_TO_LUMA = {0: 255, 1: 175, 2: 85, 3: 0}

def generate_base256_dm_code(text, pixel_scale=14):
    data_bytes = text.encode('utf-8')
    data_len = len(data_bytes)
    
    # Cálculo Dinâmico da Versão do Grid:
    # Células livres na matriz = (g - 1)*(g - 1) - 2 (gabaritos em (1,1) e (1,2))
    # O cabeçalho ocupa 3 bytes (2 para tamanho + 1 para o grid_size)
    grid_size = 20
    for g in [10, 12, 14, 16, 18, 20]:
        max_data_bytes = ((g - 1) * (g - 1) - 2) - 3
        if data_len <= max_data_bytes:
            grid_size = g
            break
    else:
        max_possible = ((20 - 1) * (20 - 1) - 2) - 3
        data_bytes = data_bytes[:max_possible]
        data_len = len(data_bytes)
        grid_size = 20

    print(f"Texto: '{text}' | Bytes: {data_len} | Versão do Grid: {grid_size}x{grid_size}")
    
    raw_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    # Cabeçalho de 3 bytes: [Tamanho (2 bytes)] + [Grid Size (1 byte)]
    header_bytes = data_len.to_bytes(2, 'big') + bytes([grid_size])
    full_payload = header_bytes + data_bytes
    
    # L-Frame superior e esquerdo (Padrão de relógio óptico para auto-detecção do Grid)
    for x in range(3, grid_size):
        raw_grid[0, x] = 255 if x % 2 == 0 else 0
    for y in range(1, grid_size):
        raw_grid[y, 0] = 255 if y % 2 == 0 else 0

    # Inserção dos dados + Padding Pseudoaleatório (Camuflagem de Alta Entropia)
    idx = 0
    # Semente inicial para o gerador de ruído visual determinístico
    rng_state = 0x3C  
    
    for y in range(1, grid_size):
        for x in range(1, grid_size):
            if y == 1 and x < 3:
                continue
            if idx < len(full_payload):
                raw_grid[y, x] = full_payload[idx]
                idx += 1
            else:
                # PADDING RANDOMIZADO: Gera um byte de ruído visual pseudoaleatório (0 a 255)
                # Utiliza um algoritmo de deslocamento de bits (LFSR/XOR Shift) leve e determinístico
                rng_state ^= (rng_state << 3) & 0xFF
                rng_state ^= (rng_state >> 5) & 0xFF
                rng_state ^= (rng_state << 4) & 0xFF
                raw_grid[y, x] = rng_state
    img_size = grid_size * 2
    pixel_matrix = np.zeros((img_size, img_size), dtype=np.uint8)
    
    for my in range(grid_size):
        for mx in range(grid_size):
            byte_val = raw_grid[my, mx]
            
            # Células de Gabarito Fixo para a Penta-Calibração
            if my == 0 and mx == 0:
                sub_mat = np.array([[3, 3], [3, 3]], dtype=np.uint8) # Preto (3)
            elif my == 0 and mx == 1:
                sub_mat = np.array([[3, 3], [3, 3]], dtype=np.uint8) # Preto (3)
            elif my == 0 and mx == 2:
                sub_mat = np.array([[0, 0], [0, 0]], dtype=np.uint8) # Branco (0)
            elif my == 1 and mx == 1:
                sub_mat = np.array([[2, 2], [2, 2]], dtype=np.uint8) # Cinza Escuro (2)
            elif my == 1 and mx == 2:
                sub_mat = np.array([[1, 1], [1, 1]], dtype=np.uint8) # Cinza Claro (1)
            else:
                sub_mat = byte_to_submatrix(byte_val)
                
            sub_img = np.vectorize(LEVEL_TO_LUMA.get)(sub_mat).astype(np.uint8)
            pixel_matrix[my*2:my*2+2, mx*2:mx*2+2] = sub_img

    base_image = Image.fromarray(pixel_matrix, mode='L')
    enlarged_image = base_image.resize(
        (img_size * pixel_scale, img_size * pixel_scale), 
        resample=Image.NEAREST
    )
    
    draw = ImageDraw.Draw(enlarged_image)
    w_e, h_e = enlarged_image.size
    draw.rectangle([0, 0, w_e - 1, h_e - 1], outline=0, width=3)
    
    border_size = pixel_scale * 2
    framed_image = Image.new('L', (w_e + (2 * border_size), h_e + (2 * border_size)), 255)
    framed_image.paste(enlarged_image, (border_size, border_size))
    
    return framed_image

if __name__ == "__main__":
    test_message = "A presente proposta de inovação tem como objetivo principal a aplicação da lógica multivalorada para incrementar a capacidade de armazenamento de dados por área de um código visual matricial, proporcionando um avança na tecnologia existente."
    img = generate_base256_dm_code(test_message)
    img.save("dm_code_submatrix.png")
    print("DM Code Dinâmico gerado com sucesso!")
