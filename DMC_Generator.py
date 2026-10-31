import numpy as np
from PIL import Image, ImageDraw

# 16 formas geométricas básicas puras (4x4 pixels ou representadas em 2x2)
# Para Base-256, usamos o arranjo direto de 4 níveis em 4 sub-píxeis (4^4 = 256)
# Cada valor de 0 a 255 no byte da mensagem mapeia diretamente para uma macro-célula 2x2,
# onde cada sub-píxel assume um dos 4 níveis radiométricos (0=Branco, 1=Cinza Claro, 2=Cinza Escuro, 3=Preto).

def byte_to_submatrix(byte_val):
    """Converte um byte (0-255) em uma submatriz 2x2 de 4 níveis (0, 1, 2, 3)."""
    # Decompõe o byte de 8 bits em 4 valores de 2 bits cada
    p0 = (byte_val >> 6) & 0x03
    p1 = (byte_val >> 4) & 0x03
    p2 = (byte_val >> 2) & 0x03
    p3 = byte_val & 0x03
    return np.array([[p0, p1], [p2, p3]], dtype=np.uint8)

# Mapeamento de 4 níveis radiométricos para intensidade de cinza (0 a 255)
# 0 = Branco Puro (255), 1 = Cinza Claro (190), 2 = Cinza Escuro (90), 3 = Preto Puro (0)
LEVEL_TO_LUMA = {0: 255, 1: 190, 2: 90, 3: 0}

def generate_base256_dm_code(text, grid_size=14, pixel_scale=15):
    data_bytes = text.encode('utf-8')
    data_len = len(data_bytes)
    print(f"Texto: '{text}' | Bytes codificados: {data_len} (Capacidade máx: 167 bytes)")
    
    macro_grid = np.zeros((grid_size, grid_size, 2), dtype=np.int32) # Guarda [nivel_ou_byte]
    
    # Cabeçalho de Controle e Gabaritos de Calibração Radiométrica nos cantos
    # Célula (0,0): 2x2 Preto Puro (Nível 3) -> Calibração de Preto
    # Célula (1,1): 2x2 Cinza Escuro (Nível 2) -> Calibração de Cinza Escuro
    # Célula (1,2): 2x2 Cinza Claro (Nível 1) -> Calibração de Cinza Claro
    # Célula (0,2): 2x2 Branco Puro (Nível 0) -> Calibração de Branco
    
    # Preenchemos a matriz com os bytes da mensagem
    # Reservamos o L-Frame e o topo para sincronismo e cabeçalho de tamanho
    
    raw_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    # Header: Tamanho da mensagem armazenado em 2 bytes nas primeiras células de dados
    header_bytes = data_len.to_bytes(2, 'big')
    full_payload = header_bytes + data_bytes
    
    # Gabaritos fixos de referência nos cantos de calibração
    raw_grid[0, 0] = 255 # Preto total de referência
    raw_grid[1, 1] = 170 # Cinza referência 2
    raw_grid[1, 2] = 85  # Cinza referência 1
    
    # L-Frame superior e esquerdo tracejado estruturado
    for x in range(3, grid_size):
        raw_grid[0, x] = 255 if x % 2 == 0 else 0
    for y in range(1, grid_size):
        raw_grid[y, 0] = 255 if y % 2 == 0 else 0

    # Inserção direta dos bytes da mensagem nas células livres
    idx = 0
    for y in range(1, grid_size):
        for x in range(1, grid_size):
            if y == 1 and x < 3:
                continue
            if idx < len(full_payload):
                raw_grid[y, x] = full_payload[idx]
                idx += 1
            else:
                raw_grid[y, x] = 0

    # Renderização da imagem micro-pixelada (cada macro-célula vira 2x2 sub-píxeis)
    img_size = grid_size * 2
    pixel_matrix = np.zeros((img_size, img_size), dtype=np.uint8)
    
    for my in range(grid_size):
        for mx in range(grid_size):
            byte_val = raw_grid[my, mx]
            
            # Sobrescreve as células de gabarito visual para calibração perfeita
            if my == 0 and mx == 0:
                sub_mat = np.array([[3, 3], [3, 3]], dtype=np.uint8) # Preto (3)
            elif my == 1 and mx == 1:
                sub_mat = np.array([[2, 2], [2, 2]], dtype=np.uint8) # Cinza Escuro (2)
            elif my == 1 and mx == 2:
                sub_mat = np.array([[1, 1], [1, 1]], dtype=np.uint8) # Cinza Claro (1)
            elif my == 0 and mx == 2:
                sub_mat = np.array([[0, 0], [0, 0]], dtype=np.uint8) # Branco (0)
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
    draw.rectangle([0, 0, w_e - 1, h_e - 1], outline=0, width=2)
    
    border_size = pixel_scale * 2
    framed_image = Image.new('L', (w_e + (2 * border_size), h_e + (2 * border_size)), 255)
    framed_image.paste(enlarged_image, (border_size, border_size))
    
    return framed_image

if __name__ == "__main__":
    test_message = "DM Code Base-256 Hibrido 4 Niveis Radiometricos!"
    img = generate_base256_dm_code(test_message)
    img.save("dm_code_submatrix.png")
    print("DM Code Base-256 (4 Níveis) gerado com sucesso!")
