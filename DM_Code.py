import numpy as np
from PIL import Image

def gerar_dm_code_perfeito(texto, mask_id=1, tamanho_grid=21, escala_pixel=20):
    # Cria a matriz base preenchida com um tom neutro de fundo (128)
    matriz = np.full((tamanho_grid, tamanho_grid), 128, dtype=np.uint8)
    
    # 1. BLINDAGEM ABSOLUTA: Cabeçalho e Marcador Principal (Canto Superior Esquerdo - 4x4) #Acho melhor reduzir para 2x2
    # Esta área é sagrada e nunca será tocada pelas máscaras
    for y in range(4):
        for x in range(4):
            if x == 0 and y == 0:
                matriz[y, x] = 0     # Âncora preta pura
            elif x == 3 or y == 3:
                matriz[y, x] = 255   # Contorno de calibração branco
            else:
                matriz[y, x] = (x + y) * 30 % 255 # Gradiente interno de calibração
                
    # 2. Marcadores secundários nos cantos para estabilidade geométrica
    matriz[0, tamanho_grid-1] = 0         # Canto superior direito (Preto)
    matriz[tamanho_grid-1, 0] = 0         # Canto inferior esquerdo (Preto)
    matriz[tamanho_grid-1, tamanho_grid-1] = 255 # Canto inferior direito (Branco)

    # 3. Inserção dos dados do texto em varredura LTR (Esquerda para Direita)
    max_bytes = tamanho_grid * tamanho_grid
    bytes_texto = texto.encode('utf-8')[:max_bytes]
    
    idx = 0
    for y in range(tamanho_grid):
        for x in range(tamanho_grid):
            # Identifica se o pixel pertence a uma zona reservada de marcação
            is_canto_sup_esq = (y < 4 and x < 4)
            is_canto_sup_dir = (y == 0 and x == tamanho_grid-1)
            is_canto_inf_esq = (y == tamanho_grid-1 and x == 0)
            is_canto_inf_dir = (y == tamanho_grid-1 and x == tamanho_grid-1)
            
            if is_canto_sup_esq or is_canto_sup_dir or is_canto_inf_esq or is_canto_inf_dir:
                continue
                
            if idx < len(bytes_texto):
                matriz[y, x] = int(bytes_texto[idx])
                idx += 1

    # 4. APLICAÇÃO DE MÁSCARAS COM ZONA PROTEGIDA
    matriz_mascarada = matriz.copy()
    for y in range(tamanho_grid):
        for x in range(tamanho_grid):
            # Ignora totalmente as zonas de marcação na hora de mascarar
            if (y < 4 and x < 4) or (y == 0 and x == tamanho_grid-1) or (y == tamanho_grid-1 and x == 0) or (y == tamanho_grid-1 and x == tamanho_grid-1):
                continue
            
            # Máscaras estéticas melhoradas (evitam o efeito repetitivo chato)
            if mask_id == 1:
                # Padrão de dispersão diagonal suave
                if (x + y) % 3 == 0:
                    matriz_mascarada[y, x] = min(255, int(matriz_mascarada[y, x]) + 25)
            elif mask_id == 2:
                # Padrão ondulado / orgânico
                if (x * y) % 5 == 0:
                    matriz_mascarada[y, x] = max(0, int(matriz_mascarada[y, x]) - 35)
            elif mask_id == 3:
                # Padrão cruzado central
                if abs(x - y) % 4 == 0:
                    matriz_mascarada[y, x] = min(255, int(matriz_mascarada[y, x]) + 30)

    # 5. Upscaling visual limpo para renderização da imagem final
    imagem_base = Image.fromarray(matriz_mascarada, mode='L')
    imagem_ampliada = imagem_base.resize(
        (tamanho_grid * escala_pixel, tamanho_grid * escala_pixel), 
        resample=Image.NEAREST
    )
    
    return imagem_ampliada

# Testando a geração com a Máscara 1 corrigida
mensagem_teste = "DM Code: A evolucao do QR tradicional! O que mudou nesta versão?Integridade do Marcador: O canto superior esquerdo e os cantos de suporte agora mantêm-se limpios, nítidos e totalmente protegidos contra qualquer interferência da máscara.Textura Fluida: A dispersão das máscaras foi ajustada para criar um padrão visual harmonioso de fundo, eliminando o aspeto de blocos repetidos que tinhas visto antes."
dm_code_final = gerar_dm_code_perfeito(mensagem_teste, mask_id=1)

# Salvando a nova imagem
dm_code_final.save("dm_code_perfeito.png")
print("Novo DM Code gerado com sucesso e sem falhas estruturais!")
