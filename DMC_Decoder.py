import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def sample_cell_luma(warped_gray, mx, my, sub_size):
    vals = []
    for py in range(2):
        for px in range(2):
            cy = (my * 2 + py) * sub_size + (sub_size // 2)
            cx = (mx * 2 + px) * sub_size + (sub_size // 2)
            patch = warped_gray[cy-2:cy+3, cx-2:cx+3]
            vals.append(float(np.median(patch)))
    return float(np.mean(vals))

def decode_base256_matrix(warped_gray, grid_size=14):
    try:
        h, w = warped_gray.shape
        total_sub = grid_size * 2
        sub_size = w // total_sub
        
        # 1. PENTA-CALIBRAÇÃO RADIOMÉTRICA DINÂMICA
        ref_black = sample_cell_luma(warped_gray, 0, 0, sub_size)   # Célula (0,0): Preto
        ref_dark  = sample_cell_luma(warped_gray, 1, 1, sub_size)   # Célula (1,1): Cinza Escuro
        ref_light = sample_cell_luma(warped_gray, 1, 2, sub_size)   # Célula (1,2): Cinza Claro
        ref_white = sample_cell_luma(warped_gray, 2, 0, sub_size)   # L-Frame / Branco referência
        
        # Define os limiares exatos entre os 4 níveis com margens adaptativas
        t_black_dark = (ref_black + ref_dark) / 2.0
        t_dark_light = (ref_dark + ref_light) / 2.0
        t_light_white = (ref_light + ref_white) / 2.0

        macro_bytes = np.zeros((grid_size, grid_size), dtype=np.int32)
        debug_warp = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)

        for i in range(total_sub + 1):
            pos = min(i * sub_size, w - 1)
            line_color = (255, 180, 0) if i % 2 == 0 else (180, 100, 0)
            cv2.line(debug_warp, (pos, 0), (pos, h - 1), line_color, 1)
            cv2.line(debug_warp, (0, pos), (w - 1, pos), line_color, 1)

        for my in range(grid_size):
            for mx in range(grid_size):
                p_vals = []
                for py in range(2):
                    for px in range(2):
                        ny = my * 2 + py
                        nx = mx * 2 + px
                        
                        center_y = ny * sub_size + (sub_size // 2)
                        center_x = nx * sub_size + (sub_size // 2)
                        
                        patch = warped_gray[center_y-2:center_y+3, center_x-2:center_x+3]
                        luma = float(np.median(patch))
                        
                        # Classifica em 4 níveis (0=Branco, 1=Claro, 2=Escuro, 3=Preto)
                        if luma < t_black_dark:
                            level = 3
                            dot_color = (0, 255, 0)   # Verde (Preto)
                        elif luma < t_dark_light:
                            level = 2
                            dot_color = (0, 255, 255) # Amarelo (Cinza Escuro)
                        elif luma < t_light_white:
                            level = 1
                            dot_color = (255, 0, 255) # Magenta (Cinza Claro)
                        else:
                            level = 0
                            dot_color = (0, 0, 255)   # Vermelho (Branco)
                            
                        p_vals.append(level)
                        cv2.circle(debug_warp, (center_x, center_y), 2, dot_color, -1)
                
                # Recompõe o byte de 8 bits a partir dos 4 sub-píxeis de 2 bits
                recomposed_byte = (p_vals[0] << 6) | (p_vals[1] << 4) | (p_vals[2] << 2) | p_vals[3]
                macro_bytes[my, mx] = recomposed_byte

        cv2.imshow("Debug Grid Sampling", debug_warp)

        # Extrai o fluxo sequencial de bytes ignorando o L-frame e cabeçalhos fixos
        byte_stream = bytearray()
        for y in range(1, grid_size):
            for x in range(1, grid_size):
                if y == 1 and x < 3:
                    continue
                byte_stream.append(int(macro_bytes[y, x]) & 0xFF)

        if len(byte_stream) >= 2:
            msg_len = int.from_bytes(byte_stream[:2], 'big')
            if 0 < msg_len <= len(byte_stream) - 2:
                text_bytes = byte_stream[2:2+msg_len]
                decoded = text_bytes.decode('utf-8', errors='ignore')
                if all(c.isprintable() or c in '\n\r\t' for c in decoded):
                    return decoded, grid_size

        return "", grid_size
    except Exception:
        return "", grid_size

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Iniciando decodificador Híbrido Base-256 (4 Níveis Radiométricos)... Pressione 'q' para sair.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        display_text = "Aguardando Submatrix Base-256..."
        color = (0, 0, 255)
        
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4 and cv2.contourArea(c) > 1000:
                target_cnt = approx.reshape(4, 2)
                rect = order_points(target_cnt)
                
                cv2.polylines(frame, [np.int32(rect)], True, (0, 255, 0), 3)
                
                dst_size = 420
                dst = np.array([
                    [0, 0],
                    [dst_size - 1, 0],
                    [dst_size - 1, dst_size - 1],
                    [0, dst_size - 1]], dtype="float32")
                    
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(gray, M, (dst_size, dst_size))
                
                text, gsz = decode_base256_matrix(warped)
                if text:
                    display_text = f"Texto: {text}"
                    color = (0, 255, 0)
                    print(f"[SUCESSO BASE-256]: {text}")
                break
                
        cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.imshow("Submatrix Code Decoder", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
