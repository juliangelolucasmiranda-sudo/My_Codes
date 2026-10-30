import cv2
import numpy as np

PATTERNS = [
    np.array([[0, 0], [0, 0]], dtype=np.uint8), # 0
    np.array([[1, 0], [0, 0]], dtype=np.uint8), # 1
    np.array([[0, 1], [0, 0]], dtype=np.uint8), # 2
    np.array([[0, 0], [1, 0]], dtype=np.uint8), # 3
    np.array([[0, 0], [0, 1]], dtype=np.uint8), # 4
    np.array([[1, 1], [0, 0]], dtype=np.uint8), # 5
    np.array([[0, 1], [1, 0]], dtype=np.uint8), # 6
    np.array([[0, 0], [1, 1]], dtype=np.uint8), # 7
    np.array([[0, 1], [1, 1]], dtype=np.uint8), # 8
    np.array([[1, 1], [0, 1]], dtype=np.uint8), # 9
    np.array([[1, 1], [1, 0]], dtype=np.uint8), # 10
    np.array([[1, 1], [1, 1]], dtype=np.uint8)  # 11
]

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def match_pattern(sub_bin):
    best_match = 0
    min_diff = 999
    for idx, pat in enumerate(PATTERNS):
        diff = int(np.sum(np.abs(sub_bin.astype(np.int32) - pat.astype(np.int32))))
        if diff < min_diff:
            min_diff = diff
            best_match = idx
    return best_match

def base12_to_text(digits):
    try:
        if not digits:
            return ""
        num = 0
        for d in digits:
            num = num * 12 + int(d)
        if num <= 0:
            return ""
            
        byte_len = (num.bit_length() + 7) // 8
        b_data = num.to_bytes(byte_len, 'big')
        
        # Como o tamanho da mensagem (< 256 bytes) tem 0x00 no primeiro byte dos 2 bytes de header,
        # o primeiro byte não-zero (b_data[0]) DEVE ser exatamente igual à quantidade de bytes restantes!
        if len(b_data) >= 2 and b_data[0] == len(b_data) - 1:
            text_bytes = b_data[1:]
            decoded = text_bytes.decode('utf-8') # Decodificação estrita (rejeita lixo binário)
            if all(c.isprintable() or c in '\n\r\t' for c in decoded):
                return decoded
        return ""
    except Exception:
        return ""

def decode_submatrix_matrix(warped_gray, grid_size=14):
    try:
        h, w = warped_gray.shape
        _, thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        total_sub = grid_size * 2
        sub_size = w // total_sub
        
        macro_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        debug_warp = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)

        for i in range(total_sub + 1):
            pos = min(i * sub_size, w - 1)
            line_color = (255, 180, 0) if i % 2 == 0 else (180, 100, 0)
            cv2.line(debug_warp, (pos, 0), (pos, h - 1), line_color, 1)
            cv2.line(debug_warp, (0, pos), (w - 1, pos), line_color, 1)

        for my in range(grid_size):
            for mx in range(grid_size):
                sub_bin = np.zeros((2, 2), dtype=np.uint8)
                for py in range(2):
                    for px in range(2):
                        ny = my * 2 + py
                        nx = mx * 2 + px
                        
                        center_y = ny * sub_size + (sub_size // 2)
                        center_x = nx * sub_size + (sub_size // 2)
                        
                        patch = thresh[center_y-2:center_y+3, center_x-2:center_x+3]
                        dark_pixels = np.sum(patch < 128)
                        
                        is_black = 1 if dark_pixels >= 13 else 0
                        sub_bin[py, px] = is_black
                        
                        dot_color = (0, 255, 0) if is_black else (0, 0, 255)
                        cv2.circle(debug_warp, (center_x, center_y), 2, dot_color, -1)
                
                macro_grid[my, mx] = match_pattern(sub_bin)

        cv2.imshow("Debug Grid Sampling", debug_warp)

        # Extrai todos os 167 símbolos disponíveis na ordem do fluxo
        all_digits = []
        for y in range(1, grid_size):
            for x in range(1, grid_size):
                if y == 1 and x < 3:
                    continue
                all_digits.append(int(macro_grid[y, x]))

        h1 = int(macro_grid[0, 1])
        h2 = int(macro_grid[0, 2])
        header_len = h1 * 12 + h2

        # Lista de candidatos inteligentes (inclui correção para quando data_len >= 144 zera o h1)
        candidate_lengths = []
        if 0 < header_len <= len(all_digits):
            candidate_lengths.append(header_len)
        if h1 == 0:
            if 144 + h2 <= len(all_digits):
                candidate_lengths.append(144 + h2) # Quando data_len // 12 == 12
            if 156 + h2 <= len(all_digits):
                candidate_lengths.append(156 + h2) # Quando data_len // 12 == 13 (ex: 166!)

        # Adiciona busca reversa a partir da última célula não-vazia como garantia total
        last_nonzero = len(all_digits)
        while last_nonzero > 0 and all_digits[last_nonzero - 1] == 0:
            last_nonzero -= 1
        for length_try in range(min(len(all_digits), last_nonzero + 2), max(0, last_nonzero - 1), -1):
            if length_try not in candidate_lengths:
                candidate_lengths.append(length_try)

        for data_len in candidate_lengths:
            decoded_text = base12_to_text(all_digits[:data_len])
            if decoded_text:
                return decoded_text, grid_size

        return "", grid_size
    except Exception:
        return "", grid_size

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Iniciando decodificador Base-12 (Suporte a Mensagens Longas)... Pressione 'q' para sair.")
    
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
        
        display_text = "Aguardando Submatrix Code..."
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
                
                text, gsz = decode_submatrix_matrix(warped)
                if text:
                    display_text = f"Texto: {text}"
                    color = (0, 255, 0)
                    print(f"[SUCESSO ABSOLUTO]: {text}")
                break
                
        cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.imshow("Submatrix Code Decoder", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()