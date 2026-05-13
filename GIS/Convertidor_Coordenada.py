import math
from typing import List, Tuple

def sirgas_str_to_wgs84(s: str) -> List[Tuple[float, float]]:
    """
    Convierte todas las parejas x|y de una cadena del tipo:
      '455757.082701439|291000.997510888|455764.382574587|291001.058343164|EPSG:19345'
    a coordenadas (lat, lon) WGS84 en grados decimales, usando la misma
    proyección (LCC 2SP) y constantes del script de Excel.
    
    Retorna: list[tuple[float, float]] -> [(lat, lon), ...]
    """
    # --- Parseo de la cadena ---
    parts = [p.strip() for p in s.split("|") if p.strip()]
    if not parts or not parts[-1].upper().startswith("EPSG:"):
        raise ValueError("La cadena debe finalizar con 'EPSG:xxxx'.")
    epsg = parts[-1].upper()
    if epsg != "EPSG:19345":
        raise ValueError(f"EPSG no soportado: {epsg}. Solo se acepta EPSG:19345.")

    # Convierte todos los tokens numéricos previos a float
    nums = []
    for tok in parts[:-1]:
        try:
            nums.append(float(tok))
        except ValueError:
            raise ValueError(f"Valor no numérico detectado: '{tok}'")

    if len(nums) % 2 != 0:
        raise ValueError("Cantidad de números impar: debe haber pares x|y.")

    # --- Constantes/proyección (idénticas a tu script) ---
    a = 6378137.0
    f = 0.003352811
    phi1 = 13.31763267
    phi2 = 14.25096601
    phi0 = 13.78429934
    lambda0 = -88.99998297
    N0 = 295809.184
    E0 = 500000.0
    F_scale = 1.0  # factor de escala (F)

    def to_rad(grad: float) -> float:
        return grad * math.pi / 180.0

    e = math.sqrt(2 * f - f ** 2)

    # m(φ) y t(φ) como en tu código
    def m(phi_deg: float) -> float:
        phi = to_rad(phi_deg)
        return math.cos(phi) / math.sqrt(1 - (e ** 2) * (math.sin(phi) ** 2))

    def t(phi_deg: float) -> float:
        phi = to_rad(phi_deg)
        return math.tan(math.pi / 4 - phi / 2) / ((1 - e * math.sin(phi)) / (1 + e * math.sin(phi))) ** (e / 2)

    m1, m2 = m(phi1), m(phi2)
    t1, t2, t0 = t(phi1), t(phi2), t(phi0)

    n = (math.log(m1) - math.log(m2)) / (math.log(t1) - math.log(t2))
    F2 = m1 / (n * (t1 ** n))
    Ro = a * F2 * (t0 ** n)

    results = []
    # Recorre cada par (x, y)
    for i in range(0, len(nums), 2):
        x, y = nums[i], nums[i + 1]

        # Aplica factor de escala (aunque es 1.0, se deja por fidelidad)
        x = x / F_scale
        y = y / F_scale

        NP = y - N0
        EP = x - E0

        RoP = math.sqrt(EP ** 2 + (Ro - NP) ** 2)
        if n < 0:
            RoP = -RoP

        tp = (RoP / (a * F2)) ** (1 / n)
        gammap = math.degrees(math.atan2(EP, (Ro - NP)))  # atan2 para robustez
        lon = gammap / n + lambda0

        # Iteraciones de latitud como en tu script (5 iteraciones)
        lat1 = math.pi / 2 - 2 * math.atan(tp)
        lat2 = math.pi / 2 - 2 * math.atan(tp * ((1 - e * math.sin(lat1)) / (1 + e * math.sin(lat1))) ** (e / 2))
        lat3 = math.pi / 2 - 2 * math.atan(tp * ((1 - e * math.sin(lat2)) / (1 + e * math.sin(lat2))) ** (e / 2))
        lat4 = math.pi / 2 - 2 * math.atan(tp * ((1 - e * math.sin(lat3)) / (1 + e * math.sin(lat3))) ** (e / 2))
        lat5 = math.pi / 2 - 2 * math.atan(tp * ((1 - e * math.sin(lat4)) / (1 + e * math.sin(lat4))) ** (e / 2))
        lat = math.degrees(lat5)

        results.append((lat, lon))

    return results


# Ejemplo 1: Un solo punto
coord_single = '469359.139557696|284016.568715262|469359.139557696|284016.568715262|EPSG:19345'
result = sirgas_str_to_wgs84(coord_single)
print(f"Latitud: {result[0][0]:.8f}, Longitud: {result[0][1]:.8f}")
