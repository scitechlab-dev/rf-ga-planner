"""
Operaciones matemáticas y geométricas
======================================

Funciones puras (sin estado, sin I/O) de geometría terrestre y RF.

Este módulo es la capa más baja del proyecto: no depende de ningún otro módulo
del paquete. Lo consumen cartografia.py (para distancias y parseo de tiles),
linea_vista_fresnel.py (para corrección de curvatura y radio de Fresnel),
graficas.py (las mismas), y cualquier futuro modelo de propagación.

Diagnóstico de fallas en este módulo:
- Si las distancias salen mal: revisa distancia_haversine.
- Si los tiles SRTM no se ubican: revisa parsear_nombre_tile.
- Si los enlaces a gran distancia se ven "demasiado libres": revisa
  correccion_curvatura.
- Si los radios de Fresnel parecen 0 o NaN: revisa radio_fresnel.
"""

from math import radians, sin, cos, sqrt, atan2


# =============================================================================
# CONSTANTES FÍSICAS
# =============================================================================

RADIO_TIERRA_KM = 6371.0
VELOCIDAD_LUZ = 299792458  # m/s


# =============================================================================
# GEOMETRÍA TERRESTRE
# =============================================================================

def distancia_haversine(lat1, lon1, lat2, lon2):
    """
    Distancia en km entre dos coordenadas WGS-84 usando la fórmula de Haversine.
    
    Parámetros
    ----------
    lat1, lon1, lat2, lon2 : float
        Latitudes y longitudes en grados decimales.
    
    Retorna
    -------
    float : distancia en kilómetros.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * RADIO_TIERRA_KM * atan2(sqrt(a), sqrt(1-a))


def correccion_curvatura(d_km):
    """
    Descenso aparente (m) de un punto a distancia d_km respecto al plano
    tangente local, por efecto de la curvatura terrestre.
    
    Cada modelo de propagación decide si la aplica o no al perfil de terreno.
    
    Parámetros
    ----------
    d_km : float
        Distancia desde el origen en kilómetros.
    
    Retorna
    -------
    float : descenso aparente en metros.
    """
    return (d_km ** 2) / (2 * RADIO_TIERRA_KM) * 1000


def parsear_nombre_tile(nombre):
    """
    Extrae latitud y longitud (esquina inferior izquierda) de un nombre de tile
    SRTM con formato estándar ('N13W089', 'S05E120', etc.).
    
    Parámetros
    ----------
    nombre : str
        Nombre del tile, longitud >= 7 caracteres.
    
    Retorna
    -------
    tuple (int, int) : (latitud, longitud) en grados enteros, con signo.
    """
    lat = int(nombre[1:3])
    lon = -int(nombre[4:7])
    if nombre[0] == 'S':
        lat = -lat
    if nombre[3] == 'E':
        lon = -lon
    return lat, lon


# =============================================================================
# GEOMETRÍA RF
# =============================================================================

def radio_fresnel(d1_km, d2_km, frecuencia_mhz, n=1):
    """
    Radio de la n-ésima zona de Fresnel en un punto intermedio de un enlace.
    
    Parámetros
    ----------
    d1_km : float
        Distancia desde el punto A al punto de cálculo (km).
    d2_km : float
        Distancia desde el punto de cálculo al punto B (km).
    frecuencia_mhz : float
        Frecuencia del enlace (MHz).
    n : int, opcional
        Número de zona de Fresnel (1 por defecto).
    
    Retorna
    -------
    float : radio en metros (0 si d1 + d2 == 0).
    """
    d1_m = d1_km * 1000
    d2_m = d2_km * 1000
    d_total_m = d1_m + d2_m
    
    if d_total_m == 0:
        return 0
    
    longitud_onda = VELOCIDAD_LUZ / (frecuencia_mhz * 1e6)
    radio = sqrt((n * longitud_onda * d1_m * d2_m) / d_total_m)
    
    return radio


# =============================================================================
# PRUEBAS DIRECTAS
# =============================================================================

if __name__ == "__main__":
    # Smoke test rápido para validar funciones
    print("="*60)
    print("   MÓDULO OPERACIONES - SMOKE TEST")
    print("="*60)
    
    # San Salvador → Santa Ana (≈55 km)
    d = distancia_haversine(13.6929, -89.2182, 13.9942, -89.5592)
    print(f"\nDistancia SanSal-SantaAna: {d:.2f} km (esperado ≈55)")
    
    # Curvatura a 30 km (≈70 m)
    c = correccion_curvatura(30)
    print(f"Curvatura a 30 km: {c:.1f} m (esperado ≈70)")
    
    # Parseo de tile
    lat, lon = parsear_nombre_tile("N13W089")
    print(f"Tile N13W089: ({lat}, {lon}) (esperado (13, -89))")
    
    # Fresnel a mitad de un enlace de 30 km @ 900 MHz
    r = radio_fresnel(15, 15, 900)
    print(f"Fresnel 1ª zona @ mitad de 30km, 900MHz: {r:.1f} m (esperado ≈50)")
