"""
Cartografía SRTM
================

Carga, combinación y consulta de tiles SRTM (Shuttle Radar Topography Mission).

Este módulo es agnóstico del modelo de propagación: solo provee elevación y
perfiles de terreno. Toda la matemática (haversine, parseo de tiles) vive en
operaciones.py.

Soporta dos modos de entrada en el directorio de trabajo:
1. Archivos .hgt individuales (uno por tile).
2. Un único SRTM.zip que contenga los .hgt o .hgt.zip anidados.

Si ambos existen, se prioriza SRTM.zip.

Diagnóstico de fallas en este módulo:
- "No se encontraron datos SRTM": no hay .hgt ni SRTM.zip en el directorio.
- Mapa con regiones planas a 0 m: los tiles correspondientes no se cargaron
  (verifica los nombres en TILES_EL_SALVADOR).
- Elevación incorrecta en una coordenada: revisa _latlon_a_pixel y meta.
"""

import numpy as np
import zipfile
from io import BytesIO
from pathlib import Path

from operaciones import distancia_haversine, parsear_nombre_tile


# =============================================================================
# TILES POR DEFECTO (El Salvador)
# =============================================================================
# Para otra región, sobreescribe esta lista o pásala al constructor en una
# futura versión paramétrica.

TILES_EL_SALVADOR = [
    'N13W090', 'N13W089', 'N13W088',
    'N14W090', 'N14W089', 'N14W088'
]


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class Cartografia:
    """
    Carga tiles SRTM, los combina en un único mapa de elevación y permite
    consultar elevación puntual y perfiles de terreno entre dos coordenadas.
    
    Uso típico
    ----------
    >>> carto = Cartografia()                          # carga SRTM
    >>> z = carto.elevacion(13.7, -89.2)               # elevación puntual
    >>> dists, elevs = carto.perfil_terreno(           # perfil entre A y B
    ...     lat1, lon1, lat2, lon2, num_puntos=500
    ... )
    """
    
    def __init__(self, directorio=None):
        """
        Parámetros
        ----------
        directorio : str o Path, opcional
            Carpeta donde buscar los SRTM. Por defecto: la carpeta del módulo.
        """
        self.directorio = Path(directorio) if directorio else Path(__file__).parent
        self.mapa = None    # np.ndarray combinado
        self.meta = None    # dict con bounding box, resolución, etc.
        self._cargar_cartografia()
    
    # -------------------------------------------------------------------------
    # Carga de datos
    # -------------------------------------------------------------------------
    
    def _cargar_cartografia(self):
        """Decide la fuente (zip o hgt sueltos) y combina los tiles."""
        tiles_data = {}
        
        zip_path = self.directorio / "SRTM.zip"
        if zip_path.exists():
            tiles_data = self._cargar_desde_zip(zip_path)
        else:
            tiles_data = self._cargar_desde_hgt()
        
        if not tiles_data:
            raise FileNotFoundError("No se encontraron datos SRTM")
        
        self._combinar_tiles(tiles_data)
    
    def _cargar_desde_hgt(self):
        """Carga tiles desde archivos .hgt sueltos en el directorio."""
        tiles = {}
        for nombre in TILES_EL_SALVADOR:
            for ext in ['.hgt', '.HGT']:
                archivo = self.directorio / f"{nombre}{ext}"
                if archivo.exists():
                    data = self._leer_hgt(archivo)
                    lat, lon = parsear_nombre_tile(nombre)
                    tiles[(lat, lon)] = data
                    print(f"  ✓ {nombre} ({data.shape[0]}x{data.shape[0]})")
                    break
        return tiles
    
    def _cargar_desde_zip(self, zip_path):
        """Carga tiles desde un SRTM.zip (con .hgt o .hgt.zip anidados)."""
        tiles = {}
        print(f"Cargando desde: {zip_path.name}")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            archivos = zf.namelist()
            
            for nombre in TILES_EL_SALVADOR:
                encontrado = None
                for arch in archivos:
                    if nombre.lower() in arch.lower() and arch.lower().endswith(('.hgt', '.hgt.zip')):
                        encontrado = arch
                        break
                
                if not encontrado:
                    continue
                
                contenido = zf.read(encontrado)
                
                # Si es .hgt.zip anidado, descomprimir
                if encontrado.endswith('.zip'):
                    with zipfile.ZipFile(BytesIO(contenido)) as nested:
                        for f in nested.namelist():
                            if f.lower().endswith('.hgt'):
                                contenido = nested.read(f)
                                break
                
                # Decodificar bytes a matriz de elevación
                data = np.frombuffer(contenido, dtype='>i2')
                size = int(np.sqrt(len(data)))
                data = data.reshape((size, size))
                data = np.where(data == -32768, 0, data)  # void → 0
                
                lat, lon = parsear_nombre_tile(nombre)
                tiles[(lat, lon)] = data
                res = "30m" if size == 3601 else "90m"
                print(f"  ✓ {nombre} ({size}x{size}, {res})")
        
        return tiles
    
    def _leer_hgt(self, archivo):
        """Lee un archivo .hgt y devuelve la matriz de elevación."""
        size = archivo.stat().st_size
        dim = 3601 if size == 3601 * 3601 * 2 else 1201
        with open(archivo, 'rb') as f:
            data = np.fromfile(f, dtype='>i2', count=dim*dim)
            data = data.reshape((dim, dim))
            data = np.where(data == -32768, 0, data)
        return data
    
    def _combinar_tiles(self, tiles_data):
        """Combina los tiles en un único mapa contiguo + guarda metadatos."""
        lats = sorted(set(k[0] for k in tiles_data.keys()))
        lons = sorted(set(k[1] for k in tiles_data.keys()))
        
        sample = next(iter(tiles_data.values()))
        tile_size = sample.shape[0]
        
        n_filas = len(lats) * tile_size
        n_cols = len(lons) * tile_size
        
        self.mapa = np.zeros((n_filas, n_cols), dtype=np.int16)
        
        for (lat, lon), data in tiles_data.items():
            fila = (max(lats) - lat) * tile_size
            col = (lon - min(lons)) * tile_size
            self.mapa[fila:fila+tile_size, col:col+tile_size] = data
        
        self.meta = {
            'lat_max': max(lats) + 1,
            'lat_min': min(lats),
            'lon_max': max(lons) + 1,
            'lon_min': min(lons),
            'filas': n_filas,
            'cols': n_cols,
            'tile_size': tile_size,
            'resolucion_m': 30 if tile_size == 3601 else 90
        }
        
        print(f"\n✓ Mapa: {n_filas}x{n_cols} px | Resolución: {self.meta['resolucion_m']}m")
    
    # -------------------------------------------------------------------------
    # Consultas
    # -------------------------------------------------------------------------
    
    def _latlon_a_pixel(self, lat, lon):
        """Convierte (lat, lon) a (fila, col) en el mapa combinado."""
        lat_norm = (self.meta['lat_max'] - lat) / (self.meta['lat_max'] - self.meta['lat_min'])
        lon_norm = (lon - self.meta['lon_min']) / (self.meta['lon_max'] - self.meta['lon_min'])
        
        fila = int(lat_norm * self.meta['filas'])
        col = int(lon_norm * self.meta['cols'])
        
        fila = max(0, min(self.meta['filas'] - 1, fila))
        col = max(0, min(self.meta['cols'] - 1, col))
        
        return fila, col
    
    def elevacion(self, lat, lon):
        """Elevación (m) en una coordenada (lat, lon)."""
        fila, col = self._latlon_a_pixel(lat, lon)
        return float(self.mapa[fila, col])
    
    def perfil_terreno(self, lat1, lon1, lat2, lon2, num_puntos=500):
        """
        Perfil de elevación a lo largo del segmento A→B.
        
        Parámetros
        ----------
        lat1, lon1, lat2, lon2 : float
            Coordenadas WGS-84 de los puntos A y B.
        num_puntos : int
            Número de muestras a lo largo del segmento (500 por defecto).
        
        Retorna
        -------
        distancias : np.ndarray (km), de 0 a distancia_total.
        elevaciones : np.ndarray (m), elevación del terreno en cada muestra.
        """
        f1, c1 = self._latlon_a_pixel(lat1, lon1)
        f2, c2 = self._latlon_a_pixel(lat2, lon2)
        
        filas = np.linspace(f1, f2, num_puntos).astype(int)
        cols = np.linspace(c1, c2, num_puntos).astype(int)
        
        filas = np.clip(filas, 0, self.meta['filas'] - 1)
        cols = np.clip(cols, 0, self.meta['cols'] - 1)
        
        elevaciones = self.mapa[filas, cols].astype(float)
        distancia_total = distancia_haversine(lat1, lon1, lat2, lon2)
        distancias = np.linspace(0, distancia_total, num_puntos)
        
        return distancias, elevaciones


# =============================================================================
# PRUEBAS DIRECTAS
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("   MÓDULO CARTOGRAFÍA - SMOKE TEST")
    print("="*60)
    
    carto = Cartografia()
    print(f"\nMapa cargado: {carto.meta['filas']}x{carto.meta['cols']} px"
          f" @ {carto.meta['resolucion_m']}m")
    print(f"Bounding box: lat [{carto.meta['lat_min']}, {carto.meta['lat_max']}],"
          f" lon [{carto.meta['lon_min']}, {carto.meta['lon_max']}]")
    
    # Elevación de San Salvador (centro, ~659 m)
    z = carto.elevacion(13.6929, -89.2182)
    print(f"\nElevación San Salvador: {z:.0f} m")
