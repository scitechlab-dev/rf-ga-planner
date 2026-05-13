"""
Análisis de Línea de Vista con Fresnel - El Salvador
=====================================================
Incluye cálculo de zona de Fresnel y función objetivo unificada.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import zipfile
from io import BytesIO
from math import radians, sin, cos, sqrt, atan2


# =============================================================================
# CONSTANTES
# =============================================================================
RADIO_TIERRA_KM = 6371.0
VELOCIDAD_LUZ = 299792458  # m/s
TILES_EL_SALVADOR = [
    'N13W090', 'N13W089', 'N13W088',
    'N14W090', 'N14W089', 'N14W088'
]


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def distancia_haversine(lat1, lon1, lat2, lon2):
    """Distancia en km usando Haversine."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * RADIO_TIERRA_KM * atan2(sqrt(a), sqrt(1-a))


def correccion_curvatura(d_km):
    """Corrección por curvatura terrestre (metros)."""
    return (d_km ** 2) / (2 * RADIO_TIERRA_KM) * 1000


def parsear_nombre_tile(nombre):
    """Extrae lat/lon de nombre de tile."""
    lat = int(nombre[1:3])
    lon = -int(nombre[4:7])
    if nombre[0] == 'S':
        lat = -lat
    if nombre[3] == 'E':
        lon = -lon
    return lat, lon


def radio_fresnel(d1_km, d2_km, frecuencia_mhz, n=1):
    """
    Calcula el radio de la n-ésima zona de Fresnel.
    d1_km: distancia desde punto A al punto de cálculo (km)
    d2_km: distancia desde punto de cálculo a punto B (km)
    frecuencia_mhz: frecuencia en MHz
    n: número de zona de Fresnel (1 por defecto)
    Retorna: radio en metros
    """
    d1_m = d1_km * 1000
    d2_m = d2_km * 1000
    d_total_m = d1_m + d2_m
    
    if d_total_m == 0:
        return 0
    
    longitud_onda = VELOCIDAD_LUZ / (frecuencia_mhz * 1e6)  # metros
    radio = sqrt((n * longitud_onda * d1_m * d2_m) / d_total_m)
    
    return radio


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class LineaVista:
    
    def __init__(self, directorio=None):
        self.directorio = Path(directorio) if directorio else Path(__file__).parent
        self.mapa = None
        self.meta = None
        self._cargar_cartografia()
    
    def _cargar_cartografia(self):
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
                
                if encontrado.endswith('.zip'):
                    with zipfile.ZipFile(BytesIO(contenido)) as nested:
                        for f in nested.namelist():
                            if f.lower().endswith('.hgt'):
                                contenido = nested.read(f)
                                break
                
                data = np.frombuffer(contenido, dtype='>i2')
                size = int(np.sqrt(len(data)))
                data = data.reshape((size, size))
                data = np.where(data == -32768, 0, data)
                
                lat, lon = parsear_nombre_tile(nombre)
                tiles[(lat, lon)] = data
                res = "30m" if size == 3601 else "90m"
                print(f"  ✓ {nombre} ({size}x{size}, {res})")
        
        return tiles
    
    def _leer_hgt(self, archivo):
        size = archivo.stat().st_size
        dim = 3601 if size == 3601 * 3601 * 2 else 1201
        with open(archivo, 'rb') as f:
            data = np.fromfile(f, dtype='>i2', count=dim*dim)
            data = data.reshape((dim, dim))
            data = np.where(data == -32768, 0, data)
        return data
    
    def _combinar_tiles(self, tiles_data):
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
    
    def _latlon_a_pixel(self, lat, lon):
        lat_norm = (self.meta['lat_max'] - lat) / (self.meta['lat_max'] - self.meta['lat_min'])
        lon_norm = (lon - self.meta['lon_min']) / (self.meta['lon_max'] - self.meta['lon_min'])
        
        fila = int(lat_norm * self.meta['filas'])
        col = int(lon_norm * self.meta['cols'])
        
        fila = max(0, min(self.meta['filas'] - 1, fila))
        col = max(0, min(self.meta['cols'] - 1, col))
        
        return fila, col
    
    def elevacion(self, lat, lon):
        fila, col = self._latlon_a_pixel(lat, lon)
        return float(self.mapa[fila, col])
    
    def perfil_terreno(self, lat1, lon1, lat2, lon2, num_puntos=500):
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
    
    def verificar(self, lat1, lon1, altura1, lat2, lon2, altura2, 
                  frecuencia_mhz=900, curvatura=True):
        """Verifica línea de vista y despeje de zona de Fresnel."""
        distancias, elevaciones = self.perfil_terreno(lat1, lon1, lat2, lon2)
        
        if curvatura:
            correcciones = np.array([correccion_curvatura(d) for d in distancias])
            elev_efectiva = elevaciones - correcciones
        else:
            elev_efectiva = elevaciones.copy()
        
        h_a = elev_efectiva[0] + altura1
        h_b = elev_efectiva[-1] + altura2
        d_total = distancias[-1]
        
        linea_vista = np.linspace(h_a, h_b, len(distancias))
        
        # Calcular radio de Fresnel en cada punto
        radios_fresnel = np.array([
            radio_fresnel(d, d_total - d, frecuencia_mhz) 
            for d in distancias
        ])
        
        # Zona inferior de Fresnel (la crítica para obstrucciones)
        fresnel_inferior = linea_vista - radios_fresnel
        
        # Margen respecto a línea de vista
        margen_los = linea_vista - elev_efectiva
        
        # Margen respecto a 60% de Fresnel (criterio estándar RF)
        margen_fresnel_60 = (linea_vista - 0.6 * radios_fresnel) - elev_efectiva
        
        # Margen respecto a 100% de Fresnel
        margen_fresnel_100 = fresnel_inferior - elev_efectiva
        
        # Análisis (excluyendo extremos)
        margen_los_min = np.min(margen_los[1:-1])
        margen_fresnel_60_min = np.min(margen_fresnel_60[1:-1])
        margen_fresnel_100_min = np.min(margen_fresnel_100[1:-1])
        
        # Determinar estado
        if margen_los_min < 0:
            estado = 'obstruido'
            idx_obs = np.argmin(margen_los[1:-1]) + 1
            obstruccion_km = distancias[idx_obs]
        elif margen_fresnel_60_min < 0:
            estado = 'fresnel_60_obstruido'
            idx_obs = np.argmin(margen_fresnel_60[1:-1]) + 1
            obstruccion_km = distancias[idx_obs]
        elif margen_fresnel_100_min < 0:
            estado = 'fresnel_parcial'
            idx_obs = np.argmin(margen_fresnel_100[1:-1]) + 1
            obstruccion_km = distancias[idx_obs]
        else:
            estado = 'libre'
            obstruccion_km = None
        
        return {
            'estado': estado,
            'libre': estado == 'libre',
            'fresnel_60_libre': margen_fresnel_60_min >= 0,
            'distancia_km': d_total,
            'obstruccion_km': obstruccion_km,
            'margen_los_m': margen_los_min,
            'margen_fresnel_60_m': margen_fresnel_60_min,
            'margen_fresnel_100_m': margen_fresnel_100_min,
            'frecuencia_mhz': frecuencia_mhz
        }
    
    def graficar(self, lat1, lon1, altura1, lat2, lon2, altura2,
                 frecuencia_mhz=900, curvatura=True, 
                 guardar="perfil_linea_vista.png",
                 nombre_a="Punto A", nombre_b="Punto B"):
        """Genera y GUARDA el gráfico con zona de Fresnel."""
        
        distancias, elevaciones = self.perfil_terreno(lat1, lon1, lat2, lon2)
        
        if curvatura:
            correcciones = np.array([correccion_curvatura(d) for d in distancias])
            elev_efectiva = elevaciones - correcciones
        else:
            elev_efectiva = elevaciones.copy()
        
        h_a = elev_efectiva[0] + altura1
        h_b = elev_efectiva[-1] + altura2
        d_total = distancias[-1]
        
        linea_vista = np.linspace(h_a, h_b, len(distancias))
        
        # Calcular zona de Fresnel
        radios_fresnel = np.array([
            radio_fresnel(d, d_total - d, frecuencia_mhz) 
            for d in distancias
        ])
        
        fresnel_superior = linea_vista + radios_fresnel
        fresnel_inferior = linea_vista - radios_fresnel
        fresnel_60_superior = linea_vista + 0.6 * radios_fresnel
        fresnel_60_inferior = linea_vista - 0.6 * radios_fresnel
        
        resultado = self.verificar(lat1, lon1, altura1, lat2, lon2, altura2, 
                                   frecuencia_mhz, curvatura)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Terreno relleno
        ax.fill_between(distancias, 0, elev_efectiva, color='#8B4513', alpha=0.4)
        ax.plot(distancias, elev_efectiva, color='#654321', linewidth=2, label='Terreno')
        
        # Zona de Fresnel 100% (más clara)
        ax.fill_between(distancias, fresnel_inferior, fresnel_superior,
                        color='#87CEEB', alpha=0.3, label='Fresnel 100%')
        
        # Zona de Fresnel 60% (más oscura)
        ax.fill_between(distancias, fresnel_60_inferior, fresnel_60_superior,
                        color='#4169E1', alpha=0.3, label='Fresnel 60%')
        
        # Línea de vista
        if resultado['estado'] == 'libre':
            color_los = '#228B22'
            estilo = '-'
        elif resultado['estado'] == 'fresnel_parcial':
            color_los = '#FFA500'
            estilo = '-'
        elif resultado['estado'] == 'fresnel_60_obstruido':
            color_los = '#FF8C00'
            estilo = '--'
        else:
            color_los = '#DC143C'
            estilo = '--'
        
        ax.plot(distancias, linea_vista, color=color_los, linewidth=2.5,
                linestyle=estilo, label='Línea de vista')
        
        # Puntos A y B
        ax.plot(0, h_a, 'b^', markersize=14, label=f'{nombre_a} ({altura1}m)', zorder=5)
        ax.plot(distancias[-1], h_b, 'rv', markersize=14, label=f'{nombre_b} ({altura2}m)', zorder=5)
        
        # Torres/antenas
        ax.plot([0, 0], [elev_efectiva[0], h_a], 'b-', linewidth=3, alpha=0.7)
        ax.plot([distancias[-1], distancias[-1]], [elev_efectiva[-1], h_b], 'r-', linewidth=3, alpha=0.7)
        
        # Marcar obstrucción si existe
        if resultado['obstruccion_km'] is not None:
            idx = np.argmin(np.abs(distancias - resultado['obstruccion_km']))
            ax.plot(distancias[idx], elev_efectiva[idx], 'rX',
                    markersize=18, markeredgewidth=3, label='Obstrucción', zorder=6)
            ax.axvline(distancias[idx], color='red', linestyle=':', alpha=0.4)
        
        # Configuración
        ax.set_xlabel('Distancia (km)', fontsize=12)
        ax.set_ylabel('Elevación (m)', fontsize=12)
        
        # Título según estado
        titulos = {
            'libre': ("✓ ENLACE LIBRE (Fresnel 100%)", 'green'),
            'fresnel_parcial': ("◐ FRESNEL PARCIAL (>60%)", 'orange'),
            'fresnel_60_obstruido': ("⚠ FRESNEL 60% OBSTRUIDO", 'darkorange'),
            'obstruido': ("✗ LÍNEA DE VISTA OBSTRUIDA", 'red')
        }
        titulo, color_titulo = titulos[resultado['estado']]
        ax.set_title(titulo, fontsize=16, fontweight='bold', color=color_titulo)
        
        # Cuadro de información
        info = f"═══ ANÁLISIS RF ═══\n"
        info += f"Frecuencia: {frecuencia_mhz} MHz\n"
        info += f"Distancia: {resultado['distancia_km']:.2f} km\n"
        info += f"─────────────────\n"
        info += f"{nombre_a}:\n"
        info += f"  Terreno: {elev_efectiva[0]:.0f}m\n"
        info += f"  Antena: +{altura1}m\n"
        info += f"  Total: {h_a:.0f}m\n"
        info += f"─────────────────\n"
        info += f"{nombre_b}:\n"
        info += f"  Terreno: {elev_efectiva[-1]:.0f}m\n"
        info += f"  Antena: +{altura2}m\n"
        info += f"  Total: {h_b:.0f}m\n"
        info += f"─────────────────\n"
        info += f"Margen LOS: {resultado['margen_los_m']:.1f}m\n"
        info += f"Margen F60%: {resultado['margen_fresnel_60_m']:.1f}m\n"
        info += f"Margen F100%: {resultado['margen_fresnel_100_m']:.1f}m"
        
        ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                         edgecolor='gray', alpha=0.95))
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(guardar, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Gráfico guardado: {guardar}")
        plt.close(fig)
        
        return resultado
    
    def funcion_objetivo_rf(self, csv_path, indice_a, indice_b,
                            frecuencia_mhz=900,
                            altura_ied=8,
                            altura_concentrador=30,
                            graficar=False):
        """
        Función objetivo unificada para análisis RF.
        
        Parámetros:
        -----------
        csv_path : str
            Ruta al archivo CSV con columnas: Placa, Latitud, Longitud
        indice_a : int
            Índice del punto A (concentrador candidato)
        indice_b : int
            Índice del punto B (IED)
        frecuencia_mhz : float
            Frecuencia de operación (default: 900)
        altura_ied : float
            Altura de antena IED en metros (default: 8)
        altura_concentrador : float
            Altura de antena concentrador en metros (default: 30)
        graficar : bool
            Si True, genera imagen PNG (default: False)
        
        Retorna:
        --------
        dict con resultados del análisis
        """
        # Cargar CSV
        df = pd.read_csv(csv_path)
        
        # Validar índices
        n = len(df)
        if indice_a < 0 or indice_a >= n:
            raise ValueError(f"indice_a={indice_a} fuera de rango [0, {n-1}]")
        if indice_b < 0 or indice_b >= n:
            raise ValueError(f"indice_b={indice_b} fuera de rango [0, {n-1}]")
        
        # Extraer datos
        punto_a = df.iloc[indice_a]
        punto_b = df.iloc[indice_b]
        
        placa_a = str(punto_a['Placa'])
        placa_b = str(punto_b['Placa'])
        lat1, lon1 = float(punto_a['Latitud']), float(punto_a['Longitud'])
        lat2, lon2 = float(punto_b['Latitud']), float(punto_b['Longitud'])
        
        # Verificar
        resultado = self.verificar(
            lat1, lon1, altura_concentrador,
            lat2, lon2, altura_ied,
            frecuencia_mhz=frecuencia_mhz
        )
        
        # Agregar info de placas
        resultado['placa_a'] = placa_a
        resultado['placa_b'] = placa_b
        resultado['indice_a'] = indice_a
        resultado['indice_b'] = indice_b
        
        # Graficar si se solicita
        if graficar:
            nombre_archivo = f"enlace_{placa_a}_{placa_b}.png"
            self.graficar(
                lat1, lon1, altura_concentrador,
                lat2, lon2, altura_ied,
                frecuencia_mhz=frecuencia_mhz,
                guardar=nombre_archivo,
                nombre_a=f"{placa_a} (Conc)",
                nombre_b=f"{placa_b} (IED)"
            )
            resultado['archivo_grafico'] = nombre_archivo
        
        return resultado


# =============================================================================
# GENERADOR KML
# =============================================================================

def generar_kml_agrupado(datos_red: dict, archivo_salida: str = "red_rf.kml"):
    """
    Genera archivo KML con estructura AGRUPADA por concentrador.
    
    Estructura de carpetas:
    - Carpeta por cada concentrador (con sus IEDs y enlaces)
    - Carpeta separada para IEDs sin cobertura
    
    Parámetros:
    -----------
    datos_red : dict
        Estructura:
        {
            'concentradores': {
                'PLACA_CONC': {
                    'lat': float,
                    'lon': float,
                    'tipo': str,
                    'es_fijo': bool,
                    'ieds': [
                        {
                            'placa': str,
                            'lat': float,
                            'lon': float,
                            'tipo': str,
                            'distancia_km': float,
                            'margen_fresnel_60': float
                        },
                        ...
                    ]
                },
                ...
            },
            'sin_cobertura': [
                {
                    'placa': str,
                    'lat': float,
                    'lon': float,
                    'tipo': str
                },
                ...
            ]
        }
    archivo_salida : str
        Nombre del archivo KML
    """
    
    concentradores = datos_red.get('concentradores', {})
    sin_cobertura = datos_red.get('sin_cobertura', [])
    
    # Construir KML
    kml = ['<?xml version="1.0" encoding="UTF-8"?>']
    kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    kml.append('<Document>')
    kml.append('<name>Red RF - Cobertura Agrupada</name>')
    
    # Estilos
    kml.append('''
    <Style id="concentrador">
        <IconStyle>
            <color>ff0000ff</color>
            <scale>1.4</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href></Icon>
        </IconStyle>
        <LabelStyle><scale>0.9</scale></LabelStyle>
    </Style>
    <Style id="concentrador_fijo">
        <IconStyle>
            <color>ff00ff00</color>
            <scale>1.5</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href></Icon>
        </IconStyle>
        <LabelStyle><scale>0.9</scale></LabelStyle>
    </Style>
    <Style id="ied">
        <IconStyle>
            <color>ffff0000</color>
            <scale>0.9</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
        </IconStyle>
        <LabelStyle><scale>0.7</scale></LabelStyle>
    </Style>
    <Style id="ied_sin_cobertura">
        <IconStyle>
            <color>ff0000ff</color>
            <scale>1.0</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle_highlight.png</href></Icon>
        </IconStyle>
        <LabelStyle><scale>0.8</scale></LabelStyle>
    </Style>
    <Style id="linea_buena">
        <LineStyle><color>ff00ff00</color><width>2</width></LineStyle>
    </Style>
    <Style id="linea_media">
        <LineStyle><color>ff00ffff</color><width>2</width></LineStyle>
    </Style>
    <Style id="linea_marginal">
        <LineStyle><color>ff0080ff</color><width>2</width></LineStyle>
    </Style>
    ''')
    
    # ==========================================================================
    # CARPETA POR CADA CONCENTRADOR
    # ==========================================================================
    for placa_conc, datos_conc in concentradores.items():
        lat_conc = datos_conc['lat']
        lon_conc = datos_conc['lon']
        tipo_conc = datos_conc.get('tipo', 'CONCENTRADOR')
        es_fijo = datos_conc.get('es_fijo', False)
        ieds = datos_conc.get('ieds', [])
        
        # Nombre de carpeta
        marca_fijo = " [FIJO]" if es_fijo else ""
        n_ieds = len(ieds)
        
        kml.append(f'<Folder>')
        kml.append(f'<name>{placa_conc}{marca_fijo} ({n_ieds} IEDs)</name>')
        
        # Punto del concentrador
        estilo_conc = "concentrador_fijo" if es_fijo else "concentrador"
        descripcion_conc = f"Tipo: {tipo_conc}\nIEDs cubiertos: {n_ieds}"
        if es_fijo:
            descripcion_conc += "\n[CONCENTRADOR FIJO]"
        
        kml.append(f'''
        <Placemark>
            <name>★ {placa_conc}</name>
            <description>{descripcion_conc}</description>
            <styleUrl>#{estilo_conc}</styleUrl>
            <Point><coordinates>{lon_conc},{lat_conc},0</coordinates></Point>
        </Placemark>''')
        
        # Subcarpeta: IEDs cubiertos por este concentrador
        if ieds:
            kml.append(f'<Folder>')
            kml.append(f'<name>IEDs ({n_ieds})</name>')
            
            for ied in ieds:
                placa_ied = ied['placa']
                lat_ied = ied['lat']
                lon_ied = ied['lon']
                tipo_ied = ied.get('tipo', 'IED')
                dist = ied.get('distancia_km', 0)
                margen = ied.get('margen_fresnel_60', 0)
                
                descripcion_ied = f"Tipo: {tipo_ied}\nConcentrador: {placa_conc}\nDistancia: {dist:.2f} km\nMargen F60: {margen:.1f} m"
                
                kml.append(f'''
            <Placemark>
                <name>{placa_ied}</name>
                <description>{descripcion_ied}</description>
                <styleUrl>#ied</styleUrl>
                <Point><coordinates>{lon_ied},{lat_ied},0</coordinates></Point>
            </Placemark>''')
            
            kml.append('</Folder>')  # Cerrar subcarpeta IEDs
        
        # Subcarpeta: Enlaces RF
        if ieds:
            kml.append(f'<Folder>')
            kml.append(f'<name>Enlaces ({n_ieds})</name>')
            
            for ied in ieds:
                placa_ied = ied['placa']
                lat_ied = ied['lat']
                lon_ied = ied['lon']
                dist = ied.get('distancia_km', 0)
                margen = ied.get('margen_fresnel_60', 0)
                
                # Color según margen
                if margen > 20:
                    estilo = "linea_buena"
                elif margen > 5:
                    estilo = "linea_media"
                else:
                    estilo = "linea_marginal"
                
                kml.append(f'''
            <Placemark>
                <name>{placa_conc} → {placa_ied}</name>
                <description>Distancia: {dist:.2f} km\nMargen F60: {margen:.1f} m</description>
                <styleUrl>#{estilo}</styleUrl>
                <LineString>
                    <coordinates>
                        {lon_conc},{lat_conc},0
                        {lon_ied},{lat_ied},0
                    </coordinates>
                </LineString>
            </Placemark>''')
            
            kml.append('</Folder>')  # Cerrar subcarpeta Enlaces
        
        kml.append('</Folder>')  # Cerrar carpeta del concentrador
    
    # ==========================================================================
    # CARPETA: IEDs SIN COBERTURA
    # ==========================================================================
    if sin_cobertura:
        kml.append('<Folder>')
        kml.append(f'<name>⚠ SIN COBERTURA ({len(sin_cobertura)} IEDs)</name>')
        
        for ied in sin_cobertura:
            placa = ied['placa']
            lat = ied['lat']
            lon = ied['lon']
            tipo = ied.get('tipo', 'IED')
            
            kml.append(f'''
        <Placemark>
            <name>⚠ {placa}</name>
            <description>Tipo: {tipo}\n[SIN COBERTURA - Requiere solución]</description>
            <styleUrl>#ied_sin_cobertura</styleUrl>
            <Point><coordinates>{lon},{lat},0</coordinates></Point>
        </Placemark>''')
        
        kml.append('</Folder>')
    
    kml.append('</Document>')
    kml.append('</kml>')
    
    # Guardar
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.write('\n'.join(kml))
    
    # Resumen
    n_conc = len(concentradores)
    n_ieds_cubiertos = sum(len(c.get('ieds', [])) for c in concentradores.values())
    n_sin_cob = len(sin_cobertura)
    
    print(f"\n✓ KML generado: {archivo_salida}")
    print(f"  - {n_conc} concentradores (carpetas)")
    print(f"  - {n_ieds_cubiertos} IEDs cubiertos")
    if n_sin_cob > 0:
        print(f"  - {n_sin_cob} IEDs SIN COBERTURA")
    
    return archivo_salida


def generar_kml(datos_enlaces: list, archivo_salida: str = "red_rf.kml"):
    """
    Genera archivo KML con concentradores, IEDs y enlaces.
    
    Parámetros:
    -----------
    datos_enlaces : list of dict
        Cada dict debe tener:
        - placa_conc, lat_conc, lon_conc
        - placa_ied, lat_ied, lon_ied
        - margen_fresnel_60 (para color de línea)
        - distancia_km
        - es_concentrador (bool, para marcar el punto)
    archivo_salida : str
        Nombre del archivo KML
    """
    
    # Colores KML (formato aabbggrr)
    COLOR_CONCENTRADOR = "ff0000ff"  # Rojo
    COLOR_IED = "ffff0000"           # Azul
    COLOR_LINEA_BUENO = "ff00ff00"   # Verde (margen > 20m)
    COLOR_LINEA_MEDIO = "ff00ffff"   # Amarillo (margen 5-20m)
    COLOR_LINEA_MARGINAL = "ff0080ff" # Naranja (margen 0-5m)
    
    def color_por_margen(margen):
        if margen > 20:
            return COLOR_LINEA_BUENO
        elif margen > 5:
            return COLOR_LINEA_MEDIO
        else:
            return COLOR_LINEA_MARGINAL
    
    # Recolectar puntos únicos
    concentradores = {}
    ieds = {}
    
    for enlace in datos_enlaces:
        # Concentrador
        placa_c = enlace['placa_conc']
        if placa_c not in concentradores:
            concentradores[placa_c] = {
                'lat': enlace['lat_conc'],
                'lon': enlace['lon_conc']
            }
        
        # IED (ignorar dummies)
        placa_i = enlace['placa_ied']
        if placa_i != '_DUMMY_' and placa_i not in ieds:
            ieds[placa_i] = {
                'lat': enlace['lat_ied'],
                'lon': enlace['lon_ied']
            }
    
    # Construir KML
    kml = ['<?xml version="1.0" encoding="UTF-8"?>']
    kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    kml.append('<Document>')
    kml.append('<name>Red RF - Cobertura</name>')
    
    # Estilos
    kml.append('''
    <Style id="concentrador">
        <IconStyle>
            <color>ff0000ff</color>
            <scale>1.4</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href></Icon>
        </IconStyle>
        <LabelStyle><scale>0.8</scale></LabelStyle>
    </Style>
    <Style id="ied">
        <IconStyle>
            <color>ffff0000</color>
            <scale>1.0</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
        </IconStyle>
        <LabelStyle><scale>0.7</scale></LabelStyle>
    </Style>
    <Style id="linea_buena">
        <LineStyle><color>ff00ff00</color><width>2</width></LineStyle>
    </Style>
    <Style id="linea_media">
        <LineStyle><color>ff00ffff</color><width>2</width></LineStyle>
    </Style>
    <Style id="linea_marginal">
        <LineStyle><color>ff0080ff</color><width>2</width></LineStyle>
    </Style>
    ''')
    
    # Folder: Concentradores
    kml.append('<Folder><name>Concentradores</name>')
    for placa, coords in concentradores.items():
        kml.append(f'''
        <Placemark>
            <name>{placa}</name>
            <description>CONCENTRADOR</description>
            <styleUrl>#concentrador</styleUrl>
            <Point><coordinates>{coords['lon']},{coords['lat']},0</coordinates></Point>
        </Placemark>''')
    kml.append('</Folder>')
    
    # Folder: IEDs
    kml.append('<Folder><name>IEDs</name>')
    for placa, coords in ieds.items():
        kml.append(f'''
        <Placemark>
            <name>{placa}</name>
            <description>IED</description>
            <styleUrl>#ied</styleUrl>
            <Point><coordinates>{coords['lon']},{coords['lat']},0</coordinates></Point>
        </Placemark>''')
    kml.append('</Folder>')
    
    # Folder: Enlaces
    kml.append('<Folder><name>Enlaces RF</name>')
    for enlace in datos_enlaces:
        # Ignorar dummies (son solo para registrar concentradores)
        if enlace['placa_ied'] == '_DUMMY_':
            continue
        
        margen = enlace.get('margen_fresnel_60', 0)
        
        if margen > 20:
            estilo = "linea_buena"
        elif margen > 5:
            estilo = "linea_media"
        else:
            estilo = "linea_marginal"
        
        kml.append(f'''
        <Placemark>
            <name>{enlace['placa_conc']} → {enlace['placa_ied']}</name>
            <description>Distancia: {enlace['distancia_km']:.2f} km | Margen F60: {margen:.1f} m</description>
            <styleUrl>#{estilo}</styleUrl>
            <LineString>
                <coordinates>
                    {enlace['lon_conc']},{enlace['lat_conc']},0
                    {enlace['lon_ied']},{enlace['lat_ied']},0
                </coordinates>
            </LineString>
        </Placemark>''')
    kml.append('</Folder>')
    
    kml.append('</Document>')
    kml.append('</kml>')
    
    # Guardar
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.write('\n'.join(kml))
    
    print(f"\n✓ KML generado: {archivo_salida}")
    return archivo_salida


# =============================================================================
# MAIN (para pruebas directas del módulo)
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   MÓDULO LÍNEA DE VISTA + FRESNEL")
    print("="*60 + "\n")
    
    lv = LineaVista()
    print("\nMódulo cargado correctamente.")
