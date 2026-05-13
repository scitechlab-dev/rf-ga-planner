"""
Modelo RF: Línea de Vista + Zona de Fresnel
============================================

Análisis de enlaces radioeléctricos basado en geometría:
- Línea de vista directa entre antenas.
- Despeje de la primera zona de Fresnel (criterio estándar: 60% libre).

Este módulo NO genera gráficas (eso está en graficas.py) ni KML (eso está en
exportacion.py). Solo decide si un enlace es viable y con qué márgenes.

Para reemplazar este modelo por Longley-Rice (itmlogic) o ITU-R P.1812, crea
otra clase con la misma interfaz pública (verificar, funcion_objetivo_rf) que
reciba la misma Cartografía. El resto del pipeline (GA, main) no cambia.

Diagnóstico de fallas en este módulo:
- Todos los enlaces salen "obstruidos" inesperadamente: revisa alturas y la
  bandera curvatura en verificar().
- Márgenes muy negativos a corta distancia: revisa que las alturas de antena
  estén en metros sobre el terreno (no absolutas).
- funcion_objetivo_rf falla al leer CSV: revisa nombres de columnas exigidos
  (Placa, Latitud, Longitud).
"""

import numpy as np
import pandas as pd

from cartografia import Cartografia
from operaciones import correccion_curvatura, radio_fresnel


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class LineaVistaFresnel:
    """
    Analizador RF basado en línea de vista y zona de Fresnel.
    
    Recibe una instancia de Cartografía (inyección de dependencia) y delega
    en ella todo lo relativo a terreno. Esta clase solo aporta el modelo RF.
    
    Interfaz pública (estable para futuros backends como itmlogic):
        verificar(lat1, lon1, h1, lat2, lon2, h2, freq, curvatura) -> dict
        funcion_objetivo_rf(csv_path, idx_a, idx_b, ...) -> dict
    """
    
    def __init__(self, cartografia: Cartografia):
        self.carto = cartografia
    
    def verificar(self, lat1, lon1, altura1, lat2, lon2, altura2,
                  frecuencia_mhz=900, curvatura=True):
        """
        Verifica un enlace RF entre dos puntos.
        
        Parámetros
        ----------
        lat1, lon1 : float - Coordenadas del punto A.
        altura1 : float - Altura de antena A sobre el terreno (m).
        lat2, lon2 : float - Coordenadas del punto B.
        altura2 : float - Altura de antena B sobre el terreno (m).
        frecuencia_mhz : float - Frecuencia del enlace (MHz).
        curvatura : bool - Si True, aplica corrección por curvatura terrestre.
        
        Retorna
        -------
        dict con:
            estado : str  ('libre', 'fresnel_parcial', 'fresnel_60_obstruido', 'obstruido')
            libre : bool
            fresnel_60_libre : bool
            distancia_km : float
            obstruccion_km : float | None
            margen_los_m : float
            margen_fresnel_60_m : float
            margen_fresnel_100_m : float
            frecuencia_mhz : float
        """
        distancias, elevaciones = self.carto.perfil_terreno(lat1, lon1, lat2, lon2)
        
        if curvatura:
            correcciones = np.array([correccion_curvatura(d) for d in distancias])
            elev_efectiva = elevaciones - correcciones
        else:
            elev_efectiva = elevaciones.copy()
        
        h_a = elev_efectiva[0] + altura1
        h_b = elev_efectiva[-1] + altura2
        d_total = distancias[-1]
        
        linea_vista = np.linspace(h_a, h_b, len(distancias))
        
        # Radio de Fresnel en cada muestra
        radios_fresnel = np.array([
            radio_fresnel(d, d_total - d, frecuencia_mhz) 
            for d in distancias
        ])
        
        # Zona inferior de Fresnel = la crítica para obstrucciones
        fresnel_inferior = linea_vista - radios_fresnel
        
        # Márgenes punto a punto
        margen_los = linea_vista - elev_efectiva
        margen_fresnel_60 = (linea_vista - 0.6 * radios_fresnel) - elev_efectiva
        margen_fresnel_100 = fresnel_inferior - elev_efectiva
        
        # Mínimos (excluyendo extremos para no contar la antena misma)
        margen_los_min = np.min(margen_los[1:-1])
        margen_fresnel_60_min = np.min(margen_fresnel_60[1:-1])
        margen_fresnel_100_min = np.min(margen_fresnel_100[1:-1])
        
        # Clasificación del estado
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
    
    def funcion_objetivo_rf(self, csv_path, indice_a, indice_b,
                            frecuencia_mhz=900,
                            altura_ied=8,
                            altura_concentrador=30,
                            graficar=False):
        """
        Wrapper conveniente: lee un CSV, evalúa un enlace por índice y
        opcionalmente genera el gráfico del perfil.
        
        Parámetros
        ----------
        csv_path : str - Ruta al CSV (columnas: Placa, Latitud, Longitud).
        indice_a : int - Fila del concentrador candidato.
        indice_b : int - Fila del IED.
        frecuencia_mhz : float
        altura_ied : float (m)
        altura_concentrador : float (m)
        graficar : bool - Si True, genera PNG del perfil vía graficas.py.
        
        Retorna
        -------
        dict con el resultado de verificar() más info de las placas y, si se
        pidió, el nombre del archivo gráfico generado.
        """
        df = pd.read_csv(csv_path)
        
        n = len(df)
        if indice_a < 0 or indice_a >= n:
            raise ValueError(f"indice_a={indice_a} fuera de rango [0, {n-1}]")
        if indice_b < 0 or indice_b >= n:
            raise ValueError(f"indice_b={indice_b} fuera de rango [0, {n-1}]")
        
        punto_a = df.iloc[indice_a]
        punto_b = df.iloc[indice_b]
        
        placa_a = str(punto_a['Placa'])
        placa_b = str(punto_b['Placa'])
        lat1, lon1 = float(punto_a['Latitud']), float(punto_a['Longitud'])
        lat2, lon2 = float(punto_b['Latitud']), float(punto_b['Longitud'])
        
        resultado = self.verificar(
            lat1, lon1, altura_concentrador,
            lat2, lon2, altura_ied,
            frecuencia_mhz=frecuencia_mhz
        )
        
        resultado['placa_a'] = placa_a
        resultado['placa_b'] = placa_b
        resultado['indice_a'] = indice_a
        resultado['indice_b'] = indice_b
        
        if graficar:
            # Import diferido para no cargar matplotlib si no se necesita
            from graficas import graficar_perfil_fresnel
            
            nombre_archivo = f"enlace_{placa_a}_{placa_b}.png"
            graficar_perfil_fresnel(
                cartografia=self.carto,
                lat1=lat1, lon1=lon1, altura1=altura_concentrador,
                lat2=lat2, lon2=lon2, altura2=altura_ied,
                resultado=resultado,
                frecuencia_mhz=frecuencia_mhz,
                guardar=nombre_archivo,
                nombre_a=f"{placa_a} (Conc)",
                nombre_b=f"{placa_b} (IED)"
            )
            resultado['archivo_grafico'] = nombre_archivo
        
        return resultado


# =============================================================================
# PRUEBAS DIRECTAS
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("   MÓDULO LINEA_VISTA_FRESNEL - SMOKE TEST")
    print("="*60)
    
    carto = Cartografia()
    lv = LineaVistaFresnel(carto)
    print("\nAnalizador Fresnel listo (no se ejecutó ninguna verificación).")
