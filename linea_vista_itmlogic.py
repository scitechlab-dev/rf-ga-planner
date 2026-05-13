"""
Modelo RF: Longley-Rice (ITM) vía itmlogic
==========================================

Análisis de enlaces RF mediante el modelo Longley-Rice (Irregular Terrain
Model), usando el paquete `itmlogic` (puerto Python del ITM v1.2.2 de la NTIA).

A diferencia de Fresnel (geometría pura), ITM modela:
- Difracción sobre terreno irregular
- Reflexiones de tierra (con permitividad y conductividad del suelo)
- Curvatura terrestre con refractividad atmosférica
- Variabilidad estocástica (tiempo, ubicación, situación)

La interfaz pública es IDÉNTICA a LineaVistaFresnel (verificar,
funcion_objetivo_rf), por lo que se puede usar como reemplazo directo en el
GA o en cualquier orquestador.

Diferencias clave con Fresnel:
- Devuelve pérdida en dB (no márgenes geométricos en m).
- La viabilidad se decide contra un presupuesto de enlace configurable.
- El campo 'estado' es 'viable' o 'no_viable' (no hay 'fresnel_*').
- Para mantener compatibilidad con el GA actual, se exponen los aliases:
  'fresnel_60_libre' y 'libre' como sinónimos de 'viable'.

Diagnóstico de fallas en este módulo:
- ImportError: itmlogic no instalado. Instalar con `pip install itmlogic`.
- "kwx" warnings altos: parámetros fuera del rango recomendado por ITM
  (altura<0.5m o >3000m, frecuencia<20MHz o >40GHz, etc.).
- Pérdidas muy bajas (<FSL): probablemente algún parámetro del medio mal
  configurado; revisa refractividad, polarización, klim.
- Excepciones internas de itmlogic en perfiles con un solo nivel de
  elevación: agrega una pequeña perturbación o aumenta num_puntos.
"""

import math
import numpy as np
import pandas as pd

from itmlogic.preparatory_subroutines.qlrps import qlrps
from itmlogic.preparatory_subroutines.qlrpfl import qlrpfl
from itmlogic.statistics.avar import avar
from itmlogic.misc.qerfi import qerfi

from cartografia import Cartografia
from operaciones import distancia_haversine


# =============================================================================
# CONSTANTES Y UTILIDADES
# =============================================================================

# Climas ITM (parámetro klim)
CLIMA_EQUATORIAL = 1
CLIMA_CONTINENTAL_SUBTROPICAL = 2     # recomendado para El Salvador
CLIMA_MARITIMO_SUBTROPICAL = 3
CLIMA_DESERT = 4
CLIMA_CONTINENTAL_TEMPERATE = 5
CLIMA_MARITIMO_TEMPERATE_TIERRA = 6
CLIMA_MARITIMO_TEMPERATE_MAR = 7

# Polarización
POLARIZACION_HORIZONTAL = 0
POLARIZACION_VERTICAL = 1


def perdida_espacio_libre(distancia_km, frecuencia_mhz):
    """Pérdida por espacio libre (Free Space Path Loss) en dB."""
    if distancia_km <= 0:
        return 0.0
    return 32.45 + 20*math.log10(distancia_km) + 20*math.log10(frecuencia_mhz)


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class LineaVistaITM:
    """
    Analizador RF basado en el modelo Longley-Rice (ITM) vía itmlogic.
    
    Recibe una instancia de Cartografía (inyección de dependencia) y delega
    en ella todo lo relativo a terreno. Esta clase solo aporta el modelo de
    propagación.
    
    Parámetros del modelo ITM (ver Hufford 1995, NTIA Report 82-100):
    -----------------------------------------------------------------
    klim : int
        Código de clima (1-7). Por defecto 2 (continental subtropical),
        razonable para El Salvador.
    refractividad : float
        Refractividad superficial reducida al nivel del mar (N-unidades).
        Valores típicos: 301 (continental templado), 360 (subtropical/
        ecuatorial), 320 (marítimo).
    polarizacion : int
        0 horizontal, 1 vertical. Vertical es lo más común para SCADA.
    epsilon : float
        Permitividad relativa del suelo. Típicos: 4 (suelo pobre),
        15 (promedio), 25 (bueno), 81 (agua).
    sigma : float
        Conductividad del suelo (S/m). Típicos: 0.001 (pobre),
        0.005 (promedio), 0.02 (buena).
    
    Presupuesto de enlace:
    ----------------------
    perdida_max_db : float
        Pérdida máxima admisible (dB) para considerar el enlace viable.
        Para SCADA típico a 900 MHz: 130-140 dB (TX 1W, antenas 6 dBi,
        sensibilidad RX -100 dBm, margen 10 dB → ~132 dB).
    
    Cuantiles de la predicción:
    ---------------------------
    confianza, fiabilidad, situacion : float (0 a 1)
        Cuantiles del modelo de variabilidad. 0.5 = mediana. Por defecto
        se usan 0.5 para los tres; el método verificar() también computa
        90% y 95% para reportar.
    """
    
    def __init__(self, cartografia: Cartografia,
                 klim: int = CLIMA_CONTINENTAL_SUBTROPICAL,
                 refractividad: float = 360.0,
                 polarizacion: int = POLARIZACION_VERTICAL,
                 epsilon: float = 15.0,
                 sigma: float = 0.005,
                 perdida_max_db: float = 130.0,
                 confianza: float = 0.5,
                 fiabilidad: float = 0.5,
                 situacion: float = 0.5,
                 num_puntos_perfil: int = 500):
        self.carto = cartografia
        self.klim = klim
        self.refractividad = refractividad
        self.polarizacion = polarizacion
        self.epsilon = epsilon
        self.sigma = sigma
        self.perdida_max_db = perdida_max_db
        self.confianza = confianza
        self.fiabilidad = fiabilidad
        self.situacion = situacion
        self.num_puntos = num_puntos_perfil
    
    # -------------------------------------------------------------------------
    # Método interno: orquestación itmlogic
    # -------------------------------------------------------------------------
    
    def _correr_itm(self, elevaciones, distancia_km, altura_tx, altura_rx,
                    frecuencia_mhz):
        """
        Ejecuta el flujo qlrps → qlrpfl → avar para varios cuantiles.
        
        Retorna
        -------
        dict con aref (mediana), perdidas por cuantil, y warnings (kwx).
        """
        n_samples = len(elevaciones)
        step_m = (distancia_km * 1000) / (n_samples - 1)
        
        # Formato pfl que espera itmlogic:
        #   pfl[0] = n_puntos - 1 (DEBE ser int, se usa como índice)
        #   pfl[1] = paso entre puntos (m)
        #   pfl[2..n+1] = elevaciones
        pfl = [int(n_samples - 1), float(step_m)] + [float(e) for e in elevaciones]
        
        zsys = float(np.mean(elevaciones))  # elevación promedio del sistema
        
        # Paso 1: parámetros derivados (qlrps)
        wn, gme, ens, zgnd = qlrps(
            frecuencia_mhz, zsys, self.refractividad,
            self.polarizacion, self.epsilon, self.sigma
        )
        
        # Paso 2: armar dict prop con TODOS los campos que requiere qlrpfl/avar.
        # Nota: mdvarx=-1 hace que NO se sobreescriba mdvar. lvar=5 fuerza
        # inicialización completa del modelo de variabilidad en avar.
        prop = {
            'pfl': pfl,
            'hg': [float(altura_tx), float(altura_rx)],
            'fmhz': float(frecuencia_mhz),
            'wn': wn,
            'gme': gme,
            'ens': ens,
            'zgnd': zgnd,
            'klim': self.klim,
            'klimx': 0,
            'mdvar': 3,        # broadcast mode (típico para enlaces fijos)
            'mdvarx': -1,      # no sobreescribir
            'lvar': 5,         # forzar setup completo
            'kwx': 0,          # contador de warnings, lo lee ITM
        }
        
        # Paso 3: qlrpfl ejecuta la preparación P2P y llama a lrprop.
        # Llena prop['aref'] (pérdida básica de referencia, mediana, dB).
        prop = qlrpfl(prop)
        aref = prop['aref']
        
        # Paso 4: avar para varios cuantiles. avar() devuelve la pérdida
        # básica (sobre espacio libre) a las quantiles solicitadas.
        cuantiles_estandar = {
            0.5: 'mediana',
            0.9: 'p90',
            0.95: 'p95',
        }
        avar_por_cuantil = {}
        
        # Quantile principal del usuario
        z_user = qerfi([self.fiabilidad, self.situacion, self.confianza])
        prop['lvar'] = 5
        avar_user, prop = avar(z_user[0], z_user[1], z_user[2], prop)
        
        # Cuantiles estándar para reporte
        for q, etiqueta in cuantiles_estandar.items():
            z = qerfi([q, q, q])
            prop['lvar'] = 5  # reinicializar entre llamadas
            av, _ = avar(z[0], z[1], z[2], prop)
            avar_por_cuantil[etiqueta] = av
        
        return {
            'aref_db': aref,
            'avar_basica_db_user': avar_user,
            'avar_basica_db_por_cuantil': avar_por_cuantil,
            'kwx': prop.get('kwx', 0),
        }
    
    # -------------------------------------------------------------------------
    # Interfaz pública (idéntica a LineaVistaFresnel)
    # -------------------------------------------------------------------------
    
    def verificar(self, lat1, lon1, altura1, lat2, lon2, altura2,
                  frecuencia_mhz=900, curvatura=True):
        """
        Verifica un enlace RF entre dos puntos usando Longley-Rice.
        
        Parámetros: idénticos a LineaVistaFresnel.verificar.
        (curvatura se acepta por compatibilidad pero ITM siempre la usa
        internamente vía gme).
        
        Retorna
        -------
        dict con:
            estado : str ('viable' | 'no_viable' | 'error')
            viable : bool
            libre, fresnel_60_libre : bool (alias de viable, para
                compatibilidad con código que esperaba la API Fresnel)
            distancia_km : float
            frecuencia_mhz : float
            perdida_total_db : pérdida total al cuantil del usuario (FSL + avar)
            perdida_fsl_db : pérdida en espacio libre (referencia)
            perdida_aref_db : pérdida de referencia ITM (mediana, sobre FSL)
            perdida_total_db_mediana : total al 50% (mediana)
            perdida_total_db_p90 : total al 90% de confianza
            perdida_total_db_p95 : total al 95% de confianza
            margen_db : presupuesto - perdida_total (positivo = viable)
            kwx : código de warning ITM (0=OK, 1-4=advertencias crecientes)
        """
        # Perfil de terreno via cartografía
        distancias, elevaciones = self.carto.perfil_terreno(
            lat1, lon1, lat2, lon2, num_puntos=self.num_puntos
        )
        d_total = float(distancias[-1])
        fsl = perdida_espacio_libre(d_total, frecuencia_mhz)
        
        # Ejecutar ITM
        try:
            itm = self._correr_itm(
                elevaciones, d_total, altura1, altura2, frecuencia_mhz
            )
        except Exception as e:
            # Cualquier fallo numérico de ITM se reporta limpiamente
            return {
                'estado': 'error',
                'viable': False,
                'libre': False,
                'fresnel_60_libre': False,
                'distancia_km': d_total,
                'frecuencia_mhz': frecuencia_mhz,
                'error': f"{type(e).__name__}: {e}",
                'perdida_total_db': float('inf'),
                'perdida_fsl_db': fsl,
                'margen_db': float('-inf'),
            }
        
        # Pérdidas totales (FSL + pérdida básica ITM)
        perdida_user = fsl + itm['avar_basica_db_user']
        perdidas_cuantil = {
            k: fsl + v for k, v in itm['avar_basica_db_por_cuantil'].items()
        }
        
        margen = self.perdida_max_db - perdida_user
        viable = margen >= 0
        
        return {
            'estado': 'viable' if viable else 'no_viable',
            'viable': viable,
            # Aliases para compatibilidad con la API Fresnel
            'libre': viable,
            'fresnel_60_libre': viable,
            # Información del enlace
            'distancia_km': d_total,
            'frecuencia_mhz': frecuencia_mhz,
            # Pérdidas
            'perdida_total_db': perdida_user,
            'perdida_fsl_db': fsl,
            'perdida_aref_db': itm['aref_db'],
            'perdida_total_db_mediana': perdidas_cuantil.get('mediana'),
            'perdida_total_db_p90': perdidas_cuantil.get('p90'),
            'perdida_total_db_p95': perdidas_cuantil.get('p95'),
            # Presupuesto
            'margen_db': margen,
            'perdida_max_db': self.perdida_max_db,
            # Warning ITM
            'kwx': itm['kwx'],
        }
    
    def funcion_objetivo_rf(self, csv_path, indice_a, indice_b,
                            frecuencia_mhz=900,
                            altura_ied=8,
                            altura_concentrador=30,
                            graficar=False):
        """
        Wrapper conveniente análogo al de LineaVistaFresnel: lee CSV, evalúa
        por índices, y opcionalmente grafica.
        
        El gráfico utiliza graficas.graficar_perfil_itm (también side-by-side
        con Fresnel si se invoca desde comparar_modelos.py).
        """
        df = pd.read_csv(csv_path)
        n = len(df)
        if not (0 <= indice_a < n):
            raise ValueError(f"indice_a={indice_a} fuera de rango [0, {n-1}]")
        if not (0 <= indice_b < n):
            raise ValueError(f"indice_b={indice_b} fuera de rango [0, {n-1}]")
        
        a, b = df.iloc[indice_a], df.iloc[indice_b]
        placa_a, placa_b = str(a['Placa']), str(b['Placa'])
        lat1, lon1 = float(a['Latitud']), float(a['Longitud'])
        lat2, lon2 = float(b['Latitud']), float(b['Longitud'])
        
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
            from graficas import graficar_perfil_itm
            archivo = f"enlace_itm_{placa_a}_{placa_b}.png"
            graficar_perfil_itm(
                cartografia=self.carto,
                lat1=lat1, lon1=lon1, altura1=altura_concentrador,
                lat2=lat2, lon2=lon2, altura2=altura_ied,
                resultado=resultado,
                frecuencia_mhz=frecuencia_mhz,
                guardar=archivo,
                nombre_a=f"{placa_a} (Conc)",
                nombre_b=f"{placa_b} (IED)"
            )
            resultado['archivo_grafico'] = archivo
        
        return resultado


# =============================================================================
# PRUEBAS DIRECTAS
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("   MÓDULO LINEA_VISTA_ITMLOGIC - SMOKE TEST")
    print("="*60)
    
    carto = Cartografia()
    lv = LineaVistaITM(carto)
    print("\nAnalizador Longley-Rice (ITM) listo.")
    print(f"  klim={lv.klim}, refractividad={lv.refractividad}")
    print(f"  polarizacion={lv.polarizacion}, eps={lv.epsilon}, sigma={lv.sigma}")
    print(f"  Presupuesto: {lv.perdida_max_db} dB")
