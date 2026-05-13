"""
Generación de gráficas
======================

Visualización matplotlib de perfiles de enlaces RF.

Este módulo es autosuficiente: recibe la cartografía y los parámetros del
enlace, recomputa el perfil necesario y produce un PNG. NO depende de la clase
LineaVistaFresnel (recibe el dict de resultado de verificar() como parámetro,
así puede usarse igual con otros backends como itmlogic).

Diagnóstico de fallas en este módulo:
- "Module backend Agg is not interactive": esperado, usamos Agg para guardar
  PNGs sin abrir ventana. No es error.
- Gráfico vacío o sin terreno: revisa que la cartografía esté cargada.
- Etiquetas/leyendas mal posicionadas: ajusta figsize o fontsize.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend no interactivo, escribe directo a archivo
import matplotlib.pyplot as plt

from cartografia import Cartografia
from operaciones import correccion_curvatura, radio_fresnel


# =============================================================================
# GRÁFICA: PERFIL DE ENLACE FRESNEL
# =============================================================================

def graficar_perfil_fresnel(cartografia: Cartografia,
                            lat1, lon1, altura1,
                            lat2, lon2, altura2,
                            resultado: dict,
                            frecuencia_mhz=900,
                            curvatura=True,
                            guardar="perfil_linea_vista.png",
                            nombre_a="Punto A",
                            nombre_b="Punto B"):
    """
    Genera y guarda el gráfico del perfil del enlace con su zona de Fresnel.
    
    Parámetros
    ----------
    cartografia : Cartografia
        Instancia ya cargada con SRTM.
    lat1, lon1, altura1 : float
        Punto A (lat, lon en grados; altura sobre terreno en m).
    lat2, lon2, altura2 : float
        Punto B.
    resultado : dict
        Salida de LineaVistaFresnel.verificar() — usada para título y para
        marcar la obstrucción si la hay.
    frecuencia_mhz : float
    curvatura : bool
        Si True, aplica corrección por curvatura terrestre al perfil (debe
        coincidir con lo usado en verificar()).
    guardar : str
        Ruta del PNG de salida.
    nombre_a, nombre_b : str
        Etiquetas en la leyenda.
    
    Retorna
    -------
    str : ruta del archivo guardado.
    """
    # Recomputar perfil (mismo proceso que verificar, pero aquí para graficar)
    distancias, elevaciones = cartografia.perfil_terreno(lat1, lon1, lat2, lon2)
    
    if curvatura:
        correcciones = np.array([correccion_curvatura(d) for d in distancias])
        elev_efectiva = elevaciones - correcciones
    else:
        elev_efectiva = elevaciones.copy()
    
    h_a = elev_efectiva[0] + altura1
    h_b = elev_efectiva[-1] + altura2
    d_total = distancias[-1]
    
    linea_vista = np.linspace(h_a, h_b, len(distancias))
    
    radios_fresnel = np.array([
        radio_fresnel(d, d_total - d, frecuencia_mhz) 
        for d in distancias
    ])
    
    fresnel_superior = linea_vista + radios_fresnel
    fresnel_inferior = linea_vista - radios_fresnel
    fresnel_60_superior = linea_vista + 0.6 * radios_fresnel
    fresnel_60_inferior = linea_vista - 0.6 * radios_fresnel
    
    # --- Figura ---
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Terreno relleno
    ax.fill_between(distancias, 0, elev_efectiva, color='#8B4513', alpha=0.4)
    ax.plot(distancias, elev_efectiva, color='#654321', linewidth=2, label='Terreno')
    
    # Zona de Fresnel 100% (clara)
    ax.fill_between(distancias, fresnel_inferior, fresnel_superior,
                    color='#87CEEB', alpha=0.3, label='Fresnel 100%')
    
    # Zona de Fresnel 60% (oscura)
    ax.fill_between(distancias, fresnel_60_inferior, fresnel_60_superior,
                    color='#4169E1', alpha=0.3, label='Fresnel 60%')
    
    # Línea de vista (color/estilo según estado)
    if resultado['estado'] == 'libre':
        color_los, estilo = '#228B22', '-'
    elif resultado['estado'] == 'fresnel_parcial':
        color_los, estilo = '#FFA500', '-'
    elif resultado['estado'] == 'fresnel_60_obstruido':
        color_los, estilo = '#FF8C00', '--'
    else:  # obstruido
        color_los, estilo = '#DC143C', '--'
    
    ax.plot(distancias, linea_vista, color=color_los, linewidth=2.5,
            linestyle=estilo, label='Línea de vista')
    
    # Marcadores de antenas
    ax.plot(0, h_a, 'b^', markersize=14, label=f'{nombre_a} ({altura1}m)', zorder=5)
    ax.plot(distancias[-1], h_b, 'rv', markersize=14, label=f'{nombre_b} ({altura2}m)', zorder=5)
    
    # Torres
    ax.plot([0, 0], [elev_efectiva[0], h_a], 'b-', linewidth=3, alpha=0.7)
    ax.plot([distancias[-1], distancias[-1]], [elev_efectiva[-1], h_b], 'r-', linewidth=3, alpha=0.7)
    
    # Marcar obstrucción si existe
    if resultado['obstruccion_km'] is not None:
        idx = np.argmin(np.abs(distancias - resultado['obstruccion_km']))
        ax.plot(distancias[idx], elev_efectiva[idx], 'rX',
                markersize=18, markeredgewidth=3, label='Obstrucción', zorder=6)
        ax.axvline(distancias[idx], color='red', linestyle=':', alpha=0.4)
    
    # Ejes
    ax.set_xlabel('Distancia (km)', fontsize=12)
    ax.set_ylabel('Elevación (m)', fontsize=12)
    
    # Título según estado
    titulos = {
        'libre':              ("✓ ENLACE LIBRE (Fresnel 100%)",   'green'),
        'fresnel_parcial':    ("◐ FRESNEL PARCIAL (>60%)",        'orange'),
        'fresnel_60_obstruido': ("⚠ FRESNEL 60% OBSTRUIDO",       'darkorange'),
        'obstruido':          ("✗ LÍNEA DE VISTA OBSTRUIDA",      'red')
    }
    titulo, color_titulo = titulos[resultado['estado']]
    ax.set_title(titulo, fontsize=16, fontweight='bold', color=color_titulo)
    
    # Cuadro de información
    info = (
        f"═══ ANÁLISIS RF ═══\n"
        f"Frecuencia: {frecuencia_mhz} MHz\n"
        f"Distancia: {resultado['distancia_km']:.2f} km\n"
        f"─────────────────\n"
        f"{nombre_a}:\n"
        f"  Terreno: {elev_efectiva[0]:.0f}m\n"
        f"  Antena: +{altura1}m\n"
        f"  Total: {h_a:.0f}m\n"
        f"─────────────────\n"
        f"{nombre_b}:\n"
        f"  Terreno: {elev_efectiva[-1]:.0f}m\n"
        f"  Antena: +{altura2}m\n"
        f"  Total: {h_b:.0f}m\n"
        f"─────────────────\n"
        f"Margen LOS: {resultado['margen_los_m']:.1f}m\n"
        f"Margen F60%: {resultado['margen_fresnel_60_m']:.1f}m\n"
        f"Margen F100%: {resultado['margen_fresnel_100_m']:.1f}m"
    )
    
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
    
    return guardar


# =============================================================================
# GRÁFICA: PERFIL DE ENLACE LONGLEY-RICE (ITM)
# =============================================================================

def graficar_perfil_itm(cartografia: Cartografia,
                        lat1, lon1, altura1,
                        lat2, lon2, altura2,
                        resultado: dict,
                        frecuencia_mhz=900,
                        curvatura=True,
                        guardar="perfil_itm.png",
                        nombre_a="Punto A",
                        nombre_b="Punto B"):
    """
    Genera y guarda el gráfico del perfil del enlace con anotaciones del
    análisis Longley-Rice (ITM).
    
    A diferencia del gráfico Fresnel, este NO dibuja zonas de Fresnel. En su
    lugar muestra el terreno, la línea de vista, y una tabla con las pérdidas
    totales a 50%, 90% y 95% de confianza, más el veredicto contra el
    presupuesto de enlace.
    
    Parámetros
    ----------
    cartografia : Cartografia
    lat1, lon1, altura1, lat2, lon2, altura2 : float
    resultado : dict
        Salida de LineaVistaITM.verificar().
    frecuencia_mhz : float
    curvatura : bool
        Si True, aplica corrección por curvatura al perfil graficado.
    guardar : str
    nombre_a, nombre_b : str
    
    Retorna
    -------
    str : ruta del archivo guardado.
    """
    distancias, elevaciones = cartografia.perfil_terreno(lat1, lon1, lat2, lon2)
    
    if curvatura:
        correcciones = np.array([correccion_curvatura(d) for d in distancias])
        elev_efectiva = elevaciones - correcciones
    else:
        elev_efectiva = elevaciones.copy()
    
    h_a = elev_efectiva[0] + altura1
    h_b = elev_efectiva[-1] + altura2
    linea_vista = np.linspace(h_a, h_b, len(distancias))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Terreno
    ax.fill_between(distancias, 0, elev_efectiva, color='#8B4513', alpha=0.4)
    ax.plot(distancias, elev_efectiva, color='#654321', linewidth=2, label='Terreno')
    
    # Línea de vista (color según veredicto)
    estado = resultado.get('estado', 'error')
    if estado == 'viable':
        color_los, estilo = '#228B22', '-'
    elif estado == 'no_viable':
        color_los, estilo = '#DC143C', '--'
    else:
        color_los, estilo = '#808080', ':'
    
    ax.plot(distancias, linea_vista, color=color_los, linewidth=2.5,
            linestyle=estilo, label='Línea de vista')
    
    # Antenas
    ax.plot(0, h_a, 'b^', markersize=14, label=f'{nombre_a} ({altura1}m)', zorder=5)
    ax.plot(distancias[-1], h_b, 'rv', markersize=14, label=f'{nombre_b} ({altura2}m)', zorder=5)
    
    # Torres
    ax.plot([0, 0], [elev_efectiva[0], h_a], 'b-', linewidth=3, alpha=0.7)
    ax.plot([distancias[-1], distancias[-1]], [elev_efectiva[-1], h_b], 'r-', linewidth=3, alpha=0.7)
    
    # Ejes
    ax.set_xlabel('Distancia (km)', fontsize=12)
    ax.set_ylabel('Elevación (m)', fontsize=12)
    
    # Título
    titulos = {
        'viable':    ("✓ ENLACE VIABLE (Longley-Rice)",     'green'),
        'no_viable': ("✗ ENLACE NO VIABLE (Longley-Rice)", 'red'),
        'error':     ("⚠ ERROR EN ANÁLISIS ITM",             'gray'),
    }
    titulo, color_titulo = titulos.get(estado, titulos['error'])
    ax.set_title(titulo, fontsize=16, fontweight='bold', color=color_titulo)
    
    # Cuadro de información
    if estado == 'error':
        info = f"═══ ERROR ITM ═══\n{resultado.get('error', 'desconocido')}"
    else:
        info = (
            f"═══ ANÁLISIS LONGLEY-RICE ═══\n"
            f"Frecuencia: {frecuencia_mhz} MHz\n"
            f"Distancia: {resultado['distancia_km']:.2f} km\n"
            f"─────────────────────\n"
            f"Pérdida espacio libre (FSL):\n"
            f"  {resultado['perdida_fsl_db']:.1f} dB\n"
            f"Exceso ITM (aref median):\n"
            f"  {resultado['perdida_aref_db']:.1f} dB\n"
            f"─────────────────────\n"
            f"PÉRDIDA TOTAL (dB):\n"
            f"  Mediana (50%):  {resultado['perdida_total_db_mediana']:.1f}\n"
            f"  90% confianza:  {resultado['perdida_total_db_p90']:.1f}\n"
            f"  95% confianza:  {resultado['perdida_total_db_p95']:.1f}\n"
            f"─────────────────────\n"
            f"Presupuesto: {resultado['perdida_max_db']:.1f} dB\n"
            f"Margen:      {resultado['margen_db']:+.1f} dB\n"
        )
        if resultado.get('kwx', 0) > 0:
            info += f"\n⚠ kwx={resultado['kwx']} (rango de validez)"
    
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
    
    return guardar


# =============================================================================
# PRUEBAS DIRECTAS
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("   MÓDULO GRAFICAS - SMOKE TEST")
    print("="*60)
    print("\nEste módulo no se ejecuta directamente; importa desde main.py")
    print("o desde linea_vista_fresnel.py.")
