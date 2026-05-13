"""
Comparación visual: Línea de Vista Fresnel vs Longley-Rice (con selector gráfico)
=================================================================================

Script CLI con SELECCIÓN GRÁFICA de puntos: abre un mapa interactivo donde
puedes pasar el cursor para ver el nombre de cada dispositivo y hacer clic
para seleccionar los dos extremos del enlace.

Flujo:
    1. Se abre una ventana con todos los dispositivos del CSV en su ubicación
       geográfica, coloreados por tipo.
    2. Pasa el cursor sobre los puntos para ver su nombre/tipo/índice.
    3. CLIC IZQUIERDO en el primer punto (concentrador).
    4. CLIC IZQUIERDO en el segundo punto (IED).
    5. CLIC DERECHO en cualquier parte para resetear la selección.
    6. Cierra la ventana para continuar (Ctrl+W o el botón de cerrar).
    7. El script pregunta alturas, frecuencia y presupuesto en la terminal.
    8. Ejecuta ambos análisis y produce el PNG side-by-side + tabla en consola.

Uso:
    python comparar_modelos.py
"""

# --- Backend matplotlib interactivo ANTES de cualquier otro import ---
# (los módulos graficas/linea_vista_fresnel usan Agg internamente, pero al
# importarse después de matplotlib, no logran cambiar el backend ya inicializado)
import matplotlib
try:
    matplotlib.use('TkAgg')   # backend interactivo estándar en Windows con conda
except Exception:
    pass  # si falla, matplotlib usará el default

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from cartografia import Cartografia
from linea_vista_fresnel import LineaVistaFresnel
from linea_vista_itmlogic import LineaVistaITM
from operaciones import correccion_curvatura, radio_fresnel


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CSV_PATH = "datos.csv"
FRECUENCIA_DEFAULT = 900
ALTURA_IED_DEFAULT = 8
ALTURA_CONC_DEFAULT = 30
PRESUPUESTO_DB_DEFAULT = 130


# =============================================================================
# SELECTOR GRÁFICO INTERACTIVO
# =============================================================================

class SelectorPuntos:
    """
    Ventana interactiva para seleccionar dos puntos del CSV haciendo clic
    sobre un mapa geográfico.
    
    Mantiene estado de selección y dibuja markers grandes sobre los puntos
    elegidos.
    """
    
    UMBRAL_HOVER_DEG = 0.05   # ~5 km en El Salvador; threshold para tooltip
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.idx_a = None
        self.idx_b = None
        self.marker_a = None
        self.marker_b = None
        self.linea_ab = None
        
        self.fig, self.ax = plt.subplots(figsize=(13, 9))
        try:
            self.fig.canvas.manager.set_window_title("Selector de puntos - RFEvolve")
        except Exception:
            pass
        
        self._dibujar_puntos()
        self._configurar_axes()
        self._crear_tooltip()
        self._conectar_eventos()
        self._actualizar_titulo()
    
    def _dibujar_puntos(self):
        """Dibuja todos los puntos coloreados por tipo."""
        if 'Origen_DataFrame' in self.df.columns:
            tipos = self.df['Origen_DataFrame'].fillna('SIN_TIPO').unique()
        else:
            self.df['Origen_DataFrame'] = 'TODOS'
            tipos = ['TODOS']
        
        cmap = plt.colormaps['tab10']
        self.colores = {t: cmap(i % 10) for i, t in enumerate(tipos)}
        
        for tipo in tipos:
            sub = self.df[self.df['Origen_DataFrame'].fillna('SIN_TIPO') == tipo]
            self.ax.scatter(
                sub['Longitud'], sub['Latitud'],
                c=[self.colores[tipo]], label=f"{tipo} ({len(sub)})",
                alpha=0.75, s=45, edgecolors='black', linewidths=0.5,
                zorder=3
            )
    
    def _configurar_axes(self):
        self.ax.set_xlabel('Longitud (°)', fontsize=11)
        self.ax.set_ylabel('Latitud (°)', fontsize=11)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(fontsize=8, loc='upper left',
                       bbox_to_anchor=(1.02, 1.0), title='Tipo de dispositivo')
        
        self.fig.text(
            0.5, 0.02,
            "Clic izq: seleccionar punto  |  Clic der: resetear  |  "
            "Pasa el cursor sobre un punto para ver su nombre  |  "
            "Cierra la ventana para continuar",
            ha='center', fontsize=9, style='italic', color='gray'
        )
    
    def _crear_tooltip(self):
        self.tooltip = self.ax.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                     ec="gray", alpha=0.95),
            fontsize=9, fontfamily='monospace', zorder=20
        )
        self.tooltip.set_visible(False)
    
    def _conectar_eventos(self):
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
    
    def _actualizar_titulo(self):
        if self.idx_a is None:
            tit = "Selecciona el primer punto (CONCENTRADOR) — clic izquierdo"
            color = '#1f77b4'
        elif self.idx_b is None:
            placa_a = self.df.iloc[self.idx_a]['Placa']
            tit = f"A: [{self.idx_a}] {placa_a}   →   Selecciona el segundo punto (IED)"
            color = '#ff7f0e'
        else:
            placa_a = self.df.iloc[self.idx_a]['Placa']
            placa_b = self.df.iloc[self.idx_b]['Placa']
            tit = (f"A: [{self.idx_a}] {placa_a}  ↔  "
                   f"B: [{self.idx_b}] {placa_b}   |   "
                   "Cierra la ventana para analizar")
            color = '#2ca02c'
        self.ax.set_title(tit, fontsize=12, fontweight='bold', color=color)
    
    def _encontrar_punto_cercano(self, x, y, umbral_deg=None):
        """Devuelve el índice del punto más cercano a (x, y) si está dentro
        de umbral_deg grados. None si no hay nada cerca."""
        d2 = (self.df['Longitud'] - x)**2 + (self.df['Latitud'] - y)**2
        idx = int(d2.idxmin())
        dist = float(np.sqrt(d2.iloc[idx]))
        if umbral_deg is None or dist <= umbral_deg:
            return idx, dist
        return None, dist
    
    def _on_hover(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        idx, _ = self._encontrar_punto_cercano(
            event.xdata, event.ydata, umbral_deg=self.UMBRAL_HOVER_DEG
        )
        if idx is None:
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        
        row = self.df.iloc[idx]
        tipo = row.get('Origen_DataFrame', '')
        muni = row.get('Municipio', '')
        texto = f"[{idx}] {row['Placa']}\n{tipo}"
        if isinstance(muni, str) and muni:
            texto += f"\n{muni}"
        self.tooltip.xy = (row['Longitud'], row['Latitud'])
        self.tooltip.set_text(texto)
        self.tooltip.set_visible(True)
        self.fig.canvas.draw_idle()
    
    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.button == 3:  # clic derecho = resetear
            self._resetear_seleccion()
            return
        if event.button != 1:
            return
        
        idx, _ = self._encontrar_punto_cercano(event.xdata, event.ydata)
        if idx is None:
            return
        
        if self.idx_a is None:
            self.idx_a = idx
            self._dibujar_marker_a()
        elif self.idx_b is None and idx != self.idx_a:
            self.idx_b = idx
            self._dibujar_marker_b()
            self._dibujar_linea_ab()
        else:
            return  # ya hay dos o se cliqueó el mismo A
        
        self._actualizar_titulo()
        self.fig.canvas.draw_idle()
    
    def _dibujar_marker_a(self):
        row = self.df.iloc[self.idx_a]
        self.marker_a = self.ax.scatter(
            [row['Longitud']], [row['Latitud']],
            marker='^', s=450, c='#1f77b4',
            edgecolors='yellow', linewidths=3, zorder=15
        )
    
    def _dibujar_marker_b(self):
        row = self.df.iloc[self.idx_b]
        self.marker_b = self.ax.scatter(
            [row['Longitud']], [row['Latitud']],
            marker='v', s=450, c='#d62728',
            edgecolors='yellow', linewidths=3, zorder=15
        )
    
    def _dibujar_linea_ab(self):
        a = self.df.iloc[self.idx_a]
        b = self.df.iloc[self.idx_b]
        self.linea_ab, = self.ax.plot(
            [a['Longitud'], b['Longitud']],
            [a['Latitud'], b['Latitud']],
            'k--', linewidth=2, alpha=0.6, zorder=10
        )
    
    def _resetear_seleccion(self):
        self.idx_a = None
        self.idx_b = None
        for art in (self.marker_a, self.marker_b, self.linea_ab):
            if art is not None:
                try:
                    art.remove()
                except Exception:
                    pass
        self.marker_a = None
        self.marker_b = None
        self.linea_ab = None
        self._actualizar_titulo()
        self.fig.canvas.draw_idle()
    
    def ejecutar(self):
        """Muestra la ventana (bloqueante) y devuelve (idx_a, idx_b)."""
        plt.tight_layout()
        plt.show()  # bloquea hasta cierre de ventana
        return self.idx_a, self.idx_b


def seleccionar_puntos_grafico(df):
    """API funcional: lanza el selector y devuelve los índices elegidos."""
    selector = SelectorPuntos(df)
    return selector.ejecutar()


# =============================================================================
# GRÁFICO COMPARATIVO (sin cambios respecto a la versión anterior)
# =============================================================================

def graficar_comparacion(cartografia, lat1, lon1, altura1, lat2, lon2, altura2,
                         resultado_fresnel, resultado_itm,
                         frecuencia_mhz=900, curvatura=True,
                         guardar="comparacion_fresnel_vs_itm.png",
                         nombre_a="Punto A", nombre_b="Punto B"):
    """Figura 1x2 con análisis Fresnel a la izquierda y Longley-Rice a la derecha."""
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
    
    radios = np.array([radio_fresnel(d, d_total - d, frecuencia_mhz)
                       for d in distancias])
    fresnel_sup = linea_vista + radios
    fresnel_inf = linea_vista - radios
    fresnel_60_sup = linea_vista + 0.6 * radios
    fresnel_60_inf = linea_vista - 0.6 * radios
    
    fig, (ax_f, ax_i) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    # ---------- Panel izquierdo: FRESNEL ----------
    ax_f.fill_between(distancias, 0, elev_efectiva, color='#8B4513', alpha=0.4)
    ax_f.plot(distancias, elev_efectiva, color='#654321', linewidth=2, label='Terreno')
    ax_f.fill_between(distancias, fresnel_inf, fresnel_sup,
                      color='#87CEEB', alpha=0.3, label='Fresnel 100%')
    ax_f.fill_between(distancias, fresnel_60_inf, fresnel_60_sup,
                      color='#4169E1', alpha=0.3, label='Fresnel 60%')
    
    estado_f = resultado_fresnel['estado']
    color_f = {'libre': '#228B22', 'fresnel_parcial': '#FFA500',
               'fresnel_60_obstruido': '#FF8C00', 'obstruido': '#DC143C'}.get(
                   estado_f, '#808080')
    estilo_f = '-' if estado_f in ('libre', 'fresnel_parcial') else '--'
    ax_f.plot(distancias, linea_vista, color=color_f, linewidth=2.5,
              linestyle=estilo_f, label='Línea de vista')
    
    ax_f.plot(0, h_a, 'b^', markersize=14, label=f'{nombre_a} ({altura1}m)')
    ax_f.plot(d_total, h_b, 'rv', markersize=14, label=f'{nombre_b} ({altura2}m)')
    ax_f.plot([0, 0], [elev_efectiva[0], h_a], 'b-', linewidth=3, alpha=0.7)
    ax_f.plot([d_total, d_total], [elev_efectiva[-1], h_b], 'r-', linewidth=3, alpha=0.7)
    
    if resultado_fresnel.get('obstruccion_km') is not None:
        idx = np.argmin(np.abs(distancias - resultado_fresnel['obstruccion_km']))
        ax_f.plot(distancias[idx], elev_efectiva[idx], 'rX',
                  markersize=18, markeredgewidth=3, label='Obstrucción')
    
    titulos_f = {
        'libre': ("✓ FRESNEL: libre", 'green'),
        'fresnel_parcial': ("◐ FRESNEL: parcial (>60%)", 'orange'),
        'fresnel_60_obstruido': ("⚠ FRESNEL: 60% obstruido", 'darkorange'),
        'obstruido': ("✗ FRESNEL: LOS obstruida", 'red'),
    }
    tit_f, col_f = titulos_f.get(estado_f, ("FRESNEL: ?", 'black'))
    ax_f.set_title(tit_f, fontsize=14, fontweight='bold', color=col_f)
    
    info_f = (
        f"═══ FRESNEL ═══\n"
        f"Distancia: {resultado_fresnel['distancia_km']:.2f} km\n"
        f"Margen LOS:  {resultado_fresnel['margen_los_m']:.1f} m\n"
        f"Margen F60%: {resultado_fresnel['margen_fresnel_60_m']:.1f} m\n"
        f"Margen F100%: {resultado_fresnel['margen_fresnel_100_m']:.1f} m\n"
    )
    ax_f.text(0.02, 0.97, info_f, transform=ax_f.transAxes, fontsize=9,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.95))
    
    ax_f.set_xlabel('Distancia (km)', fontsize=12)
    ax_f.set_ylabel('Elevación (m)', fontsize=12)
    ax_f.legend(loc='upper right', fontsize=9)
    ax_f.grid(True, alpha=0.3, linestyle='--')
    ax_f.set_ylim(bottom=0)
    
    # ---------- Panel derecho: LONGLEY-RICE ----------
    ax_i.fill_between(distancias, 0, elev_efectiva, color='#8B4513', alpha=0.4)
    ax_i.plot(distancias, elev_efectiva, color='#654321', linewidth=2, label='Terreno')
    
    estado_i = resultado_itm['estado']
    color_i = {'viable': '#228B22', 'no_viable': '#DC143C',
               'error': '#808080'}.get(estado_i, '#808080')
    estilo_i = '-' if estado_i == 'viable' else '--'
    ax_i.plot(distancias, linea_vista, color=color_i, linewidth=2.5,
              linestyle=estilo_i, label='Línea de vista')
    
    ax_i.plot(0, h_a, 'b^', markersize=14, label=f'{nombre_a} ({altura1}m)')
    ax_i.plot(d_total, h_b, 'rv', markersize=14, label=f'{nombre_b} ({altura2}m)')
    ax_i.plot([0, 0], [elev_efectiva[0], h_a], 'b-', linewidth=3, alpha=0.7)
    ax_i.plot([d_total, d_total], [elev_efectiva[-1], h_b], 'r-', linewidth=3, alpha=0.7)
    
    titulos_i = {
        'viable': ("✓ LONGLEY-RICE: viable", 'green'),
        'no_viable': ("✗ LONGLEY-RICE: no viable", 'red'),
        'error': ("⚠ LONGLEY-RICE: error", 'gray'),
    }
    tit_i, col_i = titulos_i.get(estado_i, titulos_i['error'])
    ax_i.set_title(tit_i, fontsize=14, fontweight='bold', color=col_i)
    
    if estado_i == 'error':
        info_i = f"═══ ERROR ITM ═══\n{resultado_itm.get('error', 'desconocido')}"
    else:
        info_i = (
            f"═══ LONGLEY-RICE ═══\n"
            f"Distancia: {resultado_itm['distancia_km']:.2f} km\n"
            f"FSL:        {resultado_itm['perdida_fsl_db']:.1f} dB\n"
            f"aref (med): {resultado_itm['perdida_aref_db']:.1f} dB\n"
            f"─────────────────\n"
            f"PÉRDIDA TOTAL:\n"
            f"  50%:  {resultado_itm['perdida_total_db_mediana']:.1f} dB\n"
            f"  90%:  {resultado_itm['perdida_total_db_p90']:.1f} dB\n"
            f"  95%:  {resultado_itm['perdida_total_db_p95']:.1f} dB\n"
            f"─────────────────\n"
            f"Presupuesto: {resultado_itm['perdida_max_db']:.1f} dB\n"
            f"Margen:      {resultado_itm['margen_db']:+.1f} dB\n"
        )
        if resultado_itm.get('kwx', 0) > 0:
            info_i += f"\n⚠ kwx={resultado_itm['kwx']}"
    
    ax_i.text(0.02, 0.97, info_i, transform=ax_i.transAxes, fontsize=9,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.95))
    
    ax_i.set_xlabel('Distancia (km)', fontsize=12)
    ax_i.legend(loc='upper right', fontsize=9)
    ax_i.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(
        f"Comparación de modelos RF: {nombre_a} ↔ {nombre_b} @ {frecuencia_mhz} MHz",
        fontsize=15, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(guardar, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Gráfico comparativo guardado: {guardar}")
    plt.close(fig)
    return guardar


# =============================================================================
# RESUMEN EN CONSOLA
# =============================================================================

def imprimir_comparacion(resultado_fresnel, resultado_itm, placa_a, placa_b):
    """Tabla comparativa en consola."""
    print("\n" + "="*70)
    print(f"   COMPARACIÓN: {placa_a} → {placa_b}")
    print("="*70)
    print(f"\n{'Métrica':<28} {'Fresnel':<20} {'Longley-Rice':<20}")
    print("-"*70)
    
    d = resultado_fresnel['distancia_km']
    print(f"{'Distancia (km)':<28} {d:<20.2f} {d:<20.2f}")
    
    est_f = resultado_fresnel['estado']
    est_i = resultado_itm['estado']
    print(f"{'Estado':<28} {est_f:<20} {est_i:<20}")
    
    margen_f60 = f"{resultado_fresnel['margen_fresnel_60_m']:.1f} m"
    perdida_p90 = resultado_itm.get('perdida_total_db_p90')
    perdida_p90_str = f"{perdida_p90:.1f} dB" if perdida_p90 is not None else "N/A"
    print(f"{'Métrica':<28} {'F60: '+margen_f60:<20} {'p90: '+perdida_p90_str:<20}")
    
    viable_f = resultado_fresnel.get('libre', False)
    viable_i = resultado_itm.get('viable', False)
    print(f"{'¿Enlace OK?':<28} {str(viable_f):<20} {str(viable_i):<20}")
    print("-"*70)
    
    if viable_f == viable_i:
        print(f"\n✓ AMBOS MODELOS CONCUERDAN: enlace "
              f"{'VIABLE' if viable_f else 'NO VIABLE'}")
    else:
        print(f"\n⚠ MODELOS DISCREPAN:")
        print(f"  Fresnel:      {'VIABLE' if viable_f else 'NO VIABLE'}")
        print(f"  Longley-Rice: {'VIABLE' if viable_i else 'NO VIABLE'}")
        print(f"  → Caso interesante para el paper.")
    print("="*70 + "\n")


# =============================================================================
# UTILIDADES CLI
# =============================================================================

def pedir_float(mensaje, default=None):
    raw = input(mensaje).strip()
    if not raw and default is not None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  ⚠ Valor inválido, usando {default}")
        return default


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("   COMPARACIÓN VISUAL: FRESNEL vs LONGLEY-RICE")
    print("="*70 + "\n")
    
    # Cartografía y datos
    carto = Cartografia()
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"\n✗ No se encontró {CSV_PATH}.")
        sys.exit(1)
    
    print(f"\n✓ CSV cargado: {len(df)} registros")
    if 'Origen_DataFrame' in df.columns:
        print("\nDistribución por tipo:")
        for tipo, n in df['Origen_DataFrame'].value_counts().items():
            print(f"  {tipo}: {n}")
    
    # --- Selección gráfica ---
    print("\n→ Abriendo selector gráfico. Selecciona dos puntos y cierra la ventana.")
    idx_a, idx_b = seleccionar_puntos_grafico(df)
    
    if idx_a is None or idx_b is None:
        print("\n✗ Selección incompleta. Saliendo.")
        sys.exit(0)
    
    a, b = df.iloc[idx_a], df.iloc[idx_b]
    placa_a, placa_b = str(a['Placa']), str(b['Placa'])
    lat1, lon1 = float(a['Latitud']), float(a['Longitud'])
    lat2, lon2 = float(b['Latitud']), float(b['Longitud'])
    
    print(f"\n✓ Puntos seleccionados:")
    print(f"  A (Concentrador): [{idx_a}] {placa_a} @ ({lat1:.6f}, {lon1:.6f})")
    print(f"  B (IED):          [{idx_b}] {placa_b} @ ({lat2:.6f}, {lon2:.6f})")
    
    # --- Parámetros del enlace ---
    print("\n" + "-"*40)
    print("PARÁMETROS DEL ENLACE")
    print("-"*40)
    altura_a = pedir_float(f"Altura concentrador en m [{ALTURA_CONC_DEFAULT}]: ",
                           ALTURA_CONC_DEFAULT)
    altura_b = pedir_float(f"Altura IED en m          [{ALTURA_IED_DEFAULT}]: ",
                           ALTURA_IED_DEFAULT)
    fmhz = pedir_float(f"Frecuencia en MHz        [{FRECUENCIA_DEFAULT}]: ",
                       FRECUENCIA_DEFAULT)
    presupuesto = pedir_float(f"Presupuesto ITM en dB    [{PRESUPUESTO_DB_DEFAULT}]: ",
                              PRESUPUESTO_DB_DEFAULT)
    
    # --- Análisis ---
    print("\n" + "="*70)
    print("EJECUTANDO ANÁLISIS...")
    print("="*70)
    
    print("\n→ Análisis Fresnel...")
    lv_fresnel = LineaVistaFresnel(carto)
    res_fresnel = lv_fresnel.verificar(
        lat1, lon1, altura_a, lat2, lon2, altura_b, frecuencia_mhz=fmhz
    )
    print(f"   estado={res_fresnel['estado']}, "
          f"margen F60={res_fresnel['margen_fresnel_60_m']:.1f} m")
    
    print("→ Análisis Longley-Rice (ITM)...")
    lv_itm = LineaVistaITM(carto, perdida_max_db=presupuesto)
    res_itm = lv_itm.verificar(
        lat1, lon1, altura_a, lat2, lon2, altura_b, frecuencia_mhz=fmhz
    )
    if res_itm['estado'] != 'error':
        print(f"   estado={res_itm['estado']}, "
              f"pérdida total p90={res_itm['perdida_total_db_p90']:.1f} dB")
    else:
        print(f"   ERROR: {res_itm.get('error')}")
    
    # --- Salidas ---
    imprimir_comparacion(res_fresnel, res_itm, placa_a, placa_b)
    
    # Sanitizar nombres para el archivo (sin espacios ni caracteres raros)
    safe_a = "".join(c if c.isalnum() else "_" for c in placa_a)
    safe_b = "".join(c if c.isalnum() else "_" for c in placa_b)
    archivo = f"comparacion_{safe_a}_{safe_b}.png"
    
    graficar_comparacion(
        carto, lat1, lon1, altura_a, lat2, lon2, altura_b,
        res_fresnel, res_itm, frecuencia_mhz=fmhz,
        guardar=archivo, nombre_a=placa_a, nombre_b=placa_b
    )


if __name__ == "__main__":
    main()