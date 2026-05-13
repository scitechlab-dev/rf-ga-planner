"""
Exportación de resultados
=========================

Generación de archivos KML para visualización en Google Earth, Radio Mobile y
herramientas GIS.

Este módulo es agnóstico del modelo de propagación: recibe dicts con datos de
la red ya calculados (concentradores, IEDs, márgenes) y produce KML. NO
realiza cálculos RF ni accede a SRTM.

Dos funciones principales:
- generar_kml_agrupado: estructura jerárquica (una carpeta por concentrador,
  con sus IEDs y enlaces). Recomendado para resultados del GA.
- generar_kml: estructura plana (tres carpetas: concentradores, IEDs, enlaces).
  Útil para pruebas individuales o exportes simples.

Diagnóstico de fallas en este módulo:
- KML no se abre en Google Earth: revisa codificación UTF-8 y caracteres
  especiales en placas/descripciones.
- Iconos no aparecen: requiere conexión a internet (los iconos se cargan desde
  maps.google.com). Para uso offline, descarga los PNG y usa rutas locales.
- Líneas todas del mismo color: revisa el campo margen_fresnel_60 en los dicts
  de entrada (los umbrales son >20m verde, >5m amarillo, resto naranja).
"""


# =============================================================================
# UMBRALES PARA COLORES DE ENLACE
# =============================================================================
# Margen Fresnel 60% que diferencia los niveles de calidad visual.

UMBRAL_BUENO = 20    # margen > 20 m → línea verde
UMBRAL_MEDIO = 5     # margen > 5 m  → línea amarilla
                     # resto         → línea naranja (marginal)


def _color_estilo_por_margen(margen):
    """Devuelve el id de estilo KML según el margen Fresnel 60% (m)."""
    if margen > UMBRAL_BUENO:
        return "linea_buena"
    elif margen > UMBRAL_MEDIO:
        return "linea_media"
    else:
        return "linea_marginal"


# =============================================================================
# BLOQUE DE ESTILOS KML (compartido)
# =============================================================================

_ESTILOS_KML = '''
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
'''


# =============================================================================
# KML AGRUPADO (resultado del GA)
# =============================================================================

def generar_kml_agrupado(datos_red: dict, archivo_salida: str = "red_rf.kml"):
    """
    Genera un KML con estructura jerárquica: una carpeta por concentrador con
    sus IEDs y enlaces, más una carpeta extra para IEDs sin cobertura.
    
    Parámetros
    ----------
    datos_red : dict
        Estructura esperada:
        {
            'concentradores': {
                'PLACA_CONC': {
                    'lat': float, 'lon': float, 'tipo': str,
                    'es_fijo': bool,
                    'ieds': [
                        {'placa': str, 'lat': float, 'lon': float,
                         'tipo': str, 'distancia_km': float,
                         'margen_fresnel_60': float}, ...
                    ]
                }, ...
            },
            'sin_cobertura': [
                {'placa': str, 'lat': float, 'lon': float, 'tipo': str}, ...
            ]
        }
    archivo_salida : str
        Ruta del KML a generar.
    
    Retorna
    -------
    str : ruta del archivo guardado.
    """
    concentradores = datos_red.get('concentradores', {})
    sin_cobertura = datos_red.get('sin_cobertura', [])
    
    kml = ['<?xml version="1.0" encoding="UTF-8"?>']
    kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    kml.append('<Document>')
    kml.append('<name>Red RF - Cobertura Agrupada</name>')
    kml.append(_ESTILOS_KML)
    
    # --- Carpeta por cada concentrador ---
    for placa_conc, datos_conc in concentradores.items():
        lat_conc = datos_conc['lat']
        lon_conc = datos_conc['lon']
        tipo_conc = datos_conc.get('tipo', 'CONCENTRADOR')
        es_fijo = datos_conc.get('es_fijo', False)
        ieds = datos_conc.get('ieds', [])
        
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
        
        # Subcarpeta de IEDs
        if ieds:
            kml.append(f'<Folder><name>IEDs ({n_ieds})</name>')
            
            for ied in ieds:
                placa_ied = ied['placa']
                lat_ied = ied['lat']
                lon_ied = ied['lon']
                tipo_ied = ied.get('tipo', 'IED')
                dist = ied.get('distancia_km', 0)
                margen = ied.get('margen_fresnel_60', 0)
                
                descripcion_ied = (f"Tipo: {tipo_ied}\nConcentrador: {placa_conc}\n"
                                   f"Distancia: {dist:.2f} km\nMargen F60: {margen:.1f} m")
                
                kml.append(f'''
            <Placemark>
                <name>{placa_ied}</name>
                <description>{descripcion_ied}</description>
                <styleUrl>#ied</styleUrl>
                <Point><coordinates>{lon_ied},{lat_ied},0</coordinates></Point>
            </Placemark>''')
            
            kml.append('</Folder>')
        
        # Subcarpeta de Enlaces
        if ieds:
            kml.append(f'<Folder><name>Enlaces ({n_ieds})</name>')
            
            for ied in ieds:
                placa_ied = ied['placa']
                lat_ied = ied['lat']
                lon_ied = ied['lon']
                dist = ied.get('distancia_km', 0)
                margen = ied.get('margen_fresnel_60', 0)
                estilo = _color_estilo_por_margen(margen)
                
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
            
            kml.append('</Folder>')
        
        kml.append('</Folder>')  # cierre carpeta concentrador
    
    # --- Carpeta: IEDs sin cobertura ---
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


# =============================================================================
# KML PLANO (pruebas individuales)
# =============================================================================

def generar_kml(datos_enlaces: list, archivo_salida: str = "red_rf.kml"):
    """
    Genera un KML plano con tres carpetas: concentradores, IEDs y enlaces.
    Útil para visualizar un solo enlace o un conjunto pequeño.
    
    Parámetros
    ----------
    datos_enlaces : list of dict
        Cada dict debe tener:
            placa_conc, lat_conc, lon_conc,
            placa_ied, lat_ied, lon_ied,
            margen_fresnel_60, distancia_km
        Use 'placa_ied' == '_DUMMY_' para registrar un concentrador sin enlace.
    archivo_salida : str
        Ruta del KML a generar.
    
    Retorna
    -------
    str : ruta del archivo guardado.
    """
    # Deduplicar puntos
    concentradores = {}
    ieds = {}
    
    for enlace in datos_enlaces:
        placa_c = enlace['placa_conc']
        if placa_c not in concentradores:
            concentradores[placa_c] = {
                'lat': enlace['lat_conc'],
                'lon': enlace['lon_conc']
            }
        
        placa_i = enlace['placa_ied']
        if placa_i != '_DUMMY_' and placa_i not in ieds:
            ieds[placa_i] = {
                'lat': enlace['lat_ied'],
                'lon': enlace['lon_ied']
            }
    
    kml = ['<?xml version="1.0" encoding="UTF-8"?>']
    kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    kml.append('<Document>')
    kml.append('<name>Red RF - Cobertura</name>')
    kml.append(_ESTILOS_KML)
    
    # Concentradores
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
    
    # IEDs
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
    
    # Enlaces
    kml.append('<Folder><name>Enlaces RF</name>')
    for enlace in datos_enlaces:
        if enlace['placa_ied'] == '_DUMMY_':
            continue
        
        margen = enlace.get('margen_fresnel_60', 0)
        estilo = _color_estilo_por_margen(margen)
        
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
    
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.write('\n'.join(kml))
    
    print(f"\n✓ KML generado: {archivo_salida}")
    return archivo_salida


# =============================================================================
# PRUEBAS DIRECTAS
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("   MÓDULO EXPORTACION - SMOKE TEST")
    print("="*60)
    print("\nEste módulo no carga datos por sí solo; importa desde main.py")
    print("o pásale dicts construidos manualmente.")
