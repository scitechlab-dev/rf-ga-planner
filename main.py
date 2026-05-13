"""
Main - Análisis RF y Optimización de Concentradores
====================================================
Con selección interactiva de tipos de dispositivo.
VERSIÓN 2.0: Soporte para concentradores fijos (SITIO DE REPETICION)

Modos:
  1. Prueba individual de enlace
  2. Optimización con algoritmo genético (con opción de concentradores fijos)
"""

from cartografia import Cartografia
from linea_vista_fresnel import LineaVistaFresnel
from exportacion import generar_kml, generar_kml_agrupado
from algoritmo_genetico import OptimizadorCobertura, ConfigGA
import pandas as pd

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
CSV_PATH = "datos.csv"
FRECUENCIA_MHZ = 900
DISTANCIA_MAX_KM = 30.0

# Tipo que identifica los sitios de repetición existentes
TIPO_SITIO_REPETICION = "SITIO DE REPETICION"

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def seleccionar_concentradores_fijos(df: pd.DataFrame) -> list:
    """
    Permite al usuario seleccionar qué SITIOS DE REPETICION usar como
    concentradores fijos (inamovibles) en el algoritmo genético.
    
    Retorna:
    --------
    Lista de índices de los puntos seleccionados como fijos, o lista vacía
    si el usuario no quiere usar fijos.
    """
    # Buscar todos los SITIO DE REPETICION
    if 'Origen_DataFrame' not in df.columns:
        print("⚠ No existe columna 'Origen_DataFrame'. No se pueden identificar sitios de repetición.")
        return []
    
    mask_repeticion = df['Origen_DataFrame'] == TIPO_SITIO_REPETICION
    sitios_repeticion = df[mask_repeticion]
    
    if len(sitios_repeticion) == 0:
        print(f"⚠ No se encontraron puntos de tipo '{TIPO_SITIO_REPETICION}'.")
        return []
    
    print("\n" + "="*70)
    print("CONFIGURACIÓN DE CONCENTRADORES FIJOS")
    print("="*70)
    print(f"\nSe encontraron {len(sitios_repeticion)} sitios de repetición existentes:")
    print("-"*70)
    
    # Mostrar lista con índices
    sitios_info = []
    for i, (idx, row) in enumerate(sitios_repeticion.iterrows()):
        placa = row['Placa']
        lat = row['Latitud']
        lon = row['Longitud']
        municipio = row.get('Municipio', 'N/A')
        print(f"  {i+1:2d}. {placa:<25} ({lat:.6f}, {lon:.6f}) - {municipio}")
        sitios_info.append({
            'numero': i + 1,
            'indice_df': idx,
            'placa': placa,
            'lat': lat,
            'lon': lon
        })
    
    print("-"*70)
    print("\n¿Desea usar algunos de estos sitios como concentradores FIJOS?")
    print("(Los concentradores fijos NO pueden ser removidos por el algoritmo)")
    print("-"*70)
    
    usar_fijos = input("\n¿Usar concentradores fijos? (s/n): ").strip().lower()
    
    if usar_fijos not in ['s', 'si', 'sí', 'y', 'yes']:
        print("\n→ No se usarán concentradores fijos. El AG buscará desde cero.")
        return []
    
    # Selección de cuáles usar
    print("\n" + "-"*70)
    print("SELECCIÓN DE SITIOS FIJOS")
    print("-"*70)
    print("Opciones:")
    print("  - Ingrese números separados por coma (ej: 1,3,5)")
    print("  - Ingrese rango con guión (ej: 1-7)")
    print("  - Ingrese 'todos' para seleccionar todos")
    print("  - Ingrese 'ninguno' para cancelar")
    print("-"*70)
    
    while True:
        seleccion = input("\n¿Cuáles sitios usar como fijos? ").strip().lower()
        
        if seleccion == 'ninguno':
            print("\n→ Cancelado. No se usarán concentradores fijos.")
            return []
        
        if seleccion == 'todos':
            indices_seleccionados = [s['indice_df'] for s in sitios_info]
            break
        
        try:
            numeros_seleccionados = set()
            
            # Parsear entrada
            partes = seleccion.replace(' ', '').split(',')
            
            for parte in partes:
                if '-' in parte:
                    # Rango
                    inicio, fin = parte.split('-')
                    inicio, fin = int(inicio), int(fin)
                    numeros_seleccionados.update(range(inicio, fin + 1))
                else:
                    # Individual
                    numeros_seleccionados.add(int(parte))
            
            # Validar números
            n_sitios = len(sitios_info)
            if all(1 <= n <= n_sitios for n in numeros_seleccionados):
                indices_seleccionados = [
                    sitios_info[n-1]['indice_df'] 
                    for n in sorted(numeros_seleccionados)
                ]
                break
            else:
                print(f"  ⚠ Números fuera de rango. Use valores entre 1 y {n_sitios}")
        
        except ValueError:
            print("  ⚠ Formato inválido. Use números, comas y/o guiones.")
    
    # Mostrar resumen de selección
    print("\n" + "-"*70)
    print("RESUMEN DE CONCENTRADORES FIJOS SELECCIONADOS:")
    print("-"*70)
    
    for idx in indices_seleccionados:
        row = df.loc[idx]
        placa = row['Placa']
        lat = row['Latitud']
        lon = row['Longitud']
        print(f"  ✓ {placa} @ ({lat:.6f}, {lon:.6f})")
    
    print(f"\n→ Total: {len(indices_seleccionados)} concentradores fijos")
    print("-"*70)
    
    confirmar = input("\n¿Confirmar selección? (s/n): ").strip().lower()
    if confirmar not in ['s', 'si', 'sí', 'y', 'yes']:
        print("\nReiniciando selección...")
        return seleccionar_concentradores_fijos(df)
    
    return indices_seleccionados


# =============================================================================
# INICIALIZACIÓN
# =============================================================================
print("\n" + "="*60)
print("   SISTEMA DE ANÁLISIS RF - EL SALVADOR")
print("   Versión 2.0 - Con soporte para concentradores fijos")
print("="*60 + "\n")

# Cargar cartografía SRTM (una sola vez, reutilizable por cualquier modelo RF)
carto = Cartografia()

# Crear analizador RF basado en Fresnel
lv = LineaVistaFresnel(carto)

# Cargar CSV para mostrar info
df = pd.read_csv(CSV_PATH)
n_registros = len(df)

print(f"\n✓ CSV cargado: {CSV_PATH}")
print(f"✓ Registros encontrados: {n_registros}")

# Mostrar tipos disponibles
if 'Origen_DataFrame' in df.columns:
    print("\n✓ Tipos de dispositivos:")
    for tipo, count in df['Origen_DataFrame'].value_counts().items():
        marca = " ★" if tipo == TIPO_SITIO_REPETICION else ""
        print(f"    - {tipo}: {count}{marca}")

# =============================================================================
# CONFIGURACIÓN DE ALTURAS
# =============================================================================
print("\n" + "-"*40)
print("CONFIGURACIÓN DE ALTURAS")
print("-"*40)

altura_ied_input = input("Altura antena IED en metros [8]: ").strip()
ALTURA_IED = float(altura_ied_input) if altura_ied_input else 8

altura_conc_input = input("Altura torre concentrador en metros [30]: ").strip()
ALTURA_CONCENTRADOR = float(altura_conc_input) if altura_conc_input else 30

print(f"\n✓ Altura IED: {ALTURA_IED} m")
print(f"✓ Altura Concentrador: {ALTURA_CONCENTRADOR} m")

# =============================================================================
# SELECCIÓN DE MODO
# =============================================================================
print("\n" + "-"*40)
print("SELECCIONE MODO DE OPERACIÓN")
print("-"*40)
print("  1. Prueba individual de enlace")
print("  2. Optimización GA (selección interactiva de IEDs)")
print("-"*40)

while True:
    try:
        modo = int(input("\nIngrese opción (1 o 2): "))
        if modo in [1, 2]:
            break
        print("  ⚠ Opción inválida")
    except ValueError:
        print("  ⚠ Ingrese un número válido")


# =============================================================================
# MODO 1: PRUEBA INDIVIDUAL
# =============================================================================
if modo == 1:
    print("\n" + "="*40)
    print("MODO: PRUEBA INDIVIDUAL")
    print("="*40)
    
    print(f"\nPrimeros 10 registros:")
    cols_mostrar = ['Origen_DataFrame', 'Placa', 'Latitud', 'Longitud']
    cols_existentes = [c for c in cols_mostrar if c in df.columns]
    print(df[cols_existentes].head(10).to_string(index=True))
    
    print("\n" + "-"*40)
    print("SELECCIÓN DE PUNTOS A EVALUAR")
    print("-"*40)
    
    while True:
        try:
            indice_a = int(input(f"\nIngrese índice punto A - Concentrador (0 a {n_registros-1}): "))
            if 0 <= indice_a < n_registros:
                break
            print(f"  ⚠ Índice fuera de rango")
        except ValueError:
            print("  ⚠ Ingrese un número válido")
    
    while True:
        try:
            indice_b = int(input(f"Ingrese índice punto B - IED (0 a {n_registros-1}): "))
            if 0 <= indice_b < n_registros:
                break
            print(f"  ⚠ Índice fuera de rango")
        except ValueError:
            print("  ⚠ Ingrese un número válido")
    
    graficar_input = input("\n¿Generar gráfico? (s/n): ").strip().lower()
    graficar = graficar_input in ['s', 'si', 'sí', 'y', 'yes']
    
    print("\n" + "="*40)
    print("EJECUTANDO ANÁLISIS...")
    print("="*40)
    
    resultado = lv.funcion_objetivo_rf(
        csv_path=CSV_PATH,
        indice_a=indice_a,
        indice_b=indice_b,
        frecuencia_mhz=FRECUENCIA_MHZ,
        altura_ied=ALTURA_IED,
        altura_concentrador=ALTURA_CONCENTRADOR,
        graficar=graficar
    )
    
    print("\n" + "="*40)
    print("RESULTADOS")
    print("="*40)
    print(f"Enlace: {resultado['placa_a']} → {resultado['placa_b']}")
    print(f"Estado: {resultado['estado'].upper()}")
    print(f"Distancia: {resultado['distancia_km']:.2f} km")
    print(f"Fresnel 60% libre: {resultado['fresnel_60_libre']}")
    print(f"Margen LOS: {resultado['margen_los_m']:.1f} m")
    print(f"Margen Fresnel 60%: {resultado['margen_fresnel_60_m']:.1f} m")
    print(f"Margen Fresnel 100%: {resultado['margen_fresnel_100_m']:.1f} m")
    
    if resultado.get('archivo_grafico'):
        print(f"\n✓ Gráfico: {resultado['archivo_grafico']}")
    
    # Generar KML
    punto_a = df.iloc[indice_a]
    punto_b = df.iloc[indice_b]
    
    datos_kml = [{
        'placa_conc': resultado['placa_a'],
        'lat_conc': float(punto_a['Latitud']),
        'lon_conc': float(punto_a['Longitud']),
        'placa_ied': resultado['placa_b'],
        'lat_ied': float(punto_b['Latitud']),
        'lon_ied': float(punto_b['Longitud']),
        'margen_fresnel_60': resultado['margen_fresnel_60_m'],
        'distancia_km': resultado['distancia_km']
    }]
    
    archivo_kml = f"enlace_{resultado['placa_a']}_{resultado['placa_b']}.kml"
    generar_kml(datos_kml, archivo_kml)
    
    print("="*40 + "\n")


# =============================================================================
# MODO 2: OPTIMIZACIÓN GA
# =============================================================================
elif modo == 2:
    print("\n" + "="*40)
    print("MODO: OPTIMIZACIÓN GENÉTICA")
    print("="*40)
    
    # =========================================================================
    # NUEVO: Preguntar por concentradores fijos
    # =========================================================================
    concentradores_fijos = seleccionar_concentradores_fijos(df)
    
    # Configuración del GA
    print("\n" + "="*50)
    print("CONFIGURACIÓN DEL ALGORITMO GENÉTICO")
    print("="*50)
    print(f"  - Población: 100 individuos")
    print(f"  - Generaciones sugeridas: 200")
    print(f"  - Distancia máxima enlace: {DISTANCIA_MAX_KM} km")
    print(f"  - Frecuencia: {FRECUENCIA_MHZ} MHz")
    print(f"  - Altura concentrador: {ALTURA_CONCENTRADOR} m")
    print(f"  - Altura IED: {ALTURA_IED} m")
    if concentradores_fijos:
        print(f"  - Concentradores fijos: {len(concentradores_fijos)}")
    
    # Permitir ajustar generaciones
    gen_input = input("\nGeneraciones [200]: ").strip()
    generaciones = int(gen_input) if gen_input else 200
    
    # Permitir ajustar máximo de concentradores
    if concentradores_fijos:
        print(f"\n(Nota: Tienes {len(concentradores_fijos)} concentradores fijos)")
        print("El límite máximo incluye los fijos + adicionales")
    max_conc_input = input("Máximo concentradores totales [sin límite]: ").strip()
    max_concentradores = int(max_conc_input) if max_conc_input else None
    
    # Crear configuración
    config = ConfigGA(
        poblacion_size=100,
        generaciones=generaciones,
        prob_mutacion=0.1,
        prob_cruce=0.8,
        elitismo=5,
        max_sin_mejora=30,
        distancia_max_km=DISTANCIA_MAX_KM,
        max_concentradores=max_concentradores
    )
    
    # Crear optimizador CON SELECCIÓN INTERACTIVA Y CONCENTRADORES FIJOS
    optimizador = OptimizadorCobertura(
        linea_vista=lv,
        csv_path=CSV_PATH,
        config=config,
        frecuencia_mhz=FRECUENCIA_MHZ,
        altura_ied=ALTURA_IED,
        altura_concentrador=ALTURA_CONCENTRADOR,
        interactivo=True,
        concentradores_fijos=concentradores_fijos if concentradores_fijos else None
    )
    
    # Confirmar inicio
    if concentradores_fijos:
        msg = f"\n¿Iniciar optimización con {generaciones} generaciones y {len(concentradores_fijos)} concentradores fijos? (s/n): "
    else:
        msg = f"\n¿Iniciar optimización con {generaciones} generaciones? (s/n): "
    
    continuar = input(msg).strip().lower()
    if continuar not in ['s', 'si', 'sí', 'y', 'yes']:
        print("Optimización cancelada.")
        exit()
    
    # Ejecutar optimización
    resultado = optimizador.optimizar()
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("RESULTADOS DE OPTIMIZACIÓN")
    print("="*70)
    
    if resultado['exito']:
        print("\n✓ SOLUCIÓN ENCONTRADA - 100% COBERTURA")
    else:
        print(f"\n⚠ COBERTURA PARCIAL: {resultado['cobertura']*100:.1f}%")
    
    print(f"\nIEDs cubiertos: {resultado['n_cubiertos']} de {resultado['n_ieds']}")
    print(f"\nConcentradores totales: {resultado['n_concentradores']}")
    if 'n_fijos' in resultado and resultado['n_fijos'] > 0:
        print(f"  → Fijos (inamovibles): {resultado['n_fijos']}")
        print(f"  → Adicionales (encontrados por AG): {resultado['n_adicionales']}")
    
    print(f"\nDistancia promedio a concentrador: {resultado['distancia_promedio_km']:.2f} km")
    print(f"Margen Fresnel promedio: {resultado.get('margen_promedio_m', 0):.1f} m")
    
    print(f"\nConcentradores seleccionados:")
    for i, (idx, placa, tipo) in enumerate(zip(
        resultado['concentradores_indices'], 
        resultado['concentradores_placas'],
        resultado['concentradores_tipos']
    )):
        lat = df.iloc[idx]['Latitud']
        lon = df.iloc[idx]['Longitud']
        
        # Marcar si es fijo
        es_fijo = idx in resultado.get('concentradores_fijos_indices', [])
        marca_fijo = " [FIJO]" if es_fijo else ""
        
        print(f"  {i+1}. [{tipo}] {placa}{marca_fijo} @ ({lat:.6f}, {lon:.6f})")
    
    # Generar detalle de cobertura
    print("\n" + "-"*40)
    guardar_detalle = input("¿Guardar detalle de cobertura en CSV? (s/n): ").strip().lower()
    
    if guardar_detalle in ['s', 'si', 'sí', 'y', 'yes']:
        detalle_df = optimizador.detalle_cobertura(resultado)
        archivo_detalle = "cobertura_detalle.csv"
        detalle_df.to_csv(archivo_detalle, index=False)
        print(f"\n✓ Detalle guardado: {archivo_detalle}")
        
        # Mostrar resumen
        print(f"\nResumen por rol:")
        print(detalle_df.groupby('Rol').size().to_string())
        
        print(f"\nResumen por tipo de IED:")
        print(detalle_df.groupby('Tipo_IED').size().to_string())
    
    # Preguntar si graficar algunos enlaces
    print("\n" + "-"*40)
    graficar_enlaces = input("¿Graficar algunos enlaces? (s/n): ").strip().lower()
    
    if graficar_enlaces in ['s', 'si', 'sí', 'y', 'yes']:
        n_graficos = int(input("¿Cuántos enlaces graficar? (ej: 5): ") or "5")
        
        detalle_df = optimizador.detalle_cobertura(resultado)
        ieds = detalle_df[detalle_df['Rol'] == 'IED'].head(n_graficos)
        
        for _, row in ieds.iterrows():
            if row['Concentrador_asignado'] != 'SIN COBERTURA':
                conc_idx = resultado['concentradores_placas'].index(row['Concentrador_asignado'])
                conc_idx_real = resultado['concentradores_indices'][conc_idx]
                
                lv.funcion_objetivo_rf(
                    csv_path=CSV_PATH,
                    indice_a=conc_idx_real,
                    indice_b=row['Indice'],
                    frecuencia_mhz=FRECUENCIA_MHZ,
                    altura_ied=ALTURA_IED,
                    altura_concentrador=ALTURA_CONCENTRADOR,
                    graficar=True
                )
    
    # Generar KML con toda la red (AGRUPADO POR CONCENTRADOR)
    print("\n" + "-"*40)
    print("GENERANDO KML DE LA RED...")
    
    detalle_df = optimizador.detalle_cobertura(resultado)
    
    # Construir estructura para KML agrupado
    datos_red = {
        'concentradores': {},
        'sin_cobertura': []
    }
    
    # Primero, registrar todos los concentradores
    for conc_idx, conc_placa in zip(
        resultado['concentradores_indices'],
        resultado['concentradores_placas']
    ):
        conc_data = df.iloc[conc_idx]
        es_fijo = conc_idx in resultado.get('concentradores_fijos_indices', [])
        
        datos_red['concentradores'][conc_placa] = {
            'lat': float(conc_data['Latitud']),
            'lon': float(conc_data['Longitud']),
            'tipo': conc_data.get('Origen_DataFrame', 'CONCENTRADOR'),
            'es_fijo': es_fijo,
            'ieds': []
        }
    
    # Asignar IEDs a sus concentradores
    for _, row in detalle_df.iterrows():
        # Solo procesar IEDs puros (no los que también son concentradores)
        if row['Rol'] == 'IED':
            conc_asignado = row['Concentrador_asignado']
            
            if conc_asignado == 'SIN COBERTURA':
                # IED sin cobertura
                ied_data = df.iloc[row['Indice']]
                datos_red['sin_cobertura'].append({
                    'placa': row['IED'],
                    'lat': float(ied_data['Latitud']),
                    'lon': float(ied_data['Longitud']),
                    'tipo': row['Tipo_IED']
                })
            else:
                # IED con cobertura - agregarlo a su concentrador
                if conc_asignado in datos_red['concentradores']:
                    ied_data = df.iloc[row['Indice']]
                    
                    try:
                        margen = float(row['Margen_F60_m'])
                    except:
                        margen = 0
                    
                    try:
                        distancia = float(row['Distancia_km'])
                    except:
                        distancia = 0
                    
                    datos_red['concentradores'][conc_asignado]['ieds'].append({
                        'placa': row['IED'],
                        'lat': float(ied_data['Latitud']),
                        'lon': float(ied_data['Longitud']),
                        'tipo': row['Tipo_IED'],
                        'distancia_km': distancia,
                        'margen_fresnel_60': margen
                    })
    
    # Generar KML agrupado
    generar_kml_agrupado(datos_red, "red_rf_optimizada.kml")
    
    # Resumen final
    print("\n" + "="*70)
    print("OPTIMIZACIÓN COMPLETADA")
    print("="*70)
    if resultado.get('n_fijos', 0) > 0:
        print(f"\nResumen: {resultado['n_fijos']} concentradores fijos + {resultado['n_adicionales']} adicionales")
        print(f"         = {resultado['n_concentradores']} concentradores totales para {resultado['cobertura']*100:.0f}% cobertura")
    print("="*70 + "\n")
