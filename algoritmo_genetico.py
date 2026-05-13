"""
Algoritmo Genético para Selección Óptima de Concentradores RF
=============================================================
Minimiza cantidad de concentradores garantizando 100% cobertura.
Con selección interactiva de tipos de IED.

VERSIÓN ULTRA-OPTIMIZADA:
- Fitness con dos regímenes (cobertura vs minimización)
- Depuración exhaustiva inteligente
- Validación final conservadora con búsqueda exhaustiva
- Límite duro configurable de concentradores
- Mayor paciencia y refinamiento
- SOPORTE PARA CONCENTRADORES FIJOS (v2.0)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Set, Optional
import random
from dataclasses import dataclass
from itertools import combinations


@dataclass
class ConfigGA:
    """Configuración del algoritmo genético."""
    poblacion_size: int = 100
    generaciones: int = 500
    prob_mutacion: float = 0.15      # 15% de individuos mutan
    prob_cruce: float = 0.8
    elitismo: int = 5
    torneo_size: int = 3
    max_sin_mejora: int = 100        # MÁS paciencia
    distancia_max_km: float = 30.0
    max_concentradores: Optional[int] = 5  # Límite duro
    seed: int = None


def seleccionar_tipos_ied(df: pd.DataFrame) -> List[str]:
    """
    Muestra los tipos disponibles y permite al usuario seleccionar
    cuáles considerar como IEDs (dispositivos que necesitan cobertura).
    
    Retorna:
    --------
    Lista de tipos seleccionados como IED
    """
    if 'Origen_DataFrame' not in df.columns:
        print("⚠ No existe columna 'Origen_DataFrame'. Todos los puntos serán IEDs.")
        return []
    
    # Obtener tipos únicos con conteo
    tipos_conteo = df['Origen_DataFrame'].value_counts()
    tipos_lista = tipos_conteo.index.tolist()
    
    print("\n" + "="*60)
    print("SELECCIÓN DE TIPOS DE DISPOSITIVO")
    print("="*60)
    print("\nTipos disponibles en el CSV:")
    print("-"*60)
    
    for i, tipo in enumerate(tipos_lista):
        count = tipos_conteo[tipo]
        print(f"  {i+1:2d}. {tipo:<40} ({count} puntos)")
    
    print("-"*60)
    print("\nOpciones de selección:")
    print("  - Ingrese números separados por coma (ej: 1,3,5)")
    print("  - Ingrese rango con guión (ej: 1-5)")
    print("  - Ingrese 'todos' para seleccionar todos")
    print("  - Ingrese 'ninguno' para no seleccionar ninguno (todos serán solo concentradores)")
    print("-"*60)
    
    while True:
        seleccion = input("\n¿Cuáles tipos considerar como IED? ").strip().lower()
        
        if seleccion == 'todos':
            tipos_seleccionados = tipos_lista.copy()
            break
        
        if seleccion == 'ninguno':
            tipos_seleccionados = []
            break
        
        try:
            indices_seleccionados = set()
            
            # Parsear entrada
            partes = seleccion.replace(' ', '').split(',')
            
            for parte in partes:
                if '-' in parte:
                    # Rango
                    inicio, fin = parte.split('-')
                    inicio, fin = int(inicio), int(fin)
                    indices_seleccionados.update(range(inicio, fin + 1))
                else:
                    # Individual
                    indices_seleccionados.add(int(parte))
            
            # Validar índices
            if all(1 <= i <= len(tipos_lista) for i in indices_seleccionados):
                tipos_seleccionados = [tipos_lista[i-1] for i in sorted(indices_seleccionados)]
                break
            else:
                print(f"  ⚠ Índices fuera de rango. Use valores entre 1 y {len(tipos_lista)}")
        
        except ValueError:
            print("  ⚠ Formato inválido. Use números, comas y/o guiones.")
    
    # Mostrar resumen
    print("\n" + "-"*60)
    print("RESUMEN DE SELECCIÓN:")
    print("-"*60)
    
    total_ieds = 0
    if tipos_seleccionados:
        print("\n✓ Tipos seleccionados como IED (necesitan cobertura):")
        for tipo in tipos_seleccionados:
            count = tipos_conteo[tipo]
            total_ieds += count
            print(f"    - {tipo}: {count} puntos")
    else:
        print("\n⚠ Ningún tipo seleccionado como IED")
    
    tipos_concentrador = [t for t in tipos_lista if t not in tipos_seleccionados]
    if tipos_concentrador:
        print("\n✓ Tipos que serán SOLO concentradores candidatos:")
        for tipo in tipos_concentrador:
            count = tipos_conteo[tipo]
            print(f"    - {tipo}: {count} puntos")
    
    print(f"\n→ Total IEDs a cubrir: {total_ieds}")
    print(f"→ Total puntos disponibles como concentrador: {len(df)}")
    print("-"*60)
    
    confirmar = input("\n¿Confirmar selección? (s/n): ").strip().lower()
    if confirmar not in ['s', 'si', 'sí', 'y', 'yes']:
        print("\nReiniciando selección...")
        return seleccionar_tipos_ied(df)
    
    return tipos_seleccionados


class OptimizadorCobertura:
    """
    Optimizador de cobertura RF usando algoritmo genético.
    Soporta concentradores fijos que no pueden ser removidos.
    """
    
    def __init__(self, linea_vista, csv_path: str, config: ConfigGA = None,
                 frecuencia_mhz: float = 900,
                 altura_ied: float = 8,
                 altura_concentrador: float = 30,
                 tipos_ied: Optional[List[str]] = None,
                 interactivo: bool = True,
                 concentradores_fijos: Optional[List[int]] = None):
        """
        Inicializa el optimizador.
        
        Parámetros:
        -----------
        concentradores_fijos : List[int], opcional
            Lista de índices de puntos que SIEMPRE deben ser concentradores.
            El AG no puede desactivarlos ni removerlos.
        """
        self.lv = linea_vista
        self.config = config or ConfigGA()
        self.frecuencia_mhz = frecuencia_mhz
        self.altura_ied = altura_ied
        self.altura_concentrador = altura_concentrador
        
        # Concentradores fijos (inamovibles)
        self.concentradores_fijos = set(concentradores_fijos) if concentradores_fijos else set()
        self.n_fijos = len(self.concentradores_fijos)
        
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
        
        # Cargar datos
        self.df = pd.read_csv(csv_path)
        self.n_puntos = len(self.df)
        self.placas = self.df['Placa'].tolist()
        
        # Obtener tipos
        if 'Origen_DataFrame' in self.df.columns:
            self.tipos = self.df['Origen_DataFrame'].tolist()
        else:
            self.tipos = ['DESCONOCIDO'] * self.n_puntos
        
        # Validar concentradores fijos
        if self.concentradores_fijos:
            invalidos = [i for i in self.concentradores_fijos if i < 0 or i >= self.n_puntos]
            if invalidos:
                raise ValueError(f"Índices de concentradores fijos inválidos: {invalidos}")
            
            print("\n" + "="*60)
            print("CONCENTRADORES FIJOS CONFIGURADOS")
            print("="*60)
            print(f"Se han definido {self.n_fijos} concentradores fijos:")
            for idx in sorted(self.concentradores_fijos):
                placa = self.placas[idx]
                tipo = self.tipos[idx]
                lat = self.df.iloc[idx]['Latitud']
                lon = self.df.iloc[idx]['Longitud']
                print(f"  • [{tipo}] {placa} @ ({lat:.6f}, {lon:.6f})")
            print("="*60)
        
        # Determinar tipos IED
        if tipos_ied is not None:
            self.tipos_ied = tipos_ied
        elif interactivo:
            self.tipos_ied = seleccionar_tipos_ied(self.df)
        else:
            # Todos son IEDs
            self.tipos_ied = list(self.df['Origen_DataFrame'].unique()) if 'Origen_DataFrame' in self.df.columns else []
        
        # Identificar IEDs reales
        self._identificar_ieds()
        
        # Pre-calcular matriz de cobertura
        self.matriz_cobertura = None
        self.matriz_distancias = None
        self.matriz_margenes = None
        self._precalcular_matriz()
    
    def _identificar_ieds(self):
        """Identifica cuáles puntos son IEDs reales vs solo candidatos a concentrador."""
        if 'Origen_DataFrame' in self.df.columns and self.tipos_ied:
            self.es_ied = self.df['Origen_DataFrame'].isin(self.tipos_ied)
        elif not self.tipos_ied:
            # Si no hay tipos seleccionados, ninguno es IED
            self.es_ied = pd.Series([False] * self.n_puntos)
        else:
            # Si no existe la columna, todos son IEDs
            self.es_ied = pd.Series([True] * self.n_puntos)
        
        self.indices_ieds = np.where(self.es_ied)[0]
        self.n_ieds = len(self.indices_ieds)
        
        print(f"\n✓ Puntos totales: {self.n_puntos}")
        print(f"✓ IEDs a conectar: {self.n_ieds}")
        print(f"✓ Solo candidatos a concentrador: {self.n_puntos - self.n_ieds}")
        if self.n_fijos > 0:
            print(f"✓ Concentradores fijos: {self.n_fijos}")
        
        if self.n_ieds == 0:
            print("\n⚠ ADVERTENCIA: No hay IEDs seleccionados para cubrir.")
    
    def _precalcular_matriz(self):
        """Pre-calcula matriz de cobertura: todos como concentrador → solo IEDs como destino."""
        if self.n_ieds == 0:
            print("\n⚠ No hay IEDs para calcular matriz de cobertura.")
            self.matriz_cobertura = np.zeros((self.n_puntos, 0), dtype=bool)
            self.matriz_distancias = np.full((self.n_puntos, 0), np.inf)
            self.matriz_margenes = np.full((self.n_puntos, 0), -np.inf)
            return
        
        print("\n" + "="*50)
        print("PRE-CÁLCULO DE MATRIZ DE COBERTURA")
        print("="*50)
        
        n = self.n_puntos
        n_ieds = self.n_ieds
        
        self.matriz_cobertura = np.zeros((n, n_ieds), dtype=bool)
        self.matriz_distancias = np.full((n, n_ieds), np.inf)
        self.matriz_margenes = np.full((n, n_ieds), -np.inf)
        
        total_pares = n * n_ieds
        pares_evaluados = 0
        pares_viables = 0
        
        print(f"Evaluando {total_pares} pares (concentrador → IED)...")
        
        for i in range(n):
            punto_conc = self.df.iloc[i]
            lat1 = float(punto_conc['Latitud'])
            lon1 = float(punto_conc['Longitud'])
            
            for j_idx, j in enumerate(self.indices_ieds):
                if i == j:
                    continue
                
                punto_ied = self.df.iloc[j]
                lat2 = float(punto_ied['Latitud'])
                lon2 = float(punto_ied['Longitud'])
                
                resultado = self.lv.verificar(
                    lat1, lon1, self.altura_concentrador,
                    lat2, lon2, self.altura_ied,
                    frecuencia_mhz=self.frecuencia_mhz
                )
                
                distancia = resultado['distancia_km']
                
                if distancia <= self.config.distancia_max_km and resultado['fresnel_60_libre']:
                    self.matriz_cobertura[i, j_idx] = True
                    self.matriz_distancias[i, j_idx] = distancia
                    self.matriz_margenes[i, j_idx] = resultado['margen_fresnel_60_m']
                    pares_viables += 1
                
                pares_evaluados += 1
                
                if pares_evaluados % max(1, total_pares // 10) == 0:
                    pct = 100 * pares_evaluados / total_pares
                    print(f"  Progreso: {pct:.0f}% ({pares_viables} enlaces viables)")
        
        print(f"\n✓ Pre-cálculo completado")
        print(f"  Enlaces viables: {pares_viables} de {total_pares} ({100*pares_viables/total_pares:.1f}%)")
        
        ieds_cubribles = np.sum(np.any(self.matriz_cobertura, axis=0))
        print(f"  IEDs cubribles: {ieds_cubribles} de {n_ieds}")
        
        # Mostrar cobertura de concentradores fijos
        if self.n_fijos > 0:
            ieds_por_fijos = np.zeros(n_ieds, dtype=bool)
            for fijo_idx in self.concentradores_fijos:
                ieds_por_fijos |= self.matriz_cobertura[fijo_idx, :]
            
            # Contar IEDs que son concentradores fijos (auto-cubiertos)
            ieds_autocubiertos = sum(1 for j_idx, j in enumerate(self.indices_ieds) 
                                     if j in self.concentradores_fijos)
            
            n_cubiertos_fijos = np.sum(ieds_por_fijos) + ieds_autocubiertos
            print(f"\n  → Cobertura solo con fijos: {n_cubiertos_fijos} de {n_ieds} IEDs ({100*n_cubiertos_fijos/n_ieds:.1f}%)")
            
            if n_cubiertos_fijos >= n_ieds:
                print("  ★ ¡Los concentradores fijos ya cubren el 100%!")
        
        if ieds_cubribles < n_ieds:
            ieds_sin_cobertura = np.where(~np.any(self.matriz_cobertura, axis=0))[0]
            print(f"\n⚠ ADVERTENCIA: {n_ieds - ieds_cubribles} IEDs sin cobertura posible:")
            for idx in ieds_sin_cobertura[:10]:
                ied_real_idx = self.indices_ieds[idx]
                tipo = self.tipos[ied_real_idx]
                print(f"    - {self.placas[ied_real_idx]} [{tipo}] (índice {ied_real_idx})")
            if len(ieds_sin_cobertura) > 10:
                print(f"    ... y {len(ieds_sin_cobertura) - 10} más")
    
    def _aplicar_cap(self, individuo: np.ndarray) -> np.ndarray:
        """Aplica restricción dura de máximo de concentradores."""
        # Primero, asegurar que los fijos están activos
        for fijo_idx in self.concentradores_fijos:
            individuo[fijo_idx] = 1
        
        max_conc = self.config.max_concentradores
        if max_conc is None:
            return individuo

        try:
            max_conc_int = int(max_conc)
        except Exception:
            return individuo

        if max_conc_int < 1:
            max_conc_int = 1
        
        # Ajustar máximo considerando los fijos (los fijos no cuentan contra el límite adicional)
        # O sea, si max_conc=5 y hay 3 fijos, permitimos hasta 5 adicionales + 3 fijos = 8 total
        # Alternativamente: max_conc incluye los fijos, entonces máximo total es max_conc
        # Usaré la segunda interpretación: max_conc es el TOTAL incluyendo fijos
        max_total = max(max_conc_int, self.n_fijos)  # Al menos los fijos

        # Recortar si excede el máximo
        while int(np.sum(individuo)) > max_total:
            activos = np.where(individuo == 1)[0].tolist()
            # Filtrar los que NO son fijos (solo esos se pueden remover)
            removibles = [idx for idx in activos if idx not in self.concentradores_fijos]
            
            if not removibles:
                break  # Solo quedan fijos, no podemos remover más

            if self.n_ieds == 0:
                # Remover los excedentes
                n_remover = len(activos) - max_total
                for idx in removibles[:n_remover]:
                    individuo[idx] = 0
                break

            # Calcular scores solo para removibles
            cov = self.matriz_cobertura[activos, :]

            scores = []
            for row_i, conc_idx in enumerate(activos):
                if conc_idx in self.concentradores_fijos:
                    continue  # No evaluar fijos para remoción
                
                cov_i = cov[row_i]

                if len(activos) > 1:
                    others_cov = np.any(np.delete(cov, row_i, axis=0), axis=0)
                else:
                    others_cov = np.zeros_like(cov_i, dtype=bool)

                unique = cov_i & (~others_cov)
                unique_count = int(np.sum(unique))
                coverage_count = int(np.sum(cov_i))

                dists = self.matriz_distancias[conc_idx, :]
                if coverage_count > 0:
                    mean_dist = float(np.mean(dists[cov_i]))
                else:
                    mean_dist = float("inf")

                scores.append((unique_count, coverage_count, -mean_dist, conc_idx))

            if not scores:
                break
            
            scores.sort(key=lambda t: (t[0], t[1], t[2]))
            conc_remove = scores[0][3]
            individuo[conc_remove] = 0

        # Asegurar que los fijos siguen activos
        for fijo_idx in self.concentradores_fijos:
            individuo[fijo_idx] = 1
        
        # Asegurar al menos un concentrador
        if int(np.sum(individuo)) == 0:
            individuo[random.randint(0, self.n_puntos - 1)] = 1

        return individuo

    def _crear_individuo(self) -> np.ndarray:
        """Crea un individuo aleatorio respetando el máximo y los fijos."""
        individuo = np.zeros(self.n_puntos, dtype=int)
        
        # SIEMPRE activar los concentradores fijos
        for fijo_idx in self.concentradores_fijos:
            individuo[fijo_idx] = 1

        max_conc = self.config.max_concentradores
        if max_conc is not None:
            try:
                max_conc_int = int(max_conc)
            except Exception:
                max_conc_int = None
            if max_conc_int is not None and max_conc_int < 1:
                max_conc_int = 1
        else:
            max_conc_int = None

        # Heurística de arranque para concentradores ADICIONALES
        max_arranque = max(2, self.n_puntos // 10)
        if max_conc_int is not None:
            max_adicionales = max(0, max_conc_int - self.n_fijos)
            max_arranque = min(max_arranque, max_adicionales)
        
        if max_arranque > 0:
            n_adicionales = random.randint(0, max_arranque)
            # Seleccionar índices que NO son fijos
            indices_disponibles = [i for i in range(self.n_puntos) if i not in self.concentradores_fijos]
            if indices_disponibles and n_adicionales > 0:
                indices_adicionales = random.sample(indices_disponibles, min(n_adicionales, len(indices_disponibles)))
                individuo[indices_adicionales] = 1

        return self._aplicar_cap(individuo)
    
    def _evaluar_fitness(self, individuo: np.ndarray) -> Tuple[float, Dict]:
        """
        Evalúa fitness con DOS REGÍMENES:
        - Si cobertura < 100%: presión MASIVA por cobertura
        - Si cobertura = 100%: presión MASIVA por minimización
        
        IMPORTANTE: Solo penaliza concentradores ADICIONALES (no los fijos)
        """
        concentradores = np.where(individuo == 1)[0]
        n_concentradores = len(concentradores)
        n_adicionales = n_concentradores - self.n_fijos  # Solo los no-fijos
        
        # Verificar restricción dura
        max_conc = self.config.max_concentradores
        if max_conc is not None and n_concentradores > int(max_conc):
            return -1e9, {'cobertura': 0, 'n_conc': n_concentradores, 'n_adicionales': n_adicionales, 
                          'dist_prom': np.inf, 'n_cubiertos': 0, 'margen_prom': 0}
        
        if n_concentradores == 0:
            return -1e9, {'cobertura': 0, 'n_conc': 0, 'n_adicionales': 0, 
                          'dist_prom': np.inf, 'n_cubiertos': 0, 'margen_prom': 0}
        
        if self.n_ieds == 0:
            return 0, {'cobertura': 1.0, 'n_conc': n_concentradores, 'n_adicionales': n_adicionales,
                       'dist_prom': 0, 'n_cubiertos': 0, 'margen_prom': 0}
        
        ieds_cubiertos = np.zeros(self.n_ieds, dtype=bool)
        distancias_min = np.full(self.n_ieds, np.inf)
        
        # Auto-cubrir
        for j_idx, j in enumerate(self.indices_ieds):
            if j in concentradores:
                ieds_cubiertos[j_idx] = True
                distancias_min[j_idx] = 0
        
        # Cobertura desde concentradores
        for conc_idx in concentradores:
            cobertura_conc = self.matriz_cobertura[conc_idx, :]
            distancias_conc = self.matriz_distancias[conc_idx, :]
            
            ieds_cubiertos |= cobertura_conc
            
            mascara = cobertura_conc & (distancias_conc < distancias_min)
            distancias_min[mascara] = distancias_conc[mascara]
        
        n_cubiertos = np.sum(ieds_cubiertos)
        pct_cobertura = n_cubiertos / self.n_ieds
        
        distancias_validas = distancias_min[(distancias_min < np.inf) & (distancias_min > 0)]
        dist_promedio = np.mean(distancias_validas) if len(distancias_validas) > 0 else 0
        
        # Calcular margen Fresnel promedio
        margenes_validos = []
        for j_idx, j in enumerate(self.indices_ieds):
            if j not in concentradores and ieds_cubiertos[j_idx]:
                for conc_idx in concentradores:
                    if self.matriz_cobertura[conc_idx, j_idx]:
                        if self.matriz_distancias[conc_idx, j_idx] == distancias_min[j_idx]:
                            margenes_validos.append(self.matriz_margenes[conc_idx, j_idx])
                            break
        
        margen_promedio = np.mean(margenes_validos) if margenes_validos else 0
        
        # ============================================================
        # FITNESS CON DOS REGÍMENES
        # IMPORTANTE: Penalizar solo concentradores ADICIONALES
        # ============================================================
        
        if pct_cobertura < 1.0:
            # RÉGIMEN 1: COBERTURA INCOMPLETA - presión masiva por cobertura
            fitness = (
                pct_cobertura * 1000000  # Peso ENORME para cobertura
                - n_adicionales * 5000  # Penalizar solo concentradores ADICIONALES
            )
        else:
            # RÉGIMEN 2: COBERTURA COMPLETA - presión masiva por minimización
            dist_normalizada = dist_promedio / self.config.distancia_max_km if dist_promedio > 0 else 0
            margen_normalizado = (margen_promedio + 20) / 70
            margen_normalizado = max(0, min(1, margen_normalizado))
            
            fitness = (
                500000  # Bonus base por cobertura completa
                - n_adicionales * 50000  # PENALIZACIÓN MASIVA por cada concentrador ADICIONAL
                - dist_normalizada * 1000  # Penalizar distancias largas
                + margen_normalizado * 500  # Premiar buenos márgenes
            )
            
            # Bonus extra por soluciones ultra-compactas (basado en adicionales)
            if n_adicionales == 0:
                fitness += 30000  # Solo con fijos
            elif n_adicionales <= 1:
                fitness += 20000
            elif n_adicionales <= 2:
                fitness += 10000
            elif n_adicionales <= 3:
                fitness += 5000
        
        metricas = {
            'cobertura': pct_cobertura,
            'n_cubiertos': n_cubiertos,
            'n_conc': n_concentradores,
            'n_adicionales': n_adicionales,
            'dist_prom': dist_promedio,
            'margen_prom': margen_promedio
        }
        
        return fitness, metricas
    
    def _seleccion_torneo(self, poblacion: List, fitnesses: List) -> np.ndarray:
        """Selección por torneo."""
        indices = random.sample(range(len(poblacion)), self.config.torneo_size)
        mejor_idx = max(indices, key=lambda i: fitnesses[i])
        return poblacion[mejor_idx].copy()
    
    def _cruce(self, padre1: np.ndarray, padre2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cruce de dos puntos, protegiendo genes fijos."""
        if random.random() > self.config.prob_cruce:
            return self._aplicar_cap(padre1.copy()), self._aplicar_cap(padre2.copy())

        p1, p2 = sorted(random.sample(range(self.n_puntos), 2))

        hijo1 = padre1.copy()
        hijo2 = padre2.copy()

        hijo1[p1:p2] = padre2[p1:p2]
        hijo2[p1:p2] = padre1[p1:p2]

        # Restaurar genes fijos (nunca deben cambiar a 0)
        for fijo_idx in self.concentradores_fijos:
            hijo1[fijo_idx] = 1
            hijo2[fijo_idx] = 1

        return self._aplicar_cap(hijo1), self._aplicar_cap(hijo2)
    
    def _mutar(self, individuo: np.ndarray) -> np.ndarray:
        """Mutación inteligente, protegiendo genes fijos."""
        if random.random() < self.config.prob_mutacion:
            n_mutaciones = random.randint(1, 3)
            
            # Solo mutar índices que NO son fijos
            indices_mutables = [i for i in range(self.n_puntos) if i not in self.concentradores_fijos]
            
            if indices_mutables:
                indices_mutar = random.sample(indices_mutables, min(n_mutaciones, len(indices_mutables)))
                
                for i in indices_mutar:
                    individuo[i] = 1 - individuo[i]
        
        # Asegurar que los fijos están activos
        for fijo_idx in self.concentradores_fijos:
            individuo[fijo_idx] = 1
        
        # Asegurar al menos un concentrador
        if np.sum(individuo) == 0:
            individuo[random.randint(0, self.n_puntos - 1)] = 1

        return self._aplicar_cap(individuo)
    
    def _limpiar_concentradores(self, concentradores_idx: np.ndarray) -> np.ndarray:
        """
        PASO 1: Elimina concentradores que NO cubren a NADIE.
        Conservador: solo elimina los obviamente inútiles.
        NUNCA elimina concentradores fijos.
        """
        if self.n_ieds == 0:
            return concentradores_idx
        
        concentradores_limpios = []
        
        for conc_idx in concentradores_idx:
            # NUNCA eliminar fijos
            if conc_idx in self.concentradores_fijos:
                concentradores_limpios.append(conc_idx)
                continue
            
            # Verificar si cubre a alguien
            cobertura = self.matriz_cobertura[conc_idx, :]
            
            # ¿Es un IED auto-cubierto?
            es_ied_autocubierto = False
            for j_idx, j in enumerate(self.indices_ieds):
                if j == conc_idx:
                    es_ied_autocubierto = True
                    break
            
            # ¿Cubre a alguien más?
            cubre_otros = np.sum(cobertura) > 0
            
            if es_ied_autocubierto or cubre_otros:
                concentradores_limpios.append(conc_idx)
        
        return np.array(concentradores_limpios)
    
    def _depurar_solucion(self, concentradores_idx: np.ndarray) -> np.ndarray:
        """
        PASO 2: Depuración greedy uno-por-uno.
        Intenta remover cada concentrador y verifica si mantiene 100% cobertura.
        NUNCA remueve concentradores fijos.
        """
        if self.n_ieds == 0:
            return concentradores_idx
        
        concentradores = list(concentradores_idx)
        mejorado = True
        
        print("\n" + "="*50)
        print("DEPURACIÓN GREEDY")
        print("="*50)
        if self.n_fijos > 0:
            print(f"(Protegiendo {self.n_fijos} concentradores fijos)")
        
        while mejorado and len(concentradores) > 1:
            mejorado = False
            
            # Calcular valor único de cada concentrador
            scores = []
            for conc_idx in concentradores:
                # NUNCA evaluar fijos para remoción
                if conc_idx in self.concentradores_fijos:
                    continue
                
                otros = [c for c in concentradores if c != conc_idx]
                
                # Cobertura actual del concentrador
                cov_actual = self.matriz_cobertura[conc_idx, :]
                
                # Cobertura de los otros
                if otros:
                    cov_otros = np.any(self.matriz_cobertura[otros, :], axis=0)
                else:
                    cov_otros = np.zeros(self.n_ieds, dtype=bool)
                
                # IEDs únicos (solo cubiertos por este)
                unicos = cov_actual & (~cov_otros)
                n_unicos = np.sum(unicos)
                
                # Distancia promedio
                if np.sum(cov_actual) > 0:
                    dist_prom = np.mean(self.matriz_distancias[conc_idx, cov_actual])
                else:
                    dist_prom = np.inf
                
                scores.append((n_unicos, dist_prom, conc_idx))
            
            if not scores:
                break  # Solo quedan fijos
            
            # Ordenar: primero los que tienen menos únicos, luego por distancia
            scores.sort(key=lambda x: (x[0], -x[1]))
            
            # Intentar remover el menos crítico
            for n_unicos, dist_prom, conc_idx in scores:
                if n_unicos > 0:
                    # Tiene IEDs únicos, no se puede remover
                    continue
                
                # Intentar removerlo
                concentradores_temp = [c for c in concentradores if c != conc_idx]
                
                # Verificar cobertura sin él
                if concentradores_temp:
                    cov_temp = np.any(self.matriz_cobertura[concentradores_temp, :], axis=0)
                    
                    # Auto-cobertura de IEDs que son concentradores
                    for j_idx, j in enumerate(self.indices_ieds):
                        if j in concentradores_temp:
                            cov_temp[j_idx] = True
                    
                    if np.all(cov_temp):
                        # Podemos removerlo
                        concentradores = concentradores_temp
                        print(f"  → Removido: {self.placas[conc_idx]} (redundante)")
                        mejorado = True
                        break
        
        resultado = np.array(concentradores)
        n_adicionales = len([c for c in concentradores if c not in self.concentradores_fijos])
        print(f"\nResultado depuración: {len(resultado)} concentradores ({self.n_fijos} fijos + {n_adicionales} adicionales)")
        
        return resultado
    
    def _validacion_final(self, concentradores_idx: np.ndarray, interactivo: bool = True) -> Dict:
        """
        PASO 3: Validación final exhaustiva.
        Intenta encontrar subconjuntos mínimos que mantengan cobertura.
        SIEMPRE mantiene los concentradores fijos.
        """
        print("\n" + "="*50)
        print("VALIDACIÓN EXHAUSTIVA FINAL")
        print("="*50)
        
        concentradores = list(concentradores_idx)
        n_total = len(concentradores)
        
        # Separar fijos de adicionales
        fijos = [c for c in concentradores if c in self.concentradores_fijos]
        adicionales = [c for c in concentradores if c not in self.concentradores_fijos]
        
        print(f"Concentradores actuales: {n_total} ({len(fijos)} fijos + {len(adicionales)} adicionales)")
        
        if len(adicionales) == 0:
            print("No hay concentradores adicionales para optimizar.")
            return {'concentradores': np.array(concentradores), 'mejorado': False}
        
        def verificar_cobertura(concs):
            """Verifica si un conjunto de concentradores da 100% cobertura."""
            if not concs:
                return False
            
            cov = np.any(self.matriz_cobertura[concs, :], axis=0)
            
            # Auto-cobertura
            for j_idx, j in enumerate(self.indices_ieds):
                if j in concs:
                    cov[j_idx] = True
            
            return np.all(cov)
        
        # Verificar cobertura actual
        if not verificar_cobertura(concentradores):
            print("⚠ La solución actual NO tiene 100% cobertura")
            return {'concentradores': np.array(concentradores), 'mejorado': False}
        
        # Intentar reducir adicionales
        mejor_adicionales = adicionales.copy()
        
        # Probar con menos adicionales (de n-1 hacia abajo)
        for n_adic in range(len(adicionales) - 1, -1, -1):
            if n_adic == 0:
                # Solo fijos
                if verificar_cobertura(fijos):
                    print(f"★ ¡Solo los {len(fijos)} fijos dan 100% cobertura!")
                    mejor_adicionales = []
                    break
            else:
                # Probar combinaciones de n_adic adicionales
                encontrado = False
                combos = list(combinations(adicionales, n_adic))
                
                if len(combos) > 1000:
                    print(f"  Probando {n_adic} adicionales ({len(combos)} combinaciones, limitando a 1000)...")
                    combos = random.sample(combos, 1000)
                else:
                    print(f"  Probando {n_adic} adicionales ({len(combos)} combinaciones)...")
                
                for combo in combos:
                    test_concs = fijos + list(combo)
                    if verificar_cobertura(test_concs):
                        mejor_adicionales = list(combo)
                        encontrado = True
                        break
                
                if not encontrado:
                    # No se puede reducir más
                    break
        
        mejor_total = fijos + mejor_adicionales
        mejorado = len(mejor_total) < n_total
        
        if mejorado:
            print(f"\n★ MEJORA ENCONTRADA: {n_total} → {len(mejor_total)} concentradores")
            print(f"  ({len(fijos)} fijos + {len(mejor_adicionales)} adicionales)")
        else:
            print(f"\nNo se encontró mejora. Solución óptima: {n_total} concentradores")
        
        return {
            'concentradores': np.array(mejor_total),
            'mejorado': mejorado
        }
    
    def optimizar(self) -> Dict:
        """Ejecuta el algoritmo genético completo."""
        if self.n_ieds == 0:
            print("\n⚠ No hay IEDs para optimizar cobertura.")
            # Si hay fijos, retornarlos como solución
            if self.n_fijos > 0:
                return {
                    'exito': True,
                    'cobertura': 1.0,
                    'n_ieds': 0,
                    'n_cubiertos': 0,
                    'n_concentradores': self.n_fijos,
                    'n_adicionales': 0,
                    'distancia_promedio_km': 0,
                    'concentradores_indices': list(self.concentradores_fijos),
                    'concentradores_placas': [self.placas[i] for i in self.concentradores_fijos],
                    'concentradores_tipos': [self.tipos[i] for i in self.concentradores_fijos],
                    'cromosoma': np.zeros(self.n_puntos, dtype=int)
                }
            return {
                'exito': False,
                'cobertura': 0,
                'n_ieds': 0,
                'n_cubiertos': 0,
                'n_concentradores': 0,
                'n_adicionales': 0,
                'distancia_promedio_km': 0,
                'concentradores_indices': [],
                'concentradores_placas': [],
                'concentradores_tipos': [],
                'cromosoma': np.zeros(self.n_puntos, dtype=int)
            }
        
        poblacion = [self._crear_individuo() for _ in range(self.config.poblacion_size)]
        
        mejor_global = None
        mejor_fitness_global = -np.inf
        mejor_metricas_global = None
        generaciones_sin_mejora = 0
        
        print("\nEvolución:")
        print("-" * 70)
        if self.n_fijos > 0:
            print(f"(Concentradores fijos: {self.n_fijos} | Buscando adicionales óptimos)")
            print("-" * 70)
        
        for gen in range(self.config.generaciones):
            evaluaciones = [self._evaluar_fitness(ind) for ind in poblacion]
            fitnesses = [e[0] for e in evaluaciones]
            metricas = [e[1] for e in evaluaciones]
            
            idx_mejor = np.argmax(fitnesses)
            mejor_gen = poblacion[idx_mejor]
            mejor_fitness_gen = fitnesses[idx_mejor]
            mejor_metricas_gen = metricas[idx_mejor]
            
            if mejor_fitness_gen > mejor_fitness_global:
                mejor_global = mejor_gen.copy()
                mejor_fitness_global = mejor_fitness_gen
                mejor_metricas_global = mejor_metricas_gen
                generaciones_sin_mejora = 0
                
                n_adic = mejor_metricas_gen.get('n_adicionales', mejor_metricas_gen['n_conc'] - self.n_fijos)
                print(f"Gen {gen:3d}: Cob={mejor_metricas_gen['cobertura']*100:.0f}% | "
                      f"Conc={mejor_metricas_gen['n_conc']:2d} ({self.n_fijos}F+{n_adic}A) | "
                      f"Dist={mejor_metricas_gen['dist_prom']:.1f}km | "
                      f"Margen={mejor_metricas_gen['margen_prom']:.1f}m ★")
            else:
                generaciones_sin_mejora += 1
                if gen % 20 == 0:
                    n_adic = mejor_metricas_gen.get('n_adicionales', mejor_metricas_gen['n_conc'] - self.n_fijos)
                    print(f"Gen {gen:3d}: Cob={mejor_metricas_gen['cobertura']*100:.0f}% | "
                          f"Conc={mejor_metricas_gen['n_conc']:2d} ({self.n_fijos}F+{n_adic}A) | "
                          f"Dist={mejor_metricas_gen['dist_prom']:.1f}km | "
                          f"Margen={mejor_metricas_gen['margen_prom']:.1f}m")
            
            if generaciones_sin_mejora >= self.config.max_sin_mejora:
                print(f"\n→ Parada temprana en generación {gen} (sin mejora)")
                break
            
            indices_elite = np.argsort(fitnesses)[-self.config.elitismo:]
            nueva_poblacion = [poblacion[i].copy() for i in indices_elite]
            
            while len(nueva_poblacion) < self.config.poblacion_size:
                padre1 = self._seleccion_torneo(poblacion, fitnesses)
                padre2 = self._seleccion_torneo(poblacion, fitnesses)
                
                hijo1, hijo2 = self._cruce(padre1, padre2)
                hijo1 = self._mutar(hijo1)
                hijo2 = self._mutar(hijo2)
                
                nueva_poblacion.extend([hijo1, hijo2])
            
            poblacion = nueva_poblacion[:self.config.poblacion_size]
        
        # ========================================================
        # POST-PROCESAMIENTO: Limpiar, Depurar y Validar
        # ========================================================
        
        concentradores_idx = np.where(mejor_global == 1)[0]
        
        print("\n" + "="*60)
        print("POST-PROCESAMIENTO")
        print("="*60)
        n_adic_bruto = len([c for c in concentradores_idx if c not in self.concentradores_fijos])
        print(f"Solución GA bruta: {len(concentradores_idx)} concentradores ({self.n_fijos} fijos + {n_adic_bruto} adicionales)")
        
        # PASO 1: Limpiar
        concentradores_limpios = self._limpiar_concentradores(concentradores_idx)
        if len(concentradores_limpios) < len(concentradores_idx):
            n_adic_limpio = len([c for c in concentradores_limpios if c not in self.concentradores_fijos])
            print(f"Después de limpieza: {len(concentradores_limpios)} concentradores ({self.n_fijos} fijos + {n_adic_limpio} adicionales)")
        
        # PASO 2: Depurar
        concentradores_depurados = self._depurar_solucion(concentradores_limpios)
        if len(concentradores_depurados) < len(concentradores_limpios):
            n_adic_dep = len([c for c in concentradores_depurados if c not in self.concentradores_fijos])
            print(f"Después de depuración: {len(concentradores_depurados)} concentradores ({self.n_fijos} fijos + {n_adic_dep} adicionales)")
        
        # PASO 3: Validación final exhaustiva
        print("\n" + "="*60)
        respuesta = input("¿Ejecutar validación exhaustiva final? (puede tardar) [s/n]: ").strip().lower()
        
        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            resultado_validacion = self._validacion_final(concentradores_depurados, interactivo=True)
            concentradores_finales = resultado_validacion['concentradores']
            
            if resultado_validacion['mejorado']:
                respuesta_aceptar = input(f"\n¿Aceptar solución mejorada? [s/n]: ").strip().lower()
                if respuesta_aceptar not in ['s', 'si', 'sí', 'y', 'yes']:
                    print("Manteniendo solución pre-validación")
                    concentradores_finales = concentradores_depurados
        else:
            concentradores_finales = concentradores_depurados
        
        # ========================================================
        # CONSTRUIR RESULTADO FINAL
        # ========================================================
        
        # Evaluar solución final
        individuo_final = np.zeros(self.n_puntos, dtype=int)
        individuo_final[concentradores_finales] = 1
        fitness_final, metricas_final = self._evaluar_fitness(individuo_final)
        
        n_adicionales_final = len([c for c in concentradores_finales if c not in self.concentradores_fijos])
        
        resultado = {
            'exito': metricas_final['cobertura'] >= 1.0,
            'cobertura': metricas_final['cobertura'],
            'n_ieds': self.n_ieds,
            'n_cubiertos': metricas_final['n_cubiertos'],
            'n_concentradores': len(concentradores_finales),
            'n_fijos': self.n_fijos,
            'n_adicionales': n_adicionales_final,
            'distancia_promedio_km': metricas_final['dist_prom'],
            'margen_promedio_m': metricas_final['margen_prom'],
            'concentradores_indices': concentradores_finales.tolist(),
            'concentradores_placas': [self.placas[i] for i in concentradores_finales],
            'concentradores_tipos': [self.tipos[i] for i in concentradores_finales],
            'concentradores_fijos_indices': list(self.concentradores_fijos),
            'cromosoma': individuo_final
        }
        
        return resultado
    
    def detalle_cobertura(self, resultado: Dict) -> pd.DataFrame:
        """Genera DataFrame detallado de la cobertura."""
        if self.n_ieds == 0:
            return pd.DataFrame()
        
        concentradores_idx = resultado['concentradores_indices']
        
        registros = []
        
        # Analizar cada IED
        for j_idx, j in enumerate(self.indices_ieds):
            placa = self.placas[j]
            tipo = self.tipos[j]
            es_fijo = j in self.concentradores_fijos
            
            # ¿Es un IED-CONCENTRADOR?
            if j in concentradores_idx:
                if es_fijo:
                    rol = 'IED-CONCENTRADOR-FIJO'
                else:
                    rol = 'IED-CONCENTRADOR'
                conc_asignado = placa
                dist = 0
                margen = 0
            else:
                rol = 'IED'
                
                # Buscar mejor concentrador
                mejor_dist = np.inf
                mejor_conc_idx = None
                mejor_margen = -np.inf
                
                for conc_idx in concentradores_idx:
                    if self.matriz_cobertura[conc_idx, j_idx]:
                        dist = self.matriz_distancias[conc_idx, j_idx]
                        if dist < mejor_dist:
                            mejor_dist = dist
                            mejor_conc_idx = conc_idx
                            mejor_margen = self.matriz_margenes[conc_idx, j_idx]
                
                if mejor_conc_idx is not None:
                    conc_asignado = self.placas[mejor_conc_idx]
                    dist = mejor_dist
                    margen = mejor_margen
                else:
                    conc_asignado = 'SIN COBERTURA'
                    dist = np.nan
                    margen = np.nan
            
            registros.append({
                'Indice': j,
                'IED': placa,
                'Tipo_IED': tipo,
                'Rol': rol,
                'Concentrador_asignado': conc_asignado,
                'Distancia_km': dist,
                'Margen_F60_m': margen
            })
        
        # Agregar concentradores puros (no son IEDs)
        for conc_idx in concentradores_idx:
            if conc_idx not in self.indices_ieds:
                placa = self.placas[conc_idx]
                tipo = self.tipos[conc_idx]
                es_fijo = conc_idx in self.concentradores_fijos
                
                registros.append({
                    'Indice': conc_idx,
                    'IED': placa,
                    'Tipo_IED': tipo,
                    'Rol': 'CONCENTRADOR-FIJO' if es_fijo else 'CONCENTRADOR',
                    'Concentrador_asignado': placa,
                    'Distancia_km': 0,
                    'Margen_F60_m': 0
                })
        
        return pd.DataFrame(registros)
