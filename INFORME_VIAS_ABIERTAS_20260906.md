# Las vías abiertas del informe consolidado, puestas a prueba

**Documento sobre:** `INFORME_CONSOLIDADO_TESIS_20260901`

| | |
|---|---|
| **Fecha** | 6 de septiembre de 2026 |
| **Conjuntos de datos** | `cptac_brca`, `cptac_gbm`, `bc_therapy`, `cervical_subtype` |
| **Particiones** | 50 por experimento (5 en cervical; 20 en la partición nueva) |
| **Vías probadas** | 10 · **1 funcionó** |
| **Trabajos de cálculo** | 46 |
| **Correcciones al informe** | 5 |

> El informe de 1 de septiembre cerraba con diez líneas de investigación pendientes.
> Se han ejecutado todas. **Una funcionó.** Las demás quedan cerradas con su margen de
> error medido — y varias afirmaciones del informe necesitan revisión.

> **Nota de lectura.** Los términos con enlace llevan su definición en el
> [anexo A](#anexo-a--glosario) y muestran un resumen al pasar el ratón por encima.
> Las pruebas estadísticas están explicadas en el [anexo B](#anexo-b--las-pruebas-estadísticas).

---

## 1 · Cómo funciona el sistema

El objetivo es mirar la imagen de una biopsia y predecir si el tumor tiene mutado el gen
TP53. La imagen es enorme —una <a href="#g-wsi" title="Whole-Slide Image: la digitalización completa de un portaobjetos. Puede ocupar varios gigapíxeles.">WSI</a>
tiene miles de millones de píxeles—, así que no se procesa entera: se trocea en miles de
cuadraditos llamados <a href="#g-parche" title="Cuadradito de 224x224 píxeles recortado de la imagen completa. Una biopsia da entre 3.700 y 6.600.">parches</a>.

Cada parche pasa por un <a href="#g-fundacional" title="Red neuronal grande, ya entrenada por otros con millones de imágenes, que convierte un parche en una lista de números. En este proyecto hay ocho.">modelo fundacional</a>,
que lo convierte en una lista de números o <a href="#g-embedding" title="La lista de números con la que un modelo describe un parche. Entre 768 y 2.560 números según el modelo.">embedding</a>.
Después hay que resumir los miles de parches en una sola predicción, y de eso se encarga el
<a href="#g-abmil" title="Attention-Based Multiple Instance Learning. Aprende a puntuar cada parche según su importancia y hace una media ponderada.">ABMIL</a>,
que aprende a dar más peso a los parches relevantes mediante un mecanismo de
<a href="#g-atencion" title="Puntuación que el modelo asigna a cada parche. Suman 1 entre todos.">atención</a>.

Como ningún modelo fundacional es el mejor en todo, se combinan varios. **La pregunta de la
que trata casi todo este informe es *en qué momento* conviene combinarlos.**

### Los tres momentos en que se pueden combinar los modelos

```
                                    A                B                C
                                    │                │                │
                              nivel de parche   nivel de atención  nivel de prob.
                              fusión temprana      top-k           fusión tardía
                                  FALLA           FUNCIONA          la actual
                                    │                │                │
              ┌── modelo fund. 1 ─→ embeddings ─→ ABMIL ─→ prob. 1 ──┐
              │                     │                │               │
 biopsia ─→ parches ── modelo 2 ─→ embeddings ─→ ABMIL ─→ prob. 2 ──┼─→ meta-      ─→ TP53
  (1 WSI)   (~5.000)   │                                            │  clasificador   sí/no
              └── modelo fund. 3 ─→ embeddings ─→ ABMIL ─→ prob. 3 ──┘
```

El sistema actual combina en **C**: cada modelo trabaja por su cuenta hasta el final y solo
se juntan sus tres probabilidades. Combinar antes, en **A**, pegando los embeddings de cada
parche para alimentar un único ABMIL, se probó y no funciona. Intervenir en **B**, sobre las
puntuaciones de atención, es lo único que ha dado resultado.

### Qué hace exactamente el hallazgo positivo

El ABMIL reparte el 100 % de la atención entre miles de parches, de modo que a la mayoría le
toca una miga. La intervención que funciona consiste en **quedarse solo con los 10 parches
más atendidos y descartar el resto** antes de construir el resumen de la biopsia.

```
   ATENCIÓN COMPLETA                         SOLO LOS 10 PRIMEROS
   todos los parches aportan algo            la cola se descarta

   █                                         █
   █ █                                       █ █
   █ █ █                                     █ █ █
   █ █ █ █                                   █ █ █ █
   █ █ █ █ █ ▄                               █ █ █ █ █ ▄
   █ █ █ █ █ █ ▄ ▄ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁       █ █ █ █ █ █ ▄ ▄ ▁ ▁ ╳╳╳╳╳╳╳╳╳╳╳╳
   └────────┬────────────────────────┘       └──────┬──────────┘ └────┬─────┘
        10 primeros    la cola larga:            conservados       descartada
                    miles de parches con
                    muy poco peso cada uno
```

**Truncar la cola es lo que produce la mejora.** No es cuestión de que la atención esté más
o menos concentrada: a igual grado de concentración, cortar en seco los parches irrelevantes
gana +0,0244 de <a href="#g-auc" title="Área bajo la curva ROC: mide si el modelo ordena bien los casos. 0,5 es azar y 1,0 es perfecto.">AUC</a>
frente a limitarse a afinar los pesos. Aplicado al predecir mejora; aplicado durante el
entrenamiento, perjudica.

### El procedimiento, paso a paso

Conviene subrayar que **no se reentrena nada**. Todo ocurre en el momento de predecir, sobre
modelos ya entrenados, lo que hace la intervención muy barata de aplicar y de revertir.

1. **Se cargan los ABMIL ya entrenados y no se tocan.** Sus pesos quedan exactamente como
   estaban.
2. **Se recalculan las puntuaciones de atención** de cada parche, en su forma cruda, antes de
   normalizarlas.
3. **Se recorta la cola.** Se conservan los *k* parches de puntuación más alta y se reparte
   entre ellos el peso que tenían, *respetando la proporción original*. Los demás pasan a
   pesar cero.

   > **Detalle que importa:** **no** se recalcula la atención desde cero sobre los *k*
   > supervivientes. Se conserva el reparto relativo que el modelo había aprendido y solo se
   > anula la cola. Recalcularla mezclaría dos manipulaciones —recortar y reescalar— y no se
   > sabría cuál produce el efecto.

4. **Se construye el resumen de la biopsia** con esos pesos: una media ponderada de los
   parches supervivientes, que da un vector de 512 números.
5. **Ese resumen alimenta al <a href="#g-meta" title="El modelo pequeño que recibe la salida de los modelos base y produce la predicción final.">metaclasificador</a>**,
   igual que en el procedimiento normal.

### Cómo se reparten los datos

Todo se repite sobre 50 <a href="#g-fold" title="Una división concreta de los datos en entrenamiento, validación y test. Se repite 50 veces.">particiones</a>
distintas. En cada una, las 112 biopsias de `cptac_brca` se reparten así, y **el recorte se
aplica por igual en las tres partes**: si solo se aplicara en test, se estaría evaluando al
metaclasificador con una entrada distinta de aquella con la que se entrenó.

```
  UNA PARTICIÓN · 112 BIOPSIAS · 103 PACIENTES        (se repite 50 veces)

  ├──────────────── entrenamiento ─────────────────┼── validación ──┼───── test ─────┤
  │                   75 biopsias                  │       15       │       22       │
  └────────────────────────────────────────────────┴────────────────┴────────────────┘
              │                                            │                 │
     entrena el metaclasificador          elige el umbral y el valor de k    │
                                                                    única medida
                                                                   que se reporta

  El recorte top-k se aplica en las TRES partes · ningún paciente cruza de una a otra
```

Las cifras corresponden a `cptac_brca`; en `cptac_gbm`, donde se midió el hallazgo, son
**164 / 31 / 48** sobre 243 biopsias.

| Parte | Biopsias | Para qué sirve |
|---|---:|---|
| Entrenamiento | 75 | Entrena el metaclasificador |
| Validación | 15 | Elige el umbral de decisión y el valor de *k* |
| Test | 22 | Única superficie de evaluación que se reporta |

Tres reglas gobiernan ese reparto:

- **Ningún paciente cruza de una parte a otra.** Todas las biopsias de un mismo paciente van
  juntas, para que el modelo no pueda acertar reconociendo al paciente en vez de al tumor.
- **El umbral de decisión y el valor de *k* se eligen en validación, nunca en test.**
  Validación ya se usa para la parada temprana, así que no añade ninguna fuga nueva; elegirlos
  en test invalidaría la única medida honesta que queda.
- **Los modelos base se entrenaron con estas mismas particiones**, de modo que el parche que
  recibe más atención en una biopsia de test nunca se vio durante el entrenamiento.

### Dos comprobaciones que el procedimiento debe pasar

Están automatizadas y el programa **se detiene** si alguna falla, porque un recálculo de
atención mal hecho produciría una curva perfectamente plausible y completamente falsa:

- **Sin recortar nada**, el resultado debe coincidir *exactamente* con la atención original
  del modelo. Si no coincide, el recálculo está mal.
- **Con la atención aplanada del todo**, debe coincidir *exactamente* con el control negativo
  ya publicado en la ficha P4 del informe consolidado. Eso ancla la medida nueva a una cifra
  que ya estaba validada.

Y sobre la posibilidad de fuga: **la atención se calcula solo a partir de la imagen**, sin ver
en ningún momento la etiqueta. Seleccionar parches por atención no puede, por construcción,
filtrar la respuesta.

---

## 2 · Lo único que funcionó

**Vía §4.4 del informe consolidado**, la que allí se calificaba de «contraste decisivo
pendiente». Solo inferencia: no se reentrenó ningún modelo.

| | |
|---|---|
| **Mejora de AUC** | **+0,0388** |
| **Conjunto** | `cptac_gbm` |
| **<a href="#s-holm" title="Probabilidad corregida por hacer varias comparaciones a la vez. Por debajo de 0,05 el hallazgo es firme.">p-Holm</a>** | 0,0046 |
| **<a href="#s-mde" title="Mínimo efecto detectable: la mejora más pequeña que el experimento podría distinguir del azar.">MDE</a>** | 0,0257 |
| **<a href="#s-gep" title="Ganados, empatados y perdidos, sobre las 50 particiones.">G/E/P</a>** | 35 / 1 / 14 |

**Quedarse solo con los 10 parches más atendidos** al construir el resumen de la biopsia
mejora la predicción, y la mejora sobrevive a la corrección estadística aplicada sobre *las
36 comparaciones de toda la campaña*, no solo sobre las suyas.

**Corrige además una idea previa del informe.** Se suponía que cuanto más concentrada
estuviera la atención, mejor. No es así: hay un punto óptimo intermedio. Aplanarla del todo
perjudica, y concentrarla en exceso también.

**Lo que importa es truncar, no concentrar.** Comparando dos intervenciones que dejan la
atención igual de concentrada, la que corta en seco la cola gana +0,0244. El beneficio viene
de eliminar los parches irrelevantes, no de reajustar sus pesos.

**Lo que aún no se puede afirmar** es que sirva como criterio para decidir de antemano en qué
conjunto aplicarlo: con solo 4 conjuntos la relación es coherente en 3, pero no alcanza
significación (ρ = −0,40, p = 0,60).

> **Un contraste que conviene no perder.** Hacer lo mismo *durante el entrenamiento* en vez
> de al predecir hace daño: el AUC baja de 0,8026 a 0,7695. Seleccionar parches al predecir
> ayuda; seleccionarlos al aprender perjudica, porque el modelo se queda sin ejemplos con los
> que aprender a distinguir. Son mecanismos distintos, y la yuxtaposición es en sí un resultado.

---

## 3 · Qué habría que revisar del informe consolidado

Cinco afirmaciones que la campaña no sostiene. La primera afecta a un resultado catalogado
como positivo.

### 3.1 · P5 — combinar nunca superó al mejor modelo suelto

El informe presenta como logro que el comité de tres modelos gana al mejor modelo por su
cuenta (+0,0084, «27 de 50 folds»). La réplica en tres conjuntos **reproduce esa cifra
exactamente** —el mismo +0,0084 y las mismas 27 victorias— y precisamente por eso la
desmonta: ese valor **ya era menor que el <a href="#s-mde" title="Mínimo efecto detectable.">MDE</a> del propio
experimento** (0,0118) cuando se publicó.

| Conjunto | Δ AUC | p | p-Holm | MDE |
|---|---:|---:|---:|---:|
| `cptac_brca` | +0,0084 | 0,128 | 1,000 | 0,0118 |
| `cptac_gbm` | −0,0014 | 0,689 | 1,000 | 0,0337 |
| `cervical_subtype` | +0,0061 | 0,188 | 1,000 | 0,0074 |

El «27 de 50» omite además que hubo **6 empates y 17 derrotas**: entre los 44 casos
decididos, 27 a 17 es prácticamente lo que daría una moneda.

**Y hay una explicación mecánica.** El comité parece aportar si se usa el corte por defecto
en 0,5 para decidir sí/no (acierto +0,0370, p = 0,0062). Pero si ese corte se ajusta sobre
validación, como debe hacerse, la ventaja se evapora: **+0,0008**. Lo que el comité añade no
es ordenar mejor los casos sino compensar un corte mal puesto — y eso se consigue
recalibrando un único modelo. Es el mismo diagnóstico que el informe da en las fichas N1–N2
para el hundimiento de los MLP.

### 3.2 · §4.1 — la vía de más prioridad estaba mal planteada

El §4.1 pide «entrenar los modelos fundacionales que faltan: UNI2, H-Optimus-0 y
Prov-GigaPath», y lo sitúa como la palanca más prometedora del proyecto. Los tres puntos son
inexactos:

- **UNI2 ya estaba entrenado.** Es el modelo que en el informe figura como `uni_v2`
  (1.536 dimensiones, que es lo que distingue UNI2 de UNI v1).
- **H-Optimus-0 y Prov-GigaPath no son «modelos sin entrenar», sino modelos sin datos.**
  No existen sus embeddings para `cptac_brca` ni `bc_therapy`. El cuello de botella es la
  extracción de características, no el entrenamiento — un trabajo de otro orden de magnitud.
- **Donde sí se pueden entrenar, ya están medidos y no aportan.** En `cptac_gbm`, añadir
  Prov-GigaPath da +0,0088 con un MDE de 0,0095, y H-Optimus-0 da −0,0057.

La frase «cada uno es una oportunidad del tamaño del cambio `virchow_v1 → conch_v1_5`» queda
por tanto **falsada en los dos conjuntos donde era comprobable**.

### 3.3 · §4.8 — la atención con compuerta sí estaba probada

El informe lista entre las líneas menores pendientes: «*gated attention*: nunca probada; es
un cambio de una línea en la configuración del ABMIL». Es al revés: el código usa atención
con compuerta **de forma incondicional**, y no existe ninguna opción para desactivarla.

Lo que faltaba era la ablación contraria — quitarla. Hecha: el AUC no cambia (0,8025 frente
a 0,8026), aunque <a href="#g-kappa" title="Medida de acuerdo que descuenta los aciertos esperables por azar.">kappa</a>
sí baja de 0,489 a 0,445. La compuerta no aporta capacidad de ordenación.

### 3.4 · §2.3 — la tabla de modelos base es de un solo conjunto

La tabla del §2.3 se presenta como el ranking de los ocho modelos fundacionales, pero está
medida únicamente sobre `cptac_brca`. En `cptac_gbm` el orden es distinto: `ctranspath` cae
del sexto puesto al último (0,7571 → 0,6713) y `uni_v2` sube del quinto al tercero. Solo
`conch_v1_5` es primero en ambos.

Conviene **etiquetar la tabla con su conjunto de datos** y no tratarla como una propiedad
general de los modelos. Lo mismo vale para las correlaciones de redundancia del §2.4.

### 3.5 · §3.4 — más datos no redujeron la brecha entre entrenamiento y test

El informe atribuye la brecha train→test a que el cuello de botella es muestral: medio millón
de parámetros entrenados con 75 biopsias. Pero en `cptac_gbm`, **con 164 biopsias de
entrenamiento en lugar de 75, la brecha es mayor**, no menor: de +0,21 a +0,30, frente a
+0,09 a +0,20 en `cptac_brca`.

No invalida el diagnóstico —el tamaño muestral sigue siendo pequeño en términos absolutos—,
pero sí indica que la relación no es tan directa como el §3.4.1 sugiere.

---

## 4 · Todos los resultados

> **Cómo leer la tabla.** El **MDE** es el margen de incertidumbre del experimento:
> cualquier efecto menor que él es indistinguible del azar. **Solo cuando |Δ| supera
> claramente al MDE hay hallazgo.** Casi ninguno lo hace, y ese es el resumen de la campaña.

| Vía | Experimento | Δ AUC | MDE | Veredicto |
|---|---|---:|---:|---|
| §4.4 | Quedarse con los 10 parches más atendidos (`cptac_gbm`) | **+0,0388** | 0,0257 | ✅ **Funciona** |
| §4.4 | Quedarse con los 100 más atendidos (`cptac_brca`) | +0,0186 | 0,0223 | Nulo acotado |
| §4.1 | Añadir Prov-GigaPath al trío (`cptac_gbm`) | +0,0088 | 0,0095 | Nulo acotado |
| P5 | Comité de 3 vs. mejor modelo suelto (`cptac_brca`) | +0,0084 | 0,0118 | Nulo acotado |
| §4.5 | Ampliar la compresión interna a 2048 (`virchow_v1`) | +0,0034 | 0,0261 | Nulo acotado |
| §4.7 | Elegir modelos según el órgano (mejor de 9 config.) | +0,0021 | 0,0062 | Nulo acotado |
| §4.3 | Embeddings en vez de probabilidades (cervical, 20 part.) | −0,0013 | 0,0063 | Nulo acotado |
| — | Optimizar los pesos de la mezcla (Nelder-Mead) | −0,0038 | ≈0,0100 | Nulo acotado |
| §4.2 | Combinar al nivel de parche (`cptac_gbm`) | −0,0086 | 0,0391 | Nulo acotado |
| §4.6 | Corregir la fuga de información con regresión logística | −0,0156 | 0,0246 | Nulo acotado |
| §4.6 | Usar LightGBM para combinar | −0,0281 | 0,0251 | Nulo acotado |
| §4.8 | Quedarse con 64 parches **durante el entrenamiento** | −0,0331 | ≈0,0250 | ❌ Perjudica |
| §4.5 | Quitar del todo la compresión interna (`virchow_v1`) | −0,0394 | 0,0405 | ❌ Perjudica |
| §4.6 | Red neuronal + corrección de fuga | −0,1692 | 0,0913 | ❌ Perjudica |

Detalle de la fusión temprana (§4.2), medida en los tres conjuntos donde el trío comparte
cuadrícula de parches:

| Conjunto | Δ AUC | IC 95 % | G/E/P | p-Holm | MDE |
|---|---:|---|---|---:|---:|
| `cptac_gbm` | −0,0086 | [−0,0368, +0,0175] | 28/0/22 | 1,00 | 0,0391 |
| `bc_therapy` | −0,0024 | [−0,0225, +0,0173] | 26/0/24 | 1,00 | 0,0288 |
| `cervical_subtype` | −0,0021 | [−0,0096, +0,0042] | 3/0/2 | 1,00 | 0,0111 |

---

## 5 · Vías que quedan cerradas

Cerradas con su margen medido, no por falta de resultados. Un nulo acotado es una
conclusión; un nulo sin MDE, no.

| Vía | Cierre |
|---|---|
| **Fusión temprana** · §4.2 | Nula en los tres conjuntos donde el trío comparte cuadrícula. El −0,0056 previo no era un artefacto del conjunto pequeño: se repite con 164 y 479 biopsias de entrenamiento, y en una tarea de 4 clases. |
| **Enrutado por órgano** · §4.7 | Nueve configuraciones y cuatro órganos, todas sin efecto (p-Holm = 1,000). Ni siquiera un selector perfecto, que supiera de antemano la respuesta, pasaría de +0,019. |
| **Metaclasificadores flexibles** · §4.6 | La hipótesis se cae al revés de lo previsto: al corregir la fuga, los modelos flexibles **empeoran**. LightGBM y TabPFN, medidos por primera vez sobre los checkpoints vigentes, no superan a la regresión logística. |
| **Proyección mayor** · §4.5 | Ni ampliar la compresión interna a 1024 o 2048, ni quitarla del todo, mejora nada. La hipótesis de que `virchow_v1` estaba «estrangulado» por la reducción 2560→512 queda descartada. |
| **Cervical con más particiones** · §4.3 | Con 20 particiones nuevas en vez de 5, los embeddings siguen sin batir a las probabilidades: −0,0013 con un MDE de 0,0063. Es el nulo mejor acotado de la campaña. |
| **Optimización de pesos** | La mejora de +0,0052 de la ficha N5 solo aparece optimizando sobre la partición con fuga. Sobre validación retenida, el método queda *por debajo* de una regresión logística simple. |

---

## Anexo A · Glosario

<a id="g-wsi"></a>
### WSI — *Whole-Slide Image*
La digitalización completa de un portaobjetos de biopsia. Son imágenes de varios
gigapíxeles, imposibles de meter enteras en una red neuronal.

<a id="g-parche"></a>
### Parche — *patch*
Cuadradito de 224×224 píxeles recortado de la WSI. Cada biopsia produce entre 3.700 y 6.600,
y es la unidad con la que trabaja el sistema.

<a id="g-fundacional"></a>
### Modelo fundacional — *foundational model*
Red neuronal grande que otros ya han entrenado con millones de imágenes de patología, y que
aquí se usa tal cual, sin reentrenar. Su trabajo es convertir un parche en números.

En este proyecto hay ocho: `ctranspath`, `conch_v1_5`, `uni_v2`, `virchow_v1`, `virchow2`,
`uni_v1`, `phikon_v2` y `hoptimus1`.

<a id="g-embedding"></a>
### Embedding
La lista de números con la que un modelo fundacional describe un parche. Según el modelo son
entre 768 y 2.560 números. Dos parches parecidos tienen embeddings parecidos.

<a id="g-abmil"></a>
### ABMIL — *Attention-Based Multiple Instance Learning*
La pieza que resume miles de parches en una sola predicción. El problema que resuelve es que
la etiqueta («este tumor tiene TP53 mutado») está en la biopsia entera y no en cada parche:
nadie ha marcado qué parches concretos son los relevantes. El ABMIL lo deduce solo.

<a id="g-atencion"></a>
### Atención — *attention*
La puntuación que el ABMIL da a cada parche según lo importante que le parece. Las
puntuaciones suman 1 entre todos los parches, de modo que con miles de parches a la mayoría
le toca una miga.

Todo el hallazgo positivo de esta campaña consiste en quedarse con los parches de puntuación
más alta y descartar el resto.

<a id="g-fusion"></a>
### Fusión temprana y fusión tardía
**Tardía** es lo que hace el sistema actual: cada modelo trabaja por separado hasta producir
su probabilidad, y solo al final se juntan las tres cifras (punto **C** del diagrama).

**Temprana** es juntar los embeddings de cada parche antes de resumirlos, para que un único
ABMIL vea toda la información a la vez (punto **A**). Se probó y no funciona.

<a id="g-meta"></a>
### Metaclasificador — *stacking*
El modelo pequeño que recibe las probabilidades de los tres modelos base y produce la
predicción final. Aquí suele ser una regresión logística, es decir, poco más que una media
ponderada aprendida.

<a id="g-fold"></a>
### Fold — *partición*
Una división concreta de los datos en una parte para entrenar y otra para examinar. Todo se
repite con 50 divisiones distintas para que el resultado no dependa de haber tenido suerte
con una.

Las biopsias de un mismo paciente van siempre juntas al mismo lado, para que el modelo no
pueda hacer trampa reconociendo al paciente.

<a id="g-oof"></a>
### Out-of-fold
El metaclasificador se entrena normalmente con predicciones que los modelos base hicieron
sobre datos que ya habían visto, así que le llegan demasiado buenas: aprende a confiar en una
calidad que luego no existe.

Las predicciones *out-of-fold* se generan siempre sobre datos no vistos, lo que corrige el
problema. Se probó y no mejora el resultado.

<a id="g-auc"></a>
### AUC — *Área bajo la curva ROC*
La medida principal de este informe. Responde a: si cojo un caso positivo y uno negativo al
azar, ¿con qué probabilidad el modelo puntúa más alto al positivo?

**0,50** es azar puro y **1,00** es perfecto. Los modelos de aquí se mueven entre 0,68 y
0,82. Su ventaja es que no depende de dónde se ponga el corte para decidir sí o no.

<a id="g-kappa"></a>
### Kappa, bacc, F1
Medidas complementarias que sí dependen del corte elegido. **bacc** es el acierto equilibrado
entre las dos clases; **kappa** descuenta los aciertos que se habrían obtenido por azar;
**F1** equilibra precisión y cobertura sobre la clase positiva.

Aparecen porque un modelo puede ordenar bien los casos (buen AUC) y aun así fallar al
decidir, si el corte está mal puesto. Distinguir ambas cosas resultó ser clave para desmontar
el resultado P5.

---

## Anexo B · Las pruebas estadísticas

Qué se calcula, qué pregunta responde cada cosa y cómo se lee. Se aplica igual en los diez
experimentos.

<a id="s-pareado"></a>
### Comparación pareada
**Pregunta:** ¿la versión nueva es mejor que la vieja?

Como las dos versiones se evalúan sobre *exactamente las mismas 50 divisiones de datos*, se
comparan partición a partición en lugar de comparar dos medias sueltas. Esto elimina el ruido
de que una división sea más fácil que otra y detecta diferencias mucho más pequeñas.

*Por qué importa aquí:* el informe consolidado documenta que comparar intervalos
independientes ocultó un efecto real en un conjunto de datos. Todas las cifras de este
informe son pareadas.

<a id="s-delta"></a>
### Δ (delta)
**Qué es:** la mejora media, en puntos de AUC, de la versión nueva sobre la vieja. Positivo
es mejor.

Un Δ de +0,0388 significa que el AUC sube casi cuatro centésimas. Como referencia, la
diferencia entre el mejor y el peor modelo base es de unas diez centésimas.

<a id="s-mde"></a>
### MDE — *mínimo efecto detectable*
**Pregunta:** ¿tenía este experimento capacidad de detectar el efecto que busca?

Es la mejora más pequeña que el experimento podría distinguir del azar con los datos de que
dispone. **Si Δ es menor que el MDE, no se puede afirmar nada**, ni siquiera con un resultado
que parezca favorable.

*Por qué es la cifra más importante de este informe:* permite distinguir «hemos comprobado
que no funciona» de «no teníamos datos para saberlo». Un resultado nulo sin su MDE es una
frase vacía. Es también lo que desmonta el resultado P5, cuya mejora publicada era menor que
su propio MDE.

<a id="s-ic"></a>
### IC 95 % — *intervalo de confianza por bootstrap*
**Pregunta:** ¿entre qué valores está razonablemente la mejora real?

Se calcula remuestreando los resultados 10.000 veces para ver cuánto bailaría la cifra si se
repitiera el experimento. Si el intervalo incluye el cero, la mejora podría perfectamente ser
nula.

<a id="s-wilcoxon"></a>
### Wilcoxon de rangos con signo
**Pregunta:** ¿la mejora es demasiado consistente para ser casualidad?

Es la prueba que produce el **p-valor**. Se usa esta y no la prueba t clásica porque no exige
que los datos sigan una campana de Gauss, cosa que aquí no se cumple.

Un p-valor de 0,03 significa: «si en realidad no hubiera ninguna mejora, vería un resultado
así de bueno un 3 % de las veces». Cuanto más bajo, más difícil es atribuirlo al azar.

<a id="s-holm"></a>
### Corrección de Holm
**Pregunta:** ¿el resultado aguanta sabiendo que se han probado muchas cosas a la vez?

Si se hacen 36 comparaciones, es de esperar que una o dos salgan «significativas» por pura
casualidad. Holm endurece el listón en proporción al número de pruebas realizadas.

*Cómo leerlo:* el **p-Holm** es la cifra que cuenta, no el p-valor a secas. Por debajo de
0,05 el hallazgo es firme. Varios resultados de esta campaña tenían buen p-valor y se caen al
aplicar Holm.

<a id="s-gep"></a>
### G/E/P — *ganados, empatados, perdidos*
**Pregunta:** ¿la mejora es constante o viene de unos pocos casos afortunados?

En cuántas de las 50 divisiones la versión nueva gana, empata o pierde. Un 35/1/14 es una
mejora repartida; un 27/6/17 está muy cerca de lo que daría una moneda.

**Los empates hay que declararlos siempre.** Omitirlos es lo que hacía parecer sólido el
resultado P5: «gana en 27 de 50» suena bien hasta que se ve que hubo 6 empates y 17 derrotas.

<a id="s-suelo"></a>
### Suelo de reproducibilidad
**Pregunta:** ¿esta mejora sobreviviría a repetir el experimento?

Volver a lanzar el mismo entrenamiento produce variaciones de entre 0,004 y 0,010 de AUC solo
por el azar interno del proceso. Cualquier mejora de ese tamaño es indistinguible de
relanzar el trabajo con otra semilla.

**Es lo que hunde la optimización de pesos**, cuya mejora de +0,0052 caía justo dentro de ese
margen.

<a id="s-baseline"></a>
### Recalcular el punto de partida
**Regla de trabajo, no una prueba:** nunca se compara contra una cifra guardada de otro día.
El punto de partida se vuelve a calcular en la misma ejecución que la versión nueva.

*Por qué:* el propio informe consolidado documenta que las cifras anteriores al 21 de agosto
se calcularon contra predicciones que ya no existen en el disco y no reproducen.

---

## Anexo C · Notas de método

**Todo punto de partida se recalculó en la misma ejecución** que la versión con la que se
comparaba. No se ha reutilizado ninguna cifra guardada de campañas anteriores.

**El número de particiones se contó siempre**, nunca se dio por supuesto. Un conjunto tiene 5
y no 50, y darlo por hecho ya había dado por fallido en el pasado un entrenamiento correcto.

**La partición nueva de `cervical_subtype`** se generó con validación cruzada estratificada
agrupando por paciente, y se verificó que ninguna clase falta en ningún bloque y que ningún
paciente aparece a ambos lados. Vive en un directorio propio para no alterar la partición de
5 usada por los resultados previos.

**Dos fallos reales de código encontrados de paso.** La rutina de fusión temprana nunca
guardaba su partición de validación, de modo que no tenía con qué ajustar el corte de
decisión; y su cargador de datos pedía el doble de memoria de la necesaria. Ambos corregidos.

**El hallazgo positivo recibió el escrutinio más severo de la campaña:** corrección de Holm
sobre las 36 comparaciones completas en vez de solo sobre su familia, descarte explícito de
que la selección de parches pudiera estar usando las respuestas, y comparación a igualdad de
concentración para separar el efecto de truncar del de afinar.

**Una cautela honesta:** las cinco correcciones apuntan en la misma dirección —el sistema
aporta menos de lo publicado— y todas nacen de controles que la campaña original no llegó a
hacer. Eso las hace verosímiles, pero también aconseja que se reproduzcan de forma
independiente antes de darlas por definitivas.

---

*10 líneas de investigación · 46 trabajos de cálculo · 4 conjuntos de datos*
*`cptac_brca`/TP53 · `cptac_gbm`/TP53 · `bc_therapy`/estado ER · `cervical_subtype`/subtipo*
*Estadística: bootstrap de 10.000 repeticiones, Wilcoxon de rangos con signo, corrección de
Holm-Bonferroni, MDE al 80 % de potencia*
