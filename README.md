# Visión por Computadora — README

Este repositorio acompaña la presentación **“Computer Vision”** y resume sus ideas clave, flujo de trabajo y áreas de aplicación. Al final encontrarás una breve **demostración** que utiliza la imagen `image.png`.

https://cs231n-stanford-edu.translate.goog/?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc

---

## Índice
1. [¿Qué es Computer Vision?](#qué-es-computer-vision)
2. [Historia y evolución](#historia-y-evolución)
3. [Importancia en la actualidad](#importancia-en-la-actualidad)
4. [Conceptos básicos de imágenes](#conceptos-básicos-de-imágenes)
5. [Kernels y convoluciones](#kernels-y-convoluciones)
6. [Tareas fundamentales](#tareas-fundamentales)
7. [Del píxel al entendimiento (Deep Learning)](#del-píxel-al-entendimiento-deep-learning)
8. [Pipeline de un proyecto de CV](#pipeline-de-un-proyecto-de-cv)
9. [Limitaciones](#limitaciones)
10. [Aplicaciones y oportunidades de negocio](#aplicaciones-y-oportunidades-de-negocio)
11. [Demostración](#demostración)

---

## ¿Qué es Computer Vision?
La **visión por computadora** es un campo de la inteligencia artificial que permite a las máquinas **ver, interpretar y comprender** el mundo mediante **imágenes** y **video**.

---

## Historia y evolución
| Década | Hitos destacados |
|---|---|
| 1960s | Primeros experimentos conectando cámaras a computadores. |
| 1970s | Nacen técnicas de **procesamiento de imágenes**. |
| 1980s | Análisis en distintas escalas (**scale-space**). |
| 1990s | **Calibración de cámaras** y **reconstrucción 3D**. |
| 2000s | Avances en **vehículos autónomos** y **reconocimiento facial**. |
| 2010s | Auge de **redes neuronales profundas (CNNs)**. |

---

## Importancia en la actualidad
- **Vida cotidiana**: desbloqueo facial, filtros en apps, organización de fotos.
- **Industria**: inspección visual, control de calidad, conteo y trazabilidad.
- **Transporte**: asistencia al conductor, percepción para vehículos autónomos.
> En general, **automatiza y agiliza tareas** que antes eran muy laboriosas para humanos.

---

## Conceptos básicos de imágenes
- **Imagen como matriz**: una imagen es una matriz de **pixeles** (números).
- **Escala de grises**: un único canal con valores **0–255** (negro → blanco). Formas comunes: `(H, W)` o `(H, W, 1)`.
- **Imágenes a color (RGB)**: tres canales (rojo, verde, azul), mismos **alto** y **ancho**; cada canal usa valores **0–255**.

---

## Kernels y convoluciones
- Un **kernel** es una **pequeña matriz** que “se desliza” sobre la imagen para **resaltar** o **extraer** características.
- Operadores clásicos de detección de bordes: **Sobel** (horizontal/vertical) y **Canny** (detección robusta de bordes).
- El resultado de aplicar un kernel se denomina **convolución**.

---

## Tareas fundamentales
1. **Clasificación**: *¿Qué hay en la imagen?*
2. **Detección de objetos**: *¿Dónde está?* (cajas delimitadoras)
3. **Segmentación**: *¿Cómo se ve exactamente?* (máscaras por píxel)
4. **Detección de pose / *Keypoints***: *¿En qué posición se encuentra?*

---

## Del píxel al entendimiento (Deep Learning)
En lugar de programar manualmente “busca bordes / círculos / colores”, las **CNNs** **aprenden jerárquicamente** representaciones a partir de datos:
- Capas iniciales: patrones simples (bordes, texturas).
- Capas profundas: conceptos más abstractos (partes, objetos).
- Requieren **grandes cantidades de datos** y **cómputo** para entrenarse.

---

## Pipeline de un proyecto de CV
1. **Recolección de datos**  
2. **Etiquetado** (anotaciones: clases, cajas, máscaras, puntos clave)  
3. **Partición del *dataset*** (train/val/test)  
4. **Data augmentation** (rotaciones, recortes, cambios de brillo, etc.)  
5. **Entrenamiento** (selección de modelo, hiperparámetros)  
6. **Métricas de evaluación** (precisión, IoU, mAP, F1, etc.)  
7. **Despliegue** (serving en tiempo real o *batch*)  
8. **Monitoreo** (*drift*, rendimiento, re-etiquetado y *retraining* periódicos)

---

## Limitaciones
- **Calidad y sesgo de datos**: mala representación conduce a errores sistemáticos.
- **Costo computacional**: entrenamiento e inferencia pueden ser costosos.
- **Escalabilidad y latencia**: requisitos de tiempo real vs. precisión.
- **Interpretabilidad**: modelos opacos, difícil explicar decisiones.
- **Privacidad y ética**: manejo responsable de datos sensibles (rostros, placas).

---

## Aplicaciones y oportunidades de negocio
- **Ahorro de costos** mediante automatización visual.
- **Robótica** (navegación, manipulación y seguridad).
- **Análisis de patrones de compra** y **monetización de datos**.
- **Mejora de la UX** (búsqueda por imagen, recomendaciones visuales).
- **Automatización de servicios** (kioscos, *self-checkout*, vigilancia inteligente).

---

## Demostración
A continuación se incluye una imagen de ejemplo para ilustrar el flujo de trabajo o resultados (sustituye `image.png` por la imagen correspondiente de tu proyecto si es necesario):

![Demostración](image.png)

https://www4.ujaen.es/~satorres/practicas/practica3_vc.pdf


Identidad: deja la imagen igual (sin cambio).

Eje básico: detector de bordes tipo laplaciano (resalta contornos en todas direcciones).

Desenfoque básico: blur por caja 3×3 (suaviza, quita ruido fino).

Desenfoque Gaussiano: blur gaussiano 5×5 (suaviza de forma más natural).

Enfocar: filtro sharpen 3×3 (aumenta nitidez/contornos).

Realzar: realce/relieve direccional (acentúa bordes con efecto “emboss” ligero).

Sobel horizontal: resalta bordes horizontales (cambios verticales).

Sobel vertical: resalta bordes verticales (cambios horizontales).

Sobel HyV: magnitud del gradiente (bordes en todas direcciones).

Sobel HyV con Blur: igual, pero con suavizado previo (bordes más limpios).

Sobel HyV con Blur y TH: añade umbral (TH) para mostrar solo bordes fuertes.

Sobel colorizado: color = dirección del borde, brillo = intensidad.

Sobel colorizado con Blur: lo mismo, con suavizado para reducir ruido cromático.

