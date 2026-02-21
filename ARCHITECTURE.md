(WIP file - under review and waiting to be translated)

MANIFIESTO TÉCNICO: Diseño y arquitectura del proyecto
======================================================

CAPA 1: ESTRUCTURA
-------------------------------------------

*Decisiones sobre cómo se organiza el código para que sea mantenible y escalable.*

-   **Estructura Modular (`src/`):**

    -   No uso un script monolítico ("código espagueti") porque dificulta el testeo unitario y la reutilización. Separar `train`, `model` y `dataset` permite que varios ingenieros trabajen en archivos distintos sin conflictos de *merge*.

-   **Configuración Desacoplada (`config.py`):**

    -   Extraje rutas, hiperparámetros y constantes a un archivo separado.

    -   **Por qué:** Evita "Magic Numbers" dispersos por el código. Permite cambiar el experimento (ej: subir epochs) sin tocar la lógica del bucle de entrenamiento.

-   **Type Hinting (Tipado estático):**

    -   Uso `def func(x: int) -> str`.

    -   **Por qué:** Python es dinámico, pero en equipos grandes esto reduce bugs de tipos y mejora el autocompletado en el IDE. Es un estándar de *Clean Code*.

-   **Entorno Virtual (`venv`):**

    -   **Por qué:** Aísla las dependencias. Evita que una actualización de PyTorch en mi sistema rompa este proyecto específico.

-   **Git Ignore (`.gitignore`):**

    -   **Decisión:** Excluir `data/` y `models/`.

    -   **Por qué:** GitHub no es un almacén de binarios. Subir archivos grandes ralentiza el clonado y es mala práctica de DevOps.

-   **Punto de entrada (`if __name__ == "__main__":`):**

    -   **Por qué:** Permite que mis scripts (`train.py`, `dataset.py`) puedan ser importados como módulos por otros scripts (ej: para tests) sin que se ejecute su código automáticamente.

* * * * *

CAPA 2: INGENIERÍA DE DATOS (ETL)
---------------------------------

*Decisiones tomadas en `prepare_data.py`.*

-   **Parsing de XML:**

    -   **Decisión:** Leer la etiqueta `<name>` dentro del XML en lugar de usar el nombre del archivo JPG.

    -   **Por qué:** Robustez. Los nombres de archivo son vulnerables a errores de escritura y no tienen una estructura lo suficientemente estricta como para ser fiable en este contexto. Usaremos los metadatos.

-   **Filtrado de extensiones:**

    -   **Decisión:** Filtrar por `endswith(('.jpg', '.bmp', ...))`.

    -   **Por qué:** El dataset puede contener archivos de sistema o temporales que podrían interrumpir la carga de imágenes.

-   **Gestión de errores (try/except):**

    -   **Decisión:** Si una imagen referenciada en el XML no existe, la salto y logueo el error, no rompo la ejecución.

    -   **Por qué:** Dada la gran cantidad de datos que podemos procesar con este código, la probabilidad de que haya un error tratando alguno de los archivos es muy alta, de esta forma evitamos interrumpir el proceso completo por culpa de algunos errores aislados.

-   **Estructura de carpetas (`processed/class_name`):**

    -   **Decisión:** Crear subcarpetas por clase.

    -   **Por qué:** Es el estándar que espera `ImageFolder` de PyTorch y facilita la inspección visual humana.

* * * * *

CAPA 3: EL DATASET Y DATALOADER
-------------------------------

*Decisiones en `dataset.py` y la carga de datos.*

-   **Herencia de `torch.utils.data.Dataset`:**

    -   **Decisión:** Crear una clase propia `SteelSurfaceDataset`.

    -   **Por qué:** Me da control total sobre cómo se lee el archivo (ej: si quisiera leer desde un servicio de almacenamiento remoto como Amazon S3 o una base de datos SQL en el futuro) frente a usar el `ImageFolder` genérico.

-   **Conversión a RGB (`.convert("RGB")`):**

    -   **Decisión crítica:** Las imágenes de acero son en escala de grises (1 canal), pero las fuerzo a RGB (3 canales).

    -   **Por qué:** ResNet18 pre-entrenada espera 3 canales de entrada. Si le paso 1 canal, la primera capa convolucional fallará por error de dimensiones de matriz.

-   **Lectura con PIL (Pillow):**

    -   **Por qué:** Es el estándar en Python, más seguro y eficiente en memoria que `cv2` para operaciones de carga simples en PyTorch.

-   **Num workers (`num_workers=2`):**

    -   **Decisión:** Usar sub-procesos para cargar datos.

    -   **Por qué:** Mientras la GPU entrena, la CPU carga el siguiente batch en paralelo. Evita que la GPU se quede esperando ("GPU Starvation"). No puse 16 porque en Colab el overhead de crear hilos superaría el beneficio.

-   **Pin Memory (`pin_memory=True` implícito o explícito):**

    -   **Por qué:** Acelera la transferencia de RAM (CPU) a VRAM (GPU) al usar memoria paginada bloqueada.

-   **Shuffle (`shuffle=True` en Train, `False` en Val):**

    -   **Por qué:** En Train es obligatorio para romper correlaciones temporales y que el gradiente sea estocástico real. En Val es perjudicial porque quiero resultados deterministas para comparar.

* * * * *

CAPA 4: PRE-PROCESAMIENTO Y AUGMENTATION
----------------------------------------

*Decisiones sobre los tensores.*

-   **Resize (224, 224):**

    -   **Por qué:** Es la resolución estándar de entrada de las arquitecturas ResNet. Aunque las CNNs pueden aceptar otros tamaños, usar el nativo optimiza el uso de las *features* espaciales aprendidas.

-   **Normalización (`mean=[0.485...]`, `std=[0.229...]`):**

    -   **Decisión:** Usar las estadísticas de ImageNet.

    -   **Por qué:** Los pesos de la red pre-entrenada se ajustaron viendo imágenes con esa distribución de color. Si le paso mis imágenes sin normalizar (valores 0-255 o 0-1 planos), las activaciones de las neuronas se saturarán o serán muy bajas, impidiendo el aprendizaje.

-   **ToTensor:**

    -   **Por qué:** Convierte la imagen (H, W, C) con valores [0, 255] a un Tensor (C, H, W) con valores [0.0, 1.0]. PyTorch usa el formato "Canal primero" (Channel First), a diferencia de TensorFlow.

-   **RandomRotation (30 grados):**

    -   **Por qué:** Las grietas no tienen orientación fija gravitacional. Una grieta horizontal es igual de mala que una vertical. Enseño invarianza rotacional.

-   **ColorJitter (Brillo/Contraste):**

    -   **Por qué:** Simula cambios en la iluminación de la nave industrial o sensores de cámara diferentes/sucios.

* * * * *

CAPA 5: ARQUITECTURA DEL MODELO
-------------------------------

*Decisiones en `model.py`.*

-   **Transfer Learning (Weights=Default):**

    -   **Por qué:** Aprovechar la capacidad de extracción de características (bordes, texturas) ya aprendidas en millones de imágenes. Entrenar desde cero con 1.800 fotos llevaría al *overfitting* inmediato.

-   **Arquitectura ResNet18:**

    -   **Por qué:** Tiene "Skip Connections" (conexiones residuales) que evitan el problema del desvanecimiento del gradiente (*Vanishing Gradient*) en redes profundas. Es ligera (aprox 11M parámetros) frente a VGG16 (138M), lo que la hace ideal para despliegue en industria.

-   **Sustitución de la "Head" (`model.fc`):**

    -   **Decisión:** `model.fc = nn.Linear(num_ftrs, 6)`.

    -   **Por qué:** La red original clasifica 1000 clases. Yo necesito 6. Corto la última capa y pongo una nueva no entrenada que proyecte a mi espacio de 6 dimensiones.

-   **Fine-Tuning completo (No congelar capas):**

    -   **Decisión:** No he hecho `param.requires_grad = False` en las capas base.

    -   **Por qué:** Las texturas del acero son muy diferentes a las fotos de ImageNet (perros, coches). Necesito que toda la red se adapte ligeramente a este nuevo dominio visual ("Domain Adaptation").

* * * * *

CAPA 6: ENTRENAMIENTO Y OPTIMIZACIÓN
------------------------------------

*Decisiones en `train.py`.*

-   **CrossEntropyLoss:**

    -   **Por qué:** Aplica internamente `LogSoftmax` + `NLLLoss`. Es numéricamente más estable que aplicar Softmax y luego Logaritmo manualmente. Es la función de coste obligatoria para clasificación multiclase.

-   **Optimizador Adam:**

    -   **Decisión:** Usar Adam en lugar de SGD.

    -   **Por qué:** Adam tiene "momentum" adaptativo por parámetro. En datasets ruidosos o con geometrías de pérdida complejas, converge más rápido y requiere menos ajuste manual del Learning Rate.

-   **Learning rate (0.001):**

    -   **Por qué:** Es el valor por defecto estándar para Adam. Un valor mayor (0.1) haría diverger el entrenamiento; uno menor (0.00001) sería demasiado lento.

-   **Zero grad (`optimizer.zero_grad()`):**

    -   **Por qué:** PyTorch acumula gradientes por defecto (útil para RNNs). En CNNs necesitamos limpiarlos en cada batch o la actualización sería la suma de todo el historial, lo cual sería erróneo.

-   **Device (`.to(device)`):**

    -   **Por qué:** Mover explícitamente tensores y modelo a la GPU. Si uno está en CPU y otro en GPU, PyTorch lanza error de ejecución.

* * * * *

CAPA 7: VALIDACIÓN Y MÉTRICAS
-----------------------------

*Decisiones de evaluación.*

-   **`model.eval()`:**

    -   **Por qué:** Crítico. Cambia el comportamiento de capas específicas.

        -   **Batch normalization:** En `eval`, usa la media/varianza global aprendida, no la del batch actual.

        -   **Dropout:** En `eval`, se desactiva (no apaga neuronas) para usar toda la capacidad de la red.

-   **`torch.no_grad()`:**

    -   **Por qué:** Desactiva el motor de autogradiente. Reduce el consumo de memoria VRAM a la mitad (no necesita guardar valores intermedios para la retropropagación) y acelera el cálculo.

-   **Métrica Accuracy:**

    -   **Por qué:** Métrica base fácil de interpretar.

-   **AverageMeter (Clase Utils):**

    -   **Por qué:** El Loss y Accuracy fluctúan mucho por batch. Necesito un promedio suavizado (Running Average) para tener una visión real del desempeño de la época.

-   **Validation Loss como criterio de éxito:**

    -   **Por qué:** No guardo el modelo con mejor Training Accuracy (eso sería overfitting), guardo el que tiene menor Validation Loss (capacidad de generalización).

* * * * *

CAPA 8: INFERENCIA Y PRODUCCIÓN
-------------------------------

*Decisiones en `inference.py`.*

-   **Carga con `map_location`:**

    -   **Decisión:** `torch.load(path, map_location=device)`.

    -   **Por qué:** Permite cargar un modelo entrenado en GPU dentro de una máquina que solo tiene CPU (ej: un servidor barato de producción) sin que rompa.

-   **Unsqueeze (`image.unsqueeze(0)`):**

    -   **Por qué:** El modelo espera un tensor 4D `(Batch, Channel, Height, Width)`. Una sola imagen es 3D `(C, H, W)`. Añado una dimensión falsa de batch (tamaño 1) para que encaje.

-   **Softmax Final (`F.softmax(outputs)`):**

    -   **Por qué:** La salida del modelo son "Logits" (valores crudos, pueden ser negativos o infinitos). Softmax los normaliza a una distribución de probabilidad que suma 1 (ej: 0.8, 0.1, 0.1).

-   **State Dict (`model.load_state_dict`):**

    -   **Decisión:** Guardo solo los pesos (`state_dict`), no el objeto modelo entero (`torch.save(model)`).

    -   **Por qué:** Guardar el objeto entero usa `pickle` y es frágil si cambia la estructura de la clase o la versión de PyTorch. Guardar los pesos es la forma profesional y portable.

* * * * *

CAPA 9: PRESENTACIÓN Y APLICACIÓN
---------------------------------------------

*Decisiones de README y muestra de resultados.*

-   **Matriz de Confusión:**

    -   **Por qué:** El Accuracy esconde errores graves. Necesito saber si estoy confundiendo "Scratches" con "Crazing".

-   **F1-Score:**

    -   **Por qué:** En industria, el coste de un Falso Positivo y un Falso Negativo no es igual. F1 armoniza Precision y Recall.

-   **Grad-CAM (Explainability):**

    -   **Por qué:** Confianza del usuario. Si no puedo explicar por qué la IA marcó la pieza como defectuosa, el jefe de planta no comprará el software.

-   **Reproducibilidad (Seeds):**

    -   **Por qué:** Aunque hay aleatoriedad, fijar semillas (`random_seed=42`) permite que si tú y yo corremos el código, obtengamos resultados similares para depurar.