# Proyecto de Detección e Identificación de Vacas

Este repositorio contiene un conjunto de Jupyter Notebooks y scripts para implementar un modelo de aprendizaje automático capaz de reconocer la identidad única de vacas lecheras. El proyecto se ha desarrollado como parte del trabajo academico final de la carrera Ingenieria en Sistemas de la Universidad ORT Uruguay bajo la supervisión de la tutora Mikaela Pisani. 

## Autores

- Chia Hung Hsieh (196306)
- Joselen Cecilia (233552)

## Para ejecutar estos archivos, antes sigue estos pasos:

1. Clona el repositorio en tu máquina local.
2. Crea un Entorno Virtual (opcional pero recomendado): Para evitar conflictos entre las dependencias de diferentes proyectos, es recomendable crear un entorno virtual. Puedes usar virtualenv o venv si estás en Python 3. Si estás utilizando herramientas como Anaconda, puedes crear un entorno conda :
    - `python -m venv mi_entorno_virtual`
    - `source mi_entorno_virtual/bin/activate`
3. Instala las Dependencias: Utiliza pip para instalar las dependencias desde el archivo requirements.txt:
    - `pip install -r requirements.txt`

    Esto instalará todas las bibliotecas y paquetes necesarios para ejecutar el proyecto.
4. Ejecuta el Cuaderno: sigue los pasos de cada cuaderno como se indica en los siguientes puntos.

## Descripción de los Archivos

### `helper_video_frame_precessor.ipynb`

Este cuaderno tiene como objetivo principal procesar videos, convertirlos en una serie de fotogramas (frames).

En este cuaderno se definen una serie de funciones, incluida la función `video_to_frames` que se encarga de convertir el video en una secuencia de fotogramas (imágenes individuales) y guardar cada fotograma como un archivo JPG en la carpeta de salida. Los fotogramas se nombran de manera secuencial con el nombre base del archivo y un contador.

Dentro del cuaderno se encuentra el detalle de que hace cada funcion paso a paso.

Para ejecutar este cuaderno sigue los pasos generales enumerados anteriormente y luego:
1. Abre el cuaderno `helper_video_frame_precessor.ipynb`.
2. Sigue las instrucciones en el cuaderno para ejecutar las celdas de código.
3. Asegúrate de que los archivos de video que deseas procesar estén en la ubicación correcta o ajusta las rutas en el cuaderno según sea necesario.


### `cow_augmentation.ipynb`

Este cuaderno se encarga de realizar la aumentación de datos (data augmentation) para un conjunto de imágenes. La aumentación de datos es una técnica comúnmente utilizada en el entrenamiento de modelos de aprendizaje automático, especialmente en el caso de redes neuronales convolucionales (CNN), para aumentar la diversidad de datos de entrenamiento y mejorar el rendimiento del modelo.

En este cuaderno se definen una serie de funciones, incluida la función `data_aug` para realizar la aumentación de datos en una imagen. Esta función aplica rotación, cambios de brillo, contraste, calidad JPEG y saturación a una imagen para generar variantes de la misma. También se define la función `create_augmentation` que aplica la aumentación de datos a un conjunto de imágenes.

Dentro del cuaderno se encuentra el detalle de que hace cada funcion paso a paso.

Para ejecutar este cuaderno sigue los pasos enumerados anteriormente y luego:
1. Abre el cuaderno `cow_augmentation.ipynb`.
2. Sigue las instrucciones en el cuaderno para ejecutar las celdas de código.
3. Esto realizará la aumentación de datos en las imágenes de `cow` y las imágenes `negatives` en el directorio de datos.

### `cow_detection.ipynb`

Este cuaderno se encarga de entrenar un modelo YOLO versión 8 utilizando la biblioteca Optuna para la optimización de hiperparámetros y, posteriormente, utilizar el modelo entrenado para realizar predicciones en imágenes

Luego de entrenado, el modelo se utiliza para realizar predicciones en imágenes utilizando el modelo YOLO entrenado. Las imágenes de entrada se procesan para detectar objetos (en este caso, vacas) y se recortan y redimensionan para mantener una forma cuadrada.

Dentro del cuaderno se encuentra el detalle de que hace cada funcion paso a paso.

Para ejecutar este cuaderno sigue los pasos generales enumerados anteriormente y luego:
1. Sigue las instrucciones en el cuaderno para ejecutar las celdas de código. Esto realizará la optimización de hiperparámetros y las predicciones en imágenes.

### `cow_identification.ipynb`

Este cuaderno de Jupyter se utiliza para entrenar un modelo de red Siamesa utilizando TensorFlow y Optuna para la optimización de hiperparámetros. El modelo de red Siamesa se utiliza para verificar si dos imagenes de vacas son similares o no.

Este cuaderno define un modelo Siamese utilizando la arquitectura de redes neuronales preentrenadas, como `ResNet152` o `Xception`.
También define una capa de distancia para calcular las distancias entre pares de imágenes en el espacio de características.
Crea un modelo Siamese completo que toma tres entradas (ancla, positiva y negativa) y produce distancias como salidas.

Dentro del cuaderno se encuentra el detalle de que hace cada funcion paso a paso.

Para ejecutar este cuaderno sigue los pasos generales enumerados anteriormente y luego:
1. Debes asegurarte de tener un conjunto de datos de tripletes preparado y almacenado en el directorio especificado en el código.
2. Sigue las instrucciones en el cuaderno para ejecutar las celdas de código. Al finalizar, podrás acceder a los modelos entrenados y a las métricas de rendimiento en las carpetas específicas creadas durante el proceso.

Es importante tener en cuenta que este cuaderno está diseñado para tareas avanzadas de aprendizaje profundo y optimización de hiperparámetros, por lo que puede llevar tiempo completar la ejecución, especialmente si se realizan muchas iteraciones de entrenamiento. Asegúrate de tener acceso a suficiente potencia de cómputo (especialmente GPU) y recursos de almacenamiento antes de ejecutarlo.

### `cow_identification_test.ipynb`

Este cuaderno realiza la predicción en un conjunto de pruebas utilizando un modelo Siamese y genera métricas de evaluación. 

Se ejecuta la función principal `create_test_metrics`, que carga el modelo Siamese entrenado y realiza la clasificación en las imágenes de prueba. Luego, se calcula la precisión global, la precisión para las imágenes positivas y la precisión para las imágenes negativas. También se guarda un archivo JSON con los resultados. Se genera y muestra una matriz de confusión que muestra las predicciones frente a las etiquetas verdaderas.

Dentro del cuaderno se encuentra el detalle de que hace cada funcion paso a paso.

Para ejecutar este cuaderno sigue los pasos generales enumerados anteriormente y luego:
1. Sigue las instrucciones en el cuaderno para ejecutar las celdas de código. Esto realizará la predicción en el conjunto de pruebas y calculará las métricas de evaluación utilizando el modelo Siamese.

Una vez que hayas ejecutado el cuaderno, obtendrás métricas de evaluación del modelo Siamese en el conjunto de pruebas. Asegúrate de seguir las instrucciones en el cuaderno para entender los resultados y cómo se almacenan.

### `helper_prediction_pipeline.ipynb`

Este cuaderno implementa un flujo de trabajo para realizar predicciones utilizando un modelo YOLO (You Only Look Once) junto con un modelo Siamese preeviamente entrenados. Simulando un flujo real de como se usan los modelso entrenados anteriormente.

En este cuaderno se definen una serie de funciones, incluida la función principal  `video_to_frames_with_prediction` la cual se ejecuta utilizando los modelos cargados. Esta función toma un video como entrada y realiza lo siguiente:
- Detecta vacas en cada fotograma del video utilizando YOLO.
- Luego, utiliza el modelo Siamese para verificar si las vacas detectadas son las mismas que las que se encuentran en una base de datos de imágenes de validación.
- Registra las detecciones y los resultados en un archivo CSV y guarda imágenes que muestran las vacas detectadas junto con las vacas de la base de datos.

Dentro del cuaderno se encuentra el detalle de que hace cada funcion paso a paso.

Para ejecutar este cuaderno sigue los pasos generales enumerados anteriormente y luego:
1. Sigue las instrucciones en el cuaderno para ejecutar las celdas de código. Esto realizará el proceso de predicción y verificación de identidad en el video de entrada.

Una vez que hayas ejecutado el cuaderno, habrás realizado la detección de vacas en un video y verificado su identidad utilizando el modelo Siamese. Asegúrate de seguir las instrucciones en el cuaderno para entender los resultados y cómo se almacenan.

## Links a los dataset y datos en S3

[Dataset Siamese Model](https://cowid.s3.amazonaws.com/raw_videos.zip)

[Dataset YOLO](https://cowid.s3.amazonaws.com/yolo_dataset_basemodel.zip)

[Videos de las tomas en tambos](https://cowid.s3.amazonaws.com/raw_videos.zip)


