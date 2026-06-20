Detector de Pyraminx y Cubos Rubik con IA (YOLOv11)

Este proyecto utiliza Inteligencia Artificial, específicamente técnicas avanzadas de Visión por Computadora, para detectar y clasificar distintos tipos de cubos Rubik en tiempo real a través de la cámara web de la computadora.

El sistema ha sido entrenado para localizar en la imagen y diferenciar entre tres clases distintas con alta precisión:

Pyraminx (Rompecabezas mecánico en forma de tetraedro o pirámide)

Cubo-Resuelto (Cubo Rubik tradicional 3x3x3 en su estado armado)

Cubo-No-resuelto (Cubo Rubik tradicional 3x3x3 en estado desarmado o mezclado)

Datos del Proyecto

Materia: Inteligencia Artificial

Semestre: 6to Semestre

Autores:
 Francisco Ramiro Gomez Arce #23310350

David Antonio Lepe Bermudes #23310360

Marco Teórico y Tecnologías

Para la realización de este proyecto se utilizó YOLOv11 (You Only Look Once), uno de los modelos de detección de objetos más rápidos y precisos en el estado del arte de la Inteligencia Artificial. A diferencia de otros clasificadores de imágenes que analizan la imagen por partes, YOLO aplica una única red neuronal a toda la imagen, dividiéndola en una cuadrícula y prediciendo los cuadros delimitadores (bounding boxes) y las probabilidades de clase simultáneamente.

Esto permite que nuestro sistema pueda procesar el flujo de video de la cámara web en tiempo real sin retrasos significativos.

Requisitos y Preparación del Entorno

Para que este proyecto funcione en cualquier computadora, es necesario contar con un entorno de Python instalado (se recomienda versión 3.8 o superior).

Para instalar todas las dependencias necesarias de manera automática, abre la terminal en la raíz del directorio del proyecto y ejecuta el siguiente comando:

pip install -r requirements.txt


(Nota: Este proyecto está configurado para utilizar la cámara web predeterminada del equipo. Asegúrese de otorgar los permisos de uso de cámara en la configuración de privacidad de su sistema operativo antes de la ejecución).

Instrucciones de Uso

El flujo de trabajo del proyecto se ha simplificado en dos scripts principales para facilitar su uso y demostración:

1. Inferencia en Tiempo Real (predict.py)

Este script inicializa la cámara web y carga el modelo de pesos final (best.pt) obtenido tras nuestro entrenamiento. Analiza cada fotograma del video en vivo y dibuja cuadros delimitadores alrededor de los objetos detectados junto con su porcentaje de confianza.

Comando de ejecución:

python predict.py


Para finalizar la ejecución y cerrar la ventana de la cámara, presione la tecla q.

2. Reentrenamiento del Modelo (train.py)

En caso de que se desee escalar el proyecto, agregar nuevas clases o alimentar la red neuronal con un nuevo dataset de imágenes, se provee el script de entrenamiento. Este archivo configura los hiperparámetros necesarios para enseñar al modelo desde cero.

Comando de ejecución:

python train.py


Resultados del Entrenamiento

El proceso de entrenamiento de la red neuronal se completó de manera exitosa tras abarcar un total de 300 épocas. Durante este periodo, el modelo ajustó sus pesos internos para minimizar la pérdida (loss function) y maximizar su capacidad de generalización.

Precisión General (mAP): El modelo superó el 75% de precisión media en la detección en tiempo real para las tres clases.

Comportamiento del Modelo: Tras las 300 épocas, el sistema demostró ser altamente robusto ante variaciones de iluminación y diferentes ángulos de visión de los cubos, diferenciando de manera efectiva la estructura de la pirámide (Pyraminx) frente a los patrones de colores de un cubo 3x3 armado y desarmado.

Conclusiones

Este proyecto demuestra la viabilidad de implementar modelos de redes neuronales convolucionales ligeras en equipos de cómputo estándar. La clasificación en tiempo real de objetos físicos complejos como el Pyraminx abre la puerta a futuras aplicaciones industriales, tales como sistemas automatizados de control de calidad o asistencia robótica en ensamblaje.