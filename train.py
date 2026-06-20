# Importamos la clase YOLO desde la librería ultralytics, que nos permite instanciar y usar el modelo.
from ultralytics import YOLO

def main():
    
    # Cargamos un modelo preentrenado. 
    # "yolo11n.pt" carga la versión "nano" (n) de YOLOv11. Es una arquitectura muy ligera y rápida, 
    # ideal para empezar y aplicar "transfer learning" (aprender sobre lo que ya sabe).
    model = YOLO("yolo11n.pt") 

    
    # Iniciamos el proceso de entrenamiento del modelo con nuestro dataset personalizado.
    # Se guardan los resultados (métricas, etc.) en la variable 'results'.
    results = model.train(
        
        # 'data': Ruta al archivo YAML. Este archivo contiene la configuración del dataset 
        # (dónde están las carpetas de imágenes de entrenamiento/validación y los nombres de las clases).
        data="Objec_detection_Yolo.v3-visionv3.yolov11/data.yaml", 
        
        # 'epochs': Número de veces que el modelo analizará el dataset completo. 
        # 300 iteraciones es un buen estándar para asegurar que el modelo aprenda bien las características.
        epochs=300,                
        
        # 'imgsz': Tamaño de las imágenes. Todas las imágenes se redimensionarán a 640x640 píxeles 
        # antes de pasar por la red neuronal.
        imgsz=640,                
        
        # 'batch': Cantidad de imágenes que el modelo procesa en la memoria RAM/VRAM al mismo tiempo 
        # antes de actualizar sus parámetros. 32 requiere una tarjeta gráfica con buena memoria.
        batch=32,                   
        
        # 'name': Nombre del directorio que se creará dentro de la carpeta 'runs/detect/'. 
        # Allí se guardarán los pesos finales de tu modelo y las gráficas de rendimiento.
        name="modelo_entrenado"    
    )

    # Imprime un mensaje en la consola para confirmar que el ciclo de 300 epochs ha terminado sin errores.
    print("¡Entrenamiento completado!")

# Este bloque asegura que la función main() solo se ejecute si ejecutas este script directamente.
# Es una buena práctica en Python, especialmente en Windows (y al usar PyTorch) para evitar 
# errores al usar múltiples hilos de procesamiento (multiprocessing).
if __name__ == '__main__':
    main()