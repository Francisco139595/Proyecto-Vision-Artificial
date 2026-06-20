from ultralytics import YOLO
# Importamos la clase YOLO de la librería ultralytics.


def main():

# Cargamos tu modelo personalizado.
    # En lugar de usar un modelo preentrenado genérico (como yolo11n.pt), le indicamos la ruta 
    # donde se guardaron los resultados de tu entrenamiento. 'best.pt' es el archivo con los 
    # pesos (parámetros) que lograron el mejor rendimiento validado.

    model = YOLO("runs/detect/modelo_entrenado/weights/best.pt")

# Un simple print en consola para dar instrucciones sobre cómo detener el programa.
    # Es importante saber que la 'q' se debe presionar cuando la ventana de video está activa.

    print("Encendiendo cámara web... Presiona la tecla 'q' en la ventana de la cámara para salir.")

   # Inicia el proceso de inferencia (predicción).

    model.predict(source="0", show=True, conf=0.7)

# 'source': Define de dónde saca las imágenes. "0" suele apuntar a la cámara web principal.
# 'show': Al estar en True, le dice a OpenCV que abra una ventana emergente (un "cuadro") 
  # para mostrar el video en vivo con los cuadros delimitadores (bounding boxes) dibujados.
# 'conf': Umbral de confianza (Confidence Threshold). 0.7 significa que el modelo 
 # solo mostrará en pantalla las detecciones de las que esté un 70% o más seguro. 
 # Si no detecta nada, puedes bajar este número (ej. 0.5) para hacerlo más "sensible".

if __name__ == '__main__':

    main() 
# La guarda habitual para asegurar que el código se ejecute correctamente 
# si es el script principal.


        