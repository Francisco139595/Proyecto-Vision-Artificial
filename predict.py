from ultralytics import YOLO

def main():
    # Se carga el modelo utilizando los mejores pesos obtenidos durante el entrenamiento.
    model = YOLO("runs/detect/modelo_entrenado/weights/best.pt")

    # Se imprime un aviso en la consola para indicar que la ventana de visualización se cierra con la tecla 'q'.
    print("Encendiendo cámara web... Presiona la tecla 'q' en la ventana de la cámara para salir.")
    
    # Se inicia la predicción en tiempo real con los siguientes parámetros:
    # source="0": Activa la cámara web principal del equipo.
    # show=True: Despliega una ventana emergente para mostrar el video con las detecciones en vivo.
    # conf=0.7: Establece un umbral de confianza del 70% para filtrar y evitar falsos positivos.
    # Nota: Si la ventana de la cámara no se inicializa, se sugiere cambiar "0" por el número entero 0, o añadir el parámetro stream=True.
    model.predict(source="0", show=True, conf=0.7)

    # Estructura estándar para asegurar que la ejecución directa del script se realice correctamente.
    if __name__ == '__main__':
        main()