from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/modelo_entrenado-2/weights/best.pt")

    
    print("Encendiendo cámara web... Presiona la tecla 'q' en la ventana de la cámara para salir.")
    
    model.predict(source="0", show=True, conf=0.7)

if __name__ == '__main__':
    main()