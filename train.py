from ultralytics import YOLO

def main():
    
    model = YOLO("yolo11n.pt") 

    
    results = model.train(
        
        data="Objec_detection_Yolo.v3-visionv3.yolov11/data.yaml", 
        epochs=300,                
        imgsz=640,                
        batch=32,                   
        name="modelo_entrenado"    
    )

    print("¡Entrenamiento completado!")

if __name__ == '__main__':
    main()