from ultralytics import YOLO

def main():
    
    model = YOLO("yolo11n.pt") 

    
    results = model.train(
        
        data="Objec_detection_Yolo.v2-visionv2.yolov11/data.yaml", 
        epochs=100,                
        imgsz=640,                
        batch=8,                   
        name="modelo_entrenado"    
    )

    print("¡Entrenamiento completado!")

if __name__ == '__main__':
    main()