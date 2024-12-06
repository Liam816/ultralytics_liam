from ultralytics import settings
from ultralytics import YOLO
import cv2
from PIL import Image
import torch


def test_prediction():
    print(f"settings raw: {settings}")
    settings.update(
        {
            "runs_dir": "runs/test_yolo11n_bus", 
            "datasets_dir": "",
            "weights_dir": "",
            "tensorboard": False
        }
    )
    # settings.reset()
    print(f"settings overridden: {settings}")
    # exit()

    # Create a new YOLO model from scratch
    # model = YOLO("yolo11n.yaml")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(model="yolo11n.pt", task=None, verbose=False)
    print(f"LIAM model.device: {model.device}")
    print(f"LIAM torch.cuda.is_available(): {torch.cuda.is_available()}")
    # exit()

    # # Train the model using the 'coco8.yaml' dataset for 3 epochs
    # results = model.train(data="coco8.yaml", epochs=3)

    # # Evaluate the model's performance on the validation set
    # results = model.val()

    # # Perform object detection on an image using the model
    # results = model("https://ultralytics.com/images/bus.jpg")

    # # Export the model to ONNX format
    # success = model.export(format="onnx")

    # # from PIL
    # im1 = Image.open("bus.jpg")
    # results = model.predict(source=im1, save=True)  # save plotted images

    # from ndarray
    im2 = cv2.imread("bus.jpg")
    results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

    # # from list of PIL/ndarray
    # results = model.predict(source=[im1, im2])


def test_train():
    print(f"settings raw: {settings}")
    settings.update(
        {
            "runs_dir": "runs/test_train_dec", 
            "datasets_dir": "C:/Users/ping.he/Desktop/liam/dataset/dec_hangzhou_dual_view_merged",
            "weights_dir": "",
            "tensorboard": False
        }
    )
    print(f"settings overridden: {settings}")
    # exit()

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(model="yolov8n.yaml", task=None, verbose=False)

    results = model.train(data="dec_hangzhou_dual_view.yaml", epochs=1, batch=1, imgsz=640, device=[0])


if __name__ == "__main__":
    # test_prediction()
    test_train()

