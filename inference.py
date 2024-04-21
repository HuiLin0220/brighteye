import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks
import torch
import utils
from ultralytics import YOLO
from utils import *


gpu_id = '0'
feature_name = ['ANRS','ANRI','RNFLDS',
 'RNFLDI',
 'BCLVS',
 'BCLVI',
 'NVT',
 'DH',
 'LD',
 'LC']
 
weights_path = "./weights/"
#model path
cls_center_model_weight = weights_path + "/cls_center.pt" 

model_path_list = []
for idx, f in enumerate(feature_name):
    MODEL_PATH = weights_path + str(idx) + f + '.pt'
    model_path_list.append(MODEL_PATH)
#print(model_path_list)



def run():
    #_show_torch_cuda_info()
    #load model
    cls_model = YOLO(cls_center_model_weight)

    multi_cls_models = []
    for i in range(10):
      model = YOLO(model_path_list[i])
      multi_cls_models.append(model)


    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant
        ...

        print(f"Running inference on {jpg_image_file_name}")  

        cropped_image = crop_center(jpg_image_file_name)
        
        
        results = cls_model.predict(cropped_image,device = gpu_id,imgsz=512)
        result=results[0]
        is_referable_glaucoma_likelihood = result.probs.data.cpu().numpy()[1]
        is_referable_glaucoma_likelihood = is_referable_glaucoma_likelihood.astype(numpy.float64)
        
          
        is_referable_glaucoma = bool(is_referable_glaucoma_likelihood > 0.5)
        
        
        if True:
        
            feature_pred = feature_prediction(multi_cls_models,cropped_image,gpu_id)

            features = {}
            for idx, (k, v) in enumerate(DEFAULT_GLAUCOMATOUS_FEATURES.items()):
              features[k] =  bool(feature_pred[idx]>= 0.5)
            
        else:
            features = None
        
        ...

        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features,
        )
    return 0


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    available = torch.cuda.is_available()
    print(f"Torch CUDA is available: {available}")
    if available:
        device = torch.cuda.device_count()
        print(f"\tnumber of devices: {device}")
        current_device = torch.cuda.current_device()
        print(f"\tcurrent device: {current_device}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

    
def feature_prediction(models,cropped_image,gpu_id):
    predictions_class=[]
    for idx, model in enumerate(models):
       
        results = model.predict(cropped_image,device = gpu_id,imgsz=640)
        result=results[0]
        preds = result.probs.data.cpu().numpy()[1]
         
        predictions_class.append(preds)

    return predictions_class


if __name__ == "__main__":
    raise SystemExit(run())
