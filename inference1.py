import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks


from timm.models import create_model
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from models.MIL_VT import*
import utils
from ultralytics import YOLO
from utils import *
import cv2
img_size = 512
C,H,W = 3,img_size,img_size
weights_path = "./weights/"
feature_name = ['ANRS','ANRI','RNFLDS',
 'RNFLDI',
 'BCLVS',
 'BCLVI',
 'NVT',
 'DH',
 'LD',
 'LC']
#threshold_list = [0.13,0.03,0.39,0.98,0.99,0.99,0.99,0.99,0.68,0.55] [0.13,0.08,0.39,0.99,0.5,0.5,0.5,0.5,0.5,0.5]
#threshold_list = [0.13,0.03,0.39,0.98,0.99,0.99,0.99,0.99,0.68,0.55]
device = torch.device("cuda:0")

#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#model path

detection_model_weight = weights_path + "/optic_disk_detection.pt"
#bi_cls_PATH = weights_path + "binary_cls.pth.tar"
cls_center_model_weight = weights_path + "/cls_center.pt" 
cls_disc_model_weight =  weights_path + "/cls_disc.pt" 


model_path_list = []
for idx, f in enumerate(feature_name):
    MODEL_PATH = weights_path + str(idx) + f + '.pt'
    model_path_list.append(MODEL_PATH)

mil_list = [7,9]


for idx, f in enumerate(feature_name):
    if idx in mil_list:
      MODEL_PATH = weights_path + str(idx) + f + ".pth.tar"
      model_path_list[idx] = MODEL_PATH
#print(model_path_list)
#transforms
transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  #[0.30, 0.19, 0.11], [0.25, 0.16, 0.10]#[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    ])
     
def run():
    #_show_torch_cuda_info()
    #load model
    #bi_cls_model1 = loadmodel(bi_cls_PATH1)
    #bi_cls_model2 = loadmodel(bi_cls_PATH2)
    detection_model = YOLO(detection_model_weight)
    cls_center_model = YOLO(cls_center_model_weight)
    cls_disc_model = YOLO(cls_disc_model_weight)
    multi_cls_models = []
    for i in range(10):
      if i not in mil_list:
        model = YOLO(model_path_list[i])
      else:
        model = loadmodel(model_path_list[i])
      multi_cls_models.append(model)

    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant
        ...

        print(f"Running inference on {jpg_image_file_name}")  

        Img = Image.open(jpg_image_file_name)
        #detect and crop around optic disk
        cropped_image1, detected = optic_detection3(model=detection_model, device='0', img_path=jpg_image_file_name) #optic_detection3(model=detection_model, device='0', img_path=jpg_image_file_name) #optic_detection(model=detection_model, device='0', filepath=jpg_image_file_name, Img=Img)        
        #cropped_image1=CLAHE_RGB(cropped_image1, clahe) 
        #cropped_image1 = cv2.cvtColor(cropped_image1, cv2.COLOR_BGR2RGB)
        #cropped_image1 = Image.fromarray(cropped_image1, 'RGB')
        
        
        #inputs1 = transform_test(cropped_image1)
        #inputs1 = torch.reshape(inputs1, (1,C,H,W))
        #inputs1 = inputs1.to(device)
        #binary cls on ROI
        #if detected:
            #bi_cls_model1.eval()
            #bi_cls_model = bi_cls_model1.to(device)
        #else:
           # bi_cls_model2.eval()
            #bi_cls_model = bi_cls_model2.to(device)
        #bi_cls_model.eval()
        #bi_cls_model = bi_cls_model.to(device)
        
        #outputs_class = bi_cls_model(inputs1)
       #is_referable_glaucoma_likelihood = utils.softmax(outputs_class.data.cpu().numpy())
        
        #is_referable_glaucoma_likelihood = binary_prediction(bi_cls_model,inputs)
        
        if detected:
            cls_model=cls_disc_model
        else:
            cls_model=cls_center_model
        
        
        cropped_image = crop_center(jpg_image_file_name)
        
        
        results = cls_model.predict(cropped_image1,device = '0',imgsz=512)
        result=results[0]
        is_referable_glaucoma_likelihood = result.probs.data.cpu().numpy()[1]
        is_referable_glaucoma_likelihood = is_referable_glaucoma_likelihood.astype(numpy.float64)
        
        #if is_referable_glaucoma_likelihood+0.1<=1:
          #is_referable_glaucoma_likelihood = is_referable_glaucoma_likelihood + 0.1
        
        
          
        is_referable_glaucoma = bool(is_referable_glaucoma_likelihood > 0.5)
        
        #feature 
        inputs = transform_test(Img)
        inputs = torch.reshape(inputs, (1,C,H,W))
        inputs = inputs.to(device)

        #cropped_image = optic_detection2(model=detection_model, device='0', filepath=jpg_image_file_name, Img=Img) #optic_detection:0.1510
        #inputs = transform_test(cropped_image)
        #inputs = torch.reshape(inputs, (1,C,H,W))
        #inputs = inputs.to(device)
        
        if True:
        
            feature_pred = feature_prediction(multi_cls_models,cropped_image,inputs)

            features = {}
            for idx, (k, v) in enumerate(DEFAULT_GLAUCOMATOUS_FEATURES.items()):
              features[k] =  bool(feature_pred[idx]>= 0.5)#threshold_list[idx])  
            
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


def loadmodel(MODEL_PATH):

    model = create_model(model_name='MIL_VT_small_patch16_512',
            pretrained=False,
            num_classes=2,
            drop_rate=0,
            drop_path_rate=0.1,
            drop_block_rate=None
            )
    checkpoint0 = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint0['state_dict'])
    

    return model

def binary_prediction(model,inputs):

    model.eval()
    bi_cls_model = model.to(device)
    inputs = inputs.to(device)
    outputs_class = model(inputs)

    is_referable_glaucoma_likelihood = utils.softmax(outputs_class[0].data.cpu().numpy()) 
    
    is_referable_glaucoma_likelihood = is_referable_glaucoma_likelihood[1]
    is_referable_glaucoma_likelihood = is_referable_glaucoma_likelihood.astype(numpy.float64) 
    
    return is_referable_glaucoma_likelihood
    
def feature_prediction(models,cropped_image,inputs):
    predictions_class=[]
    for idx, model in enumerate(models):
         if idx not in  mil_list:  
            results = model.predict(cropped_image,device = '0',imgsz=640)
            result=results[0]
            preds = result.probs.data.cpu().numpy()[1]
         else:   
            model.eval()
            model = model.to(device)
            outputs_class = model(inputs)
            outputs_class = utils.softmax(outputs_class.data.cpu().numpy())            
            outputs_class = numpy.asarray(outputs_class)
            preds =numpy.argmax(outputs_class)
            print(idx)
         predictions_class.append(preds)

    return predictions_class


if __name__ == "__main__":
    raise SystemExit(run())
