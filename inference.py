import random

import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks


from timm.models import create_model
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from models.MIL_VT import*
import utils


img_size = 512
C,H,W = 3,img_size,img_size
weights_path = "./weights/"
features = ['ANRS','ANRI','RNFLDS',
 'RNFLDI',
 'BCLVS',
 'BCLVI',
 'NVT',
 'DH',
 'LD',
 'LC']
device = torch.device("cuda:0")



bi_cls_PATH = weights_path + "binary_cls.pth.tar"
model_path_list = []
for idx, f in enumerate(features):
    MODEL_PATH = weights_path + str(idx) + f + ".pth.tar"
    model_path_list.append(MODEL_PATH)
print(model_path_list)

transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

     
def run():
    _show_torch_cuda_info()
    bi_cls_model = loadmodel(bi_cls_PATH)
    
    multi_cls_models = []
    for i in range(10):
      model = loadmodel(model_path_list[i])
      multi_cls_models.append(model)
    
     

    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant
        ...

        print(f"Running inference on {jpg_image_file_name}")

        
        Img = Image.open(jpg_image_file_name)
        inputs = transform_test(Img)
        inputs = torch.reshape(inputs, (1,C,H,W))
        inputs = inputs.to(device)
        
      

        is_referable_glaucoma_likelihood = binary_prediction(bi_cls_model,inputs)  
        is_referable_glaucoma = bool(is_referable_glaucoma_likelihood > 0.5)
        #is_referable_glaucoma = True
        
        
        if is_referable_glaucoma:
        
            feature_pred = feature_prediction(multi_cls_models,inputs)

            features = {}
            for idx, (k, v) in enumerate(DEFAULT_GLAUCOMATOUS_FEATURES.items()):
              features[k] =  bool(feature_pred[idx]> 0.5)  
            
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
    print(MODEL_PATH)
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
    is_referable_glaucoma_likelihood = is_referable_glaucoma_likelihood.astype(numpy.float) 
    
    return is_referable_glaucoma_likelihood
    
def feature_prediction(models,inputs):
    predictions_class=[]
    for idx, model in enumerate(models):
            
            model.eval()
            model = model.to(device)
            outputs_class = model(inputs)
            outputs_class = utils.softmax(outputs_class.data.cpu().numpy())
            #print(outputs_class)
            outputs_class = numpy.asarray(outputs_class)
            preds = numpy.argmax(outputs_class)
            predictions_class.append(preds)
            print(preds)
    return predictions_class


if __name__ == "__main__":
    raise SystemExit(run())
