#This code is based on the example provided at https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_pgan/
import torch
import streamlit as st
import matplotlib.pyplot as plt
import torchvision

st.title("Demo of Progressive GAN using PyTorch and Streamlit")
use_gpu = True if torch.cuda.is_available() else False

#@st.cache(allow_output_mutation=True)
def load_model():
    # trained on high-quality celebrity faces "celebA" dataset
    # this model outputs 512 x 512 pixel images
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                        'PGAN', model_name='celebAHQ-512',
                        pretrained=True, useGPU=use_gpu)
    # this model outputs 256 x 256 pixel images
    # model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
    #                        'PGAN', model_name='celebAHQ-256',
    #                        pretrained=True, useGPU=use_gpu)
    return model

model = load_model()
#@st.cache(allow_output_mutation=True)
def gen_images():
    num_images = 4
    noise, _ = model.buildNoiseData(num_images)
    with torch.no_grad():
        generated_images = model.test(noise)
    return generated_images

generated_images = gen_images()
#st.image(generated_images)

# let's plot these images using torchvision and matplotlib

grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
st.image(grid.permute(1, 2, 0).cpu().numpy())
#plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
#plt.show()