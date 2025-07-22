import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import joblib
#load model

model=joblib.load('svc_model.pt')

#getLabel
def get_label(img):
    #convert into gray scale
    img_g=img.convert("L")
    #resize into 100,100
    img_res=img_g.resize((100,100))
    #convert into numpy array
    img_a=np.array(img_res).flatten()
    #convert into df and Transport
    img_df=pd.DataFrame(img_a).T
    #predict with the model
    pre=model.predict(img_df)
    #return the value
    if pre=='Fractured':
        return 'Fractured'
    else:
        return 'Non-Fractured'
    return pre


#title
st.title("Bone Fracture Prediction")
st.header("A computer vision project")
file=st.file_uploader("Upload your file",type='jpg')

try:
    if file is not None:
        #read image
        img=Image.open(file)
        #show image
        st.image(img,"The Uploaded Image")
        prediction=get_label(img)
        st.write(f"The bone is :{prediction}")
    else:
        st.write("Empty file cannot be read")
        
except Exception as e:
    st.write(f"{e} occured")
    
finally:
    st.write("Thank you for using our service")