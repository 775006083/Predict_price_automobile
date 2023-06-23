import streamlit as st
import numpy as np
import joblib 

st.title("Prediction du prix d'une voiture en fonction de ses differents caracteristiques")
st.subheader("Application réalisé par Habib Aidara")
st.markdown("Cette application utilise le model machine learning pour predire le prix d'une voiture")


#Chargement du model 
model=joblib.load(filename="final_model.joblib")

#Définition d'une fonctin d'inference
def inference(symboling, normalized_losses, wheel_base, length, width, height, curb_weight,engine_size, bore, stroke, compression_ratio, horsepower, peak_rpm, city_mpg, highway_mpg):
    new_data = np.array([symboling, normalized_losses, wheel_base, length, width, height, curb_weight,engine_size, bore, stroke, compression_ratio, horsepower, peak_rpm, city_mpg, highway_mpg
                          ])
    pred = model.predict(new_data.reshape(1,-1))
    return pred




#Saisie de l'utilisateur pour chaque caracteristique de la voiture

symboling = st.number_input(label='symboling:', min_value=0, value=3)
normalized_losses = st.number_input('normalized_losses:',value=100)
wheel_base= st.number_input('wheel_base:',value=50)
length = st.number_input('length:',value=150)
width = st.number_input('width:',value=65)
height = st.number_input('height:',value=50)
curb_weight = st.number_input('curb_weight:',value=200)
engine_size = st.number_input('engine_size:',value=120)
bore = st.number_input('bore:',value=3.0)
stroke = st.number_input('strok:',value=3.0)
compression_ratio = st.number_input('compression_ratio:',value=9.0 )
horsepower = st.number_input('horsepower:',value=110 )
peak_rpm = st.number_input('peak_rpm:',value=5000 )
city_mpg = st.number_input('city_mpg:',value=20)
highway_mpg = st.number_input('city_mpg:',value=30 )

#Creation d'une bouton prediction qui retoune la prediction du model
if st.button("predict"):
    prediction= inference(symboling, normalized_losses, wheel_base, length, width, height, curb_weight,engine_size, bore, stroke, compression_ratio, horsepower, peak_rpm, city_mpg, highway_mpg)
    resultat = "Le prix en (dollard) de cette voiture est egale à: " + str(prediction[0])
    st.success(prediction)