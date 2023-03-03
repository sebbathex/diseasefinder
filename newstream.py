import streamlit as st

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from collections import Counter
#import s3fs




def create_df (): 
    daten = pd.read_csv('corneal_dystrophies - corneal_dystrophies _data Kopie(2).csv')


    daten = daten.fillna('unknown')
    daten = daten.replace('y', 'yes')
    daten = daten.replace('n', 'no')
            #daten.head(5)


    features = list(daten.head(0))
    features = features[4::]


            #Modifiziere Eingabe

    for i in features:
        daten[i] = daten[i].replace('yes', i)
        daten[i] = daten[i].replace('no', f'not_{i}')
        daten[i] = daten[i].replace('unknown', f'unknown_{i}')


            
    y = daten["Name"]
    target_array = y.values
    target_array


    namen = list(daten_encoded.head(0))

    namen = namen[4::]



    X = daten_encoded[namen].values
    y = target_array










def write_navigation_bar():
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Go to", ["Cornea rare disease finder", "Study", "About"])
    if page == "Cornea rare disease finder":
        write_main_page()
    
    elif page == "About":
        write_page_2()



def write_main_page():
    
    
   
    
    st.header('Hallo ')

    yes_no_unknown = {"I'm not sure!": 'unknown', 'Yes': 'Yes', 'No': 'No'}

    myvariabledict = {"I'm not sure.": "unknown", "0 -10 years" :"0", "11 - 20 years": "1", "21 - 30 years": "2", "31 years and older": "3"}

    myvariabledict6 = {"Unknown (family history unclear/ only affected family member)": 'unknown', 'AD': 'AD', 'AR': 'AR', 'No inheritance': 'NO', 'X-linked': 'X'}

    myvariabledict7 = {"I'm not sure!": 'unknown', 'Unilateral.': 'Yes', 'Bilateral.': 'No'}

    myvariabledict3 = {"I'm not sure!": 'unknown', 'Endothelium': 'Endothelium', 'Epithelium': 'Epithelium', 'Stroma': 'Stroma', 'Stroma/Endothelium': 'Endo/Stroma'}


    #### Fragebogen #####


    myvariables3 = st.selectbox("Primarily affected layer?", list(myvariabledict3.keys()))
    myvariable3 = myvariabledict3[myvariables3]

    if myvariable3 == 'Endothelium':

        st.sucess ("So far possible solutions for your inputs are: {}")

        myvariables = st.selectbox("Age of first time clinical appearance?",  list(myvariabledict.keys()))
        myvariable = myvariabledict[myvariables]

    




    
                    

    

    
    #if myvariable and myvariable2:
    #    st.success("The most likely disease given your input is : {} and {}".format(first_place, second_place))


def train (): 
        #Lade csv
    daten = pd.read_csv('corneal_dystrophies - corneal_dystrophies _data Kopie(2).csv')


    daten = daten.fillna('unknown')
    daten = daten.replace('y', 'yes')
    daten = daten.replace('n', 'no')
        #daten.head(5)


    features = list(daten.head(0))
    features = features[4::]


        #Modifiziere Eingabe

    for i in features:
        daten[i] = daten[i].replace('yes', i)
        daten[i] = daten[i].replace('no', f'not_{i}')
        daten[i] = daten[i].replace('unknown', f'unknown_{i}')


        # One-Hot encoding aller features

    daten_encoded = pd.get_dummies(data=daten, columns=['decade of diagnosis','recurrent erosions', 'primarily affected layer','corneal thinning', 'non progressive','inheritance', 'may be unilateral', 'microcysts', 'epithelial thickening',
                                        'stroma: rings / stars', 'stroma: central snowflakes / lines', 'stroma: cloudy appearance', 'stroma: arcus', 'stroma: honeycomb', 'stroma: confluent geographic', 'stroma: pre decemetal haze', 'stromal crystals',
                                        'diffuse stromal haze ', 'deep stromal diffuse deposits', 'irregular posterior corneal surface', 'beaten metal appearance of corneal surface', 'corneal steepening', 'tiny dots on the posterior corneal surface'])
    #daten_encoded

    daten_encoded.to_csv('daten_encoded.csv')

    # ergebnisvektor
    y = daten["Name"]
    target_array = y.values
    #target_array


    namen = list(daten_encoded.head(0))

    namen = namen[4::]



    X = daten_encoded[namen].values
    y = target_array

        #Decision Tree
        #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

   

    return ()        

      

    

def write_page_2():
    st.header("About")
    st.write("This is page 2.")


def transform_ergebnisse(liste):
            prediction_list = []
        #decade of diagnosis
            if liste[0] == "0":
                prediction_list.append([1,0,0,0,0])
            elif liste[0] == '1':
                prediction_list.append([0,1,0,0,0])
            elif liste[0] == '2':
                prediction_list.append([0,0,1,0,0])
            elif liste[0] == '3':
                prediction_list.append([0,0,0,1,0])
            elif liste[0] == 'unknown':
                prediction_list.append([0,0,0,0,1])

            #frage_2 = input('keine Wiederkehrenden erosionen, wiederkehrende erosionen, unbekannt')
            if liste[1] == "No":
                prediction_list.append([1,0,0])
            elif liste[1] == 'Yes':
                prediction_list.append([0,1,0])
            elif liste[1] == 'unknown':
                prediction_list.append([0,0,1])

            #frage_3 = input('endo, epi, stro, stro, endo, unbekannt')
            if liste[2] == "Endothelium":
                prediction_list.append([1,0,0,0,0])
            elif liste[2] == 'Epithelium':
                prediction_list.append([0,1,0,0,0])
            elif liste[2] == 'Stroma':
                prediction_list.append([0,0,1,0,0])
            elif liste[2] == 'Stroma/Endo':
                prediction_list.append([0,0,0,1,0])
            elif liste[2] == 'unknown':
                prediction_list.append([0,0,0,0,1])

            #frage_4 = input('keine Korneaausdünnung, unbekannt')

            if liste[3] == "Yes":
                prediction_list.append([1,0])
            elif liste[3] == "No":
                prediction_list.append([0,1])
            elif liste[3] == 'unknown':
                prediction_list.append([0,0])

            #frage_5 = input('nicht progressiv, unbekannt ')
            if liste[4] == "No":
                prediction_list.append([1,0])
            elif liste[4] == "Yes":
                prediction_list.append([0,1])
            elif liste[4] == 'unknown':
                prediction_list.append([0,0])

            #frage_6 = input('Vererbung:  AD,  AR,  NO,  X,  XD, XR')
            if liste[5] == "AD":
                prediction_list.append([1,0,0,0])
            elif liste[5] == 'AR':
                prediction_list.append([0,1,0,0])
            elif liste[5] == 'NO':
                prediction_list.append([0,0,1,0])
            elif liste[5] == 'X':
                prediction_list.append([0,0,0,1])
            elif liste[5] == 'unknown':
                prediction_list.append([0,0,0,0])

            #frage_7 = input('unilateral, unbekannt')
            if liste[6] == "Yes":
                prediction_list.append([1,0])
            elif liste[6] == 'No':
                prediction_list.append([0,1])
            elif liste[6] == 'unknown':
                prediction_list.append([0,0])

            #frage_8 = input('mikrozysten, unbekannt')
            if liste[7] == "Yes":
                prediction_list.append([1,0])
            elif liste[7] == 'No':
                prediction_list.append([0,1])
            elif liste[7] == 'unknown':
                prediction_list.append([0,0])

            #frage_9 = input('Epithelverdickung, unbekannt')
            if liste[8] == "Yes":
                prediction_list.append([1,0])
            elif liste[8] == 'No':
                prediction_list.append([0,1])
            elif liste[8] == 'unknown':
                prediction_list.append([0,0])

            #frage_95 = input('stroma: rings / stars   , unbekannt ')
            if liste[9] == "Yes":
                prediction_list.append([0,1])
            elif liste[9] == 'No':
                prediction_list.append([1,0])
            elif liste[9] == 'unknown':
                prediction_list.append([0,0])


            #frage_10 = input('stroma: central snowflakes,  unbekannt')
            if liste[10] == "Yes":
                prediction_list.append([0,1])
            elif liste[10] == 'No':
                prediction_list.append([1,0])
            elif liste[10] == 'unknown':
                prediction_list.append([0,0])

            #frage_11 = input('stroma: cloudy appearance, unbekannt')
            if liste[11] == "Yes":
                prediction_list.append([0,1])
            elif liste[11] == 'No':
                prediction_list.append([1,0])
            elif liste[11] == 'unknown':
                prediction_list.append([0,0])


            #frage_12 = input('stroma: arcus_stroma, unbekannt ')
            if liste[12] == "Yes":
                prediction_list.append([0,1])
            elif liste[12] == 'No':
                prediction_list.append([1,0])
            elif liste[12] == 'unknown':
                prediction_list.append([0,0])

            #frage_13 = input('stroma: honeycomb, unbekannt')
            if liste[13] == "Yes":
                prediction_list.append([0,1])
            elif liste[13] == 'No':
                prediction_list.append([1,0])
            elif liste[13] == 'unknown':
                prediction_list.append([0,0])

            #frage_14 = input('stroma: confluent geographic, unbekannt')
            if liste[14] == "Yes":
                prediction_list.append([0,1])
            elif liste[14] == 'No':
                prediction_list.append([1,0])
            elif liste[14] == 'unknown':
                prediction_list.append([0,0])

            #frage_15 = input('stroma: pre decemetal haze, unbekannnt ')
            if liste[15] == "Yes":
                prediction_list.append([0,1])
            elif liste[15] == 'No':
                prediction_list.append([1,0])
            elif liste[15] == 'unknown':
                prediction_list.append([0,0])

            #frage_16 = stromal crystals
            if liste[16] == "No":
                prediction_list.append([0,1])
            elif liste[16] == 'Yes':
                prediction_list.append([1,0])
            elif liste[16] == 'unknown':
                prediction_list.append([0,0])

            #frage_17 = diffuse stromal haze
            if liste[17] == "No":
                prediction_list.append([0,1])
            elif liste[17] == 'Yes':
                prediction_list.append([1,0])
            elif liste[17] == 'unknown':
                prediction_list.append([0,0])

            #frage_18 = deep stromal dffuse deposits
            if liste[18] == "No":
                prediction_list.append([0,1])
            elif liste[18] == 'Yes':
                prediction_list.append([1,0])
            elif liste[18] == 'unknown':
                prediction_list.append([0,0])

            #frage_19 = irregular posterior surface
            if liste[19] == "No":
                prediction_list.append([0,1])
            elif liste[19] == 'Yes':
                prediction_list.append([1,0])
            elif liste[19] == 'unknown':
                prediction_list.append([0,0])

            #frage_20 = beaten metal appearance
            if liste[20] == "No":
                prediction_list.append([0,1])
            elif liste[20] == 'Yes':
                prediction_list.append([1,0])
            elif liste[20] == 'unknown':
                prediction_list.append([0,0])

            #frage_21 = corneael steepening
            if liste[21] == "No":
                prediction_list.append([0,1])
            elif liste[21] == 'Yes':
                prediction_list.append([1,0])
            elif liste[21] == 'unknown':
                prediction_list.append([0,0])

            #frage_22 = tiny dots on the posterior surface
            if liste[22] == "No":
                prediction_list.append([0,1])
            elif liste[22] == 'Yes':
                prediction_list.append([1,0])
            elif liste[22] == 'unknown':
                prediction_list.append([0,0])

            return prediction_list    




                 

if __name__ == '__main__':
    write_navigation_bar()





