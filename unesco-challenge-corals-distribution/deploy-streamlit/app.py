# Install and import dependencies
import streamlit as st
from streamlit_extras.let_it_rain import rain
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# Web page tab configuration
st.set_page_config(page_title='Coral Reefs Habitat Prediction',
                   page_icon='./unesco-challenge-corals-distribution/deploy-streamlit/images/coral-reef-logo-web-page.png',
                   layout='wide',
                   initial_sidebar_state='expanded')

st.image('./unesco-challenge-corals-distribution/deploy-streamlit/images/blue-future-header.png')

# Sidebar with model version and choice of data input type
with st.sidebar:
    st.subheader('Coral Reefs Worldwide Distribution')

    database = st.radio('Source of input data:', ('Predictor','CSV'))

    if database == 'CSV':
        st.info("Use the button below to upload the CSV file directly from the GitHub repository.")
        file_git = st.button("Upload file from GitHub repository")
        if file_git:
            Xtest = pd.read_csv('https://raw.githubusercontent.com/guilhermegarcia-ai/ml-classification-models/main/unesco-challenge-corals-distribution/deploy-streamlit/validation-dataset/Xtest.csv', sep=',')
            st.session_state['Xtest'] = Xtest

if database == 'Predictor':
    # Header with application description
    st.title('Coral Reefs Habitat Predictor')
    
    # Main tabs
    tab1, tab2 = st.tabs(["Forms", "Decision-making process"])
    with tab1:
        with st.form("features_form"):
            salinity = st.slider(
                    "What's the Water Salinity?", min_value=5.5, max_value=40.0, value=22.8, step=0.01, format="%.2f")
            jan_temp = st.slider(
                    "What's the temperature in January?", min_value=-1.9, max_value=35.0, value=16.6, step=0.01, format="%.2f")
            jun_temp = st.slider(
                    "What's the temperature in June?", min_value=-1.9, max_value=35.0, value=16.6, step=0.01, format="%.2f")
            area = st.number_input(
                    'Area of sea (square km)', min_value=11500, max_value=5695000, value=1006532, step=1000)
            latitude = st.number_input(
                    'Latitude', min_value=-75, max_value=77, value=17, step=1)
            longitude = st.number_input(
                    'Longitude', min_value=-178, max_value=163, value=76, step=1)
            type_of_sea = st.selectbox(
                    "Type of sea",
                    ('Enclosed landlocked', 'Partly enclosed', 'Marginal', 'No land boundaries'))
            silt_sulfide = st.selectbox(
                    "Presence of silt/sulfide in water?",
                    ('Yes', 'No'))        
            threshold = st.slider(
                    'Choose the Threshold', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
                
            submitted = st.form_submit_button("Predict")

        # Creating a DataFrame with the values entered by user
        user_data = pd.DataFrame({
                                'salinity': [salinity],
                                'January_temp': [jan_temp],
                                'June_temp': [jun_temp],
                                'area': [area],
                                'latitude': [latitude],
                                'longitude': [longitude],
                                'type of sea': [type_of_sea],
                                'silt/sulfide': [silt_sulfide]
                    })

        # Creating dummies to categorical values entered by user
        user_data.replace({'type of sea':{'Enclosed landlocked':1,
                                        'Partly enclosed':2,
                                        'Marginal':3,
                                        'No land boundaries':4},
                            'silt/sulfide':{'Yes':1,
                                            'No':0}},
                            inplace=True)
                    
        # Running prediction model with values entered by user
        if submitted:
            mdl_rf = pickle.load(open('../../unesco-challenge-corals-distribution/deploy-streamlit/models/model_rf.pkl','rb'))
            ypred = mdl_rf.predict(user_data)
            prediction_proba = mdl_rf.predict_proba(user_data)[:, 1]
            prediction = (prediction_proba > threshold).astype(int)
            st.subheader('Prediction result')
                        
            if prediction == 1:
                st.success(':white_check_mark: Based on the provided data, there is presence of corals in the given sea.')
                rain(emoji='🐠', font_size=54, falling_speed=6, animation_length=1)
            else:
                st.error(':x: Based on the provided data, there is no presence of corals in the given sea.')
                rain(emoji='🚨', font_size=54, falling_speed=6, animation_length=1)
            st.info(':memo: To understand the decision-making process of the algorithm, please see the "Decision-making process" tab.')

    with tab2:
        if submitted != True:
            st.warning('To understand the decision-making process of the algorithm, please fill out the form.')
        else:
            # Data provided
            st.subheader('Data provided')
            user_data['corals'] = prediction

            user_data.replace({'type of sea':{1:'Enclosed landlocked',
                                              2: 'Partly enclosed',
                                              3: 'Marginal',
                                              4: 'No land boundaries'},
                               'silt/sulfide':{1:'Yes',
                                              0: 'No'},
                               'corals':{1:'presence',
                                         0:'absence'}},
                                inplace=True)

            def color_pred(prediction):
                color = 'lightgreen' if prediction == 'presence' else 'lightcoral'
                return f'background-color: {color}'

            st.dataframe(user_data.style.applymap(color_pred, subset=['corals']), width=1200)
            
            # Feature importance
            st.subheader('Feature importance')

            def impPlot(imp, name):
                imp_sorted = imp.sort_values(ascending=True)
                figure = px.bar(imp_sorted,
                                x=imp_sorted.values,
                                y=imp_sorted.keys(), labels = {'x':'Importance Value', 'index':'Columns'},
                                text=np.round(imp_sorted.values, 2),
                                title=name,
                                width=1000, height=600)
                figure.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                })
                st.plotly_chart(figure)
            
            def randomForest(x, y):
                feat_importances = pd.Series(mdl_rf.feature_importances_, index=x.columns).sort_values(ascending=False)
                impPlot(feat_importances, 'Random Forest Classifier')
                st.write('\n')
            
            randomForest(user_data.drop(columns='corals'), prediction)

elif database == 'CSV':
    # Header with application description
    st.title('Coral Reefs Worldwide Distribution')
    if 'Xtest' in st.session_state:
        Xtest = st.session_state['Xtest']
        
        mdl_rf = pickle.load(open('../../unesco-challenge-corals-distribution/deploy-streamlit/models/model_rf.pkl','rb'))
        ypred = mdl_rf.predict(Xtest)
        prediction_proba = mdl_rf.predict_proba(Xtest)[:, 1]
        
        # Raw dataset
        st.subheader('Raw Dataset')
        num_rows_raw = st.slider('Choose how many rows for raw dataframe:',
                                        min_value = 1, 
                                        max_value = Xtest.shape[0], 
                                        step = 10,
                                        value = 5)
        st.dataframe(Xtest.head(num_rows_raw), width=1200)

        # Predictions dataset
        st.subheader('Dataset with predictions')

        Xtest_pred = Xtest.copy()
        Xtest_pred['corals'] = ypred

        Xtest_pred.replace({'type of sea':{1:'Enclosed landlocked',
                                              2: 'Partly enclosed',
                                              3: 'Marginal',
                                              4: 'No land boundaries'},
                               'silt/sulfide':{1:'Yes',
                                              0: 'No'},
                               'corals':{1:'presence',
                                         0:'absence'}},
                                inplace=True)
        
        def color_pred(prediction):
                color = 'lightgreen' if prediction == 'presence' else 'lightcoral'
                return f'background-color: {color}'

        num_rows_pred = st.slider('Choose how many rows for predictions dataframe:',
                                        min_value = 1, 
                                        max_value = Xtest_pred.shape[0],
                                        step = 4,
                                        value = 5)
        st.dataframe(Xtest_pred.head(num_rows_pred).style.applymap(color_pred, subset=['corals']), width=1200)
        
        num_ypred_presence = Xtest_pred.head(num_rows_pred)[Xtest_pred["corals"].eq("presence")]["corals"].count()
        num_ypred_absence = Xtest_pred.head(num_rows_pred)[Xtest_pred["corals"].eq("absence")]["corals"].count()
        
        st.markdown(f'Total number of rows filtered: {Xtest_pred.head(num_rows_pred).shape[0]}')
        st.markdown(f'Amount of presence of corals: {num_ypred_presence} ({num_ypred_presence/Xtest_pred.head(num_rows_pred)["corals"].count()*100:.2f}%)')
        st.markdown(f'Amount of absence of corals: {num_ypred_absence} ({num_ypred_absence/Xtest_pred.head(num_rows_pred)["corals"].count()*100:.2f}%)')

        csv = Xtest_pred.head(num_rows_pred).to_csv(sep = ';', decimal = ',', index = False)
        st.download_button(label = 'Download dataset (.csv)',
                                data = csv,
                                file_name = 'Corals_predictions.csv',
                                mime = 'text/csv')

        # Feature importance
        st.subheader('Feature importance')

        def impPlot(imp, name):
            imp_sorted = imp.sort_values(ascending=True)
            figure = px.bar(imp_sorted,
                            x=imp_sorted.values,
                            y=imp_sorted.keys(), labels = {'x':'Importance Value', 'index':'Columns'},
                            text=np.round(imp_sorted.values, 2),
                            title=name,
                            width=1000, height=600)
            figure.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
            st.plotly_chart(figure)
            
        def randomForest(x, y):
            feat_importances = pd.Series(mdl_rf.feature_importances_, index=x.columns).sort_values(ascending=False)
            impPlot(feat_importances, 'Random Forest Classifier')
            st.write('\n')
            
        randomForest(Xtest_pred.drop(columns='corals'), ypred)
        
    else:
        st.warning('To view this page, please upload a CSV file.')