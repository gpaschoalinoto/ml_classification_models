# Install and import dependencies
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import *
from sklearn.preprocessing import LabelEncoder
import altair as alt
import plotly.express as px

# Web page tab configuration
st.set_page_config(page_title='iFood Sales Prediction',
                   page_icon='././images/ifood-logo-web-page.png',
                   layout='wide',
                   initial_sidebar_state='expanded')

# Header with application description
st.title('iFood Purchase Propensity Prediction Project')

with st.expander('About the project', expanded=False):
    st.write('The objective of the project is to deploy a predictive model that will maximize profit for the next direct marketing campaign by identifying users who are most likely to accept the campaign.')
    st.write('\n')
    st.write("Developed by https://www.linkedin.com/in/guilhermegpaschoalinoto/.")
 
# Sidebar with model version and choice of data input type
with st.sidebar:
    c1, c2 = st.columns(2)
    c1.image('././images/ifood-logo-side-bar.png', width=250)
    c2.write('')
    c1.subheader('iFood Sales Predictor - LGBM v1')

    database = st.radio('Source of input data (X):', ('Online','CSV'))

    if database == 'CSV':
        st.info('Upload CSV')
        file = st.file_uploader('Select CSV file.', type='csv')

pred_done = 0

# Main tabs
tab1, tab2 = st.tabs(["Predictions", "Detailed Analysis"])

# Settings for the first tab "Predictions"
with tab1:
    if database == 'Online':
            st.title('Registration of new users for prediction. :arrow_heading_down:')

            # Form structure for filling in features
            col_personal_info_desc, col_personal_info_emoji, col_accepted_campaigns_desc, col_accepted_campaigns_emoji = st.columns(4)
            
            # Personal information
            with col_personal_info_desc:
                st.subheader('Personal information')
                age = st.number_input('Age', min_value=0)
                education = st.selectbox('Education', ['Basic', 'Graduation', '2n Cycle', 'Master', 'PhD'])
                kidhome = st.number_input('Kidhome', min_value=0, max_value=100)
                marital_status = st.selectbox('MaritalStatus', ['Single', 'Together', 'Married', 'Divorced','Widow'])

            with col_personal_info_emoji:
                st.subheader(':man-girl-boy:')
                teenhome = st.number_input('Teenhome', min_value=0, max_value=100)
                income = st.number_input('Income', min_value=0.0, format="%.2f", step=1000.0, value=50000.0)
                time_days_customer = st.number_input('TimeDaysCustomer', min_value=0.0, format="%.2f", step=1.0, value=0.0)

            #Accepted campaigns
            with col_accepted_campaigns_desc:
                st.subheader('Accepted campaigns')
                accepted_cmp1 = st.radio('AcceptedCmp1', ('Yes', 'No'))
                accepted_cmp3 = st.radio('AcceptedCmp3', ('Yes', 'No'))
                accepted_cmp4 = st.radio('AcceptedCmp4', ('Yes', 'No'))

            with col_accepted_campaigns_emoji:
                st.subheader(':chart_with_upwards_trend:')
                accepted_cmp5 = st.radio('AcceptedCmp5', ('Yes', 'No'))

            col_purchases_desc, col_purchases_emoji, col_threshold_desc, col_prediction_button_desc = st.columns(4)
            
            # Purchases
            with col_purchases_desc:
                st.subheader('Purchases')
                mnt_fish_products = st.number_input('MntFishProducts', min_value=0)
                mnt_fruits = st.number_input('MntFruits', min_value=0)
                mnt_gold_prods = st.number_input('MntGoldProds', min_value=0)
                mnt_meat_products = st.number_input('MntMeatProducts', min_value=0)
                mnt_sweet_products = st.number_input('MntSweetProducts', min_value=0)
                mnt_wines = st.number_input('MntWines', min_value=0)

            with col_purchases_emoji:
                st.subheader(':heavy_dollar_sign:')
                num_catalog_purchases = st.number_input('NumCatalogPurchases', min_value=0)
                num_deals_purchases = st.number_input('NumDealsPurchases', min_value=0)
                num_store_purchases = st.number_input('NumStorePurchases', min_value=0)
                num_web_purchases = st.number_input('NumWebPurchases', min_value=0)
                num_web_visits_month = st.number_input('NumWebVisitsMonth', min_value=0)
                recency = st.number_input('Recency', min_value=0)

            # Threshold
            with col_threshold_desc:
                st.subheader('Threshold')
                threshold = st.slider('Choose the Threshold', min_value=0.0, max_value=1.0, step=0.01, value=0.5)

            # Prediction button
            with col_prediction_button_desc:
                st.subheader('Prediction button')

                # Creating a DataFrame with the values entered by user
                user_data = pd.DataFrame({
                    'AcceptedCmp1': [accepted_cmp1],
                    'AcceptedCmp3': [accepted_cmp3],
                    'AcceptedCmp4': [accepted_cmp4],
                    'AcceptedCmp5': [accepted_cmp5],
                    'Age': [age],
                    'Education': [education],
                    'Income': [income],
                    'Kidhome': [kidhome],
                    'MntFishProducts': [mnt_fish_products],
                    'MntFruits': [mnt_fruits],
                    'MntGoldProds': [mnt_gold_prods],
                    'MntMeatProducts': [mnt_meat_products],
                    'MntSweetProducts': [mnt_sweet_products],
                    'MntWines': [mnt_wines],
                    'NumCatalogPurchases': [num_catalog_purchases],
                    'NumDealsPurchases': [num_deals_purchases],
                    'NumStorePurchases': [num_store_purchases],
                    'NumWebPurchases': [num_web_purchases],
                    'NumWebVisitsMonth': [num_web_visits_month],
                    'Recency': [recency],
                    'Teenhome': [teenhome],
                    'Time_Days_Customer': [time_days_customer]
                })

                # Creating dummies to categorical values entered by user
                marital_status_unique_values = ['Married', 'Single', 'Together', 'Divorced','Widow']

                for status in marital_status_unique_values:
                    col_name = 'Marital_Status_' + status.replace(' ', '_')
                    user_data[col_name] = int(marital_status == status) if status == marital_status else 0

                user_data.drop(columns=['Marital_Status_Divorced'], inplace=True)
                                
                # Applying label encoder to categorical values entered by user
                label_encoder = LabelEncoder()

                user_data['Education'] = label_encoder.fit_transform(user_data['Education'])
                
                # Replacing binary categorical values entered by user
                user_data.replace({'Yes': 1, 'No': 0}, inplace=True)
                
                # Running prediction model with values entered by user
                if st.button('Predict'):
                    mdl_rf = load_model('./ifood_sales_prospecting_model_lgbm')
                    ypred = predict_model(mdl_rf, data=user_data, raw_score=True)
                    prediction_proba = mdl_rf.predict_proba(user_data)[:, 1]
                    prediction = (prediction_proba > threshold).astype(int)
                    st.subheader('Prediction result')
                    
                    if prediction == 1:
                        st.success(':white_check_mark: This client is likely to purchase the product from the campaign.')
                        pred_done = 1
                    else:
                        st.error(':x: This client is not likely to purchase the product from the campaign.')
                        pred_done = 1

    elif database == 'CSV':
        if file:
            Xtest = pd.read_csv(file)
            mdl_lgbm = load_model('./ifood_sales_prospecting_model_lgbm')

            # Running prediction model with csv file
            ypred = predict_model(mdl_lgbm, data = Xtest, raw_score = True)

            # Option to visualize x defined rows
            with st.expander('View loaded CSV file:', expanded = False):
                c1, _ = st.columns([2,4])
                qtd_linhas = c1.slider('View how many rows of the CSV:', 
                                        min_value = 1, 
                                        max_value = Xtest.shape[0], 
                                        step = 10,
                                        value = 5)
                st.dataframe(Xtest.head(qtd_linhas))

            # Option to iteract predictions with threshold
            with st.expander('View Predictions:', expanded = True):
                c1, _, c2, c3 = st.columns([2,.5,1,1])
                treshold = c1.slider('Threshold (cut-off point for considering prediction as True).',
                                    min_value = 0.0,
                                    max_value = 1.0,
                                    step = .1,
                                    value = .5)
                print('Teste teste teste', ypred.columns)
                qtd_true = ypred.loc[ypred['prediction_score_1'] > treshold].shape[0]

                c2.metric('Amount clients True', value = qtd_true)
                c3.metric('Amount clients False', value = len(ypred) - qtd_true)

            # Format of dataframe to export
            tipo_view = st.radio('', ('Complete', 'Only Predictions'))
            if tipo_view == 'Complete':
                df_view = ypred.copy()
            else:
                df_view = pd.DataFrame(ypred.iloc[:,-1].copy())

            # Color prediction_score_1 column according to threshold
            def color_pred(val):
                color = 'olive' if val > treshold else 'orangered'
                return f'background-color: {color}'

            st.dataframe(df_view.style.applymap(color_pred, subset = ['prediction_score_1']))

            csv = df_view.to_csv(sep = ';', decimal = ',', index = True)
            st.markdown(f'Format of the CSV file to be downloaded: {df_view.shape}')
            st.download_button(label = 'Download CSV file',
                                data = csv,
                                file_name = 'Predictions.csv',
                                mime = 'text/csv')

        else:
            st.warning('CSV file was not loaded.')

# Settings for the second tab "Detailed Analysis"
with tab2:
    if database == 'Online':
        if pred_done != 1:
            st.warning('To view this page, please register a new user for prediction.')
        else:
            # Explanation about not using SHAP for single samples in Online tab - LGBM model
            st.header("Feature Importance with SHAP")
        
            st.write('Analyzing only a single sample prediction using SHAP with an LGBM (LightGBM) algorithm can be risky for several reasons:')
            st.write('\n')
            st.write('Sampling bias: A single sample may not be representative of the dataset as a whole. Machine learning models, including LGBM, learn from patterns in the training data. If the chosen sample fails to capture these patterns adequately, the resulting interpretation using SHAP may be biased or inaccurate.')
            st.write("\n")
            st.write("Model complexity: LGBM is a boosting algorithm that can create fairly complex models. Interpreting a single prediction may not fully reflect how the model is using the features to make decisions across the entire dataset.")
            st.write("\n")
            st.write("Local vs. global interpretation: SHAP can provide both local (i.e., interpretations for a single data instance) and global (interpretations for the entire dataset) interpretations. While local interpretation may be useful for understanding why the model made a particular prediction for a single sample, it may not capture the complexity of the model on a global scale.")

    if database == 'CSV':
        if file and Xtest is not None:
            st.header("Analytics :chart_with_upwards_trend:")
            threshold = st.slider("Choose Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
            Xtest['Predicted_Class'] = (ypred['prediction_score_1'] > threshold).astype(int)

            # Tabs for graphs
            analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs(
                ["Plotly Scatter Plot 1", "Plotly Scatter Plot 2", "Altair Plot", "Seaborn Box Plot"])
        
            # Scatter plot Recency x Income - Pyplot
            with analysis_tab1:
                st.subheader("Recency x Income")
                if 'Predicted_Class' in Xtest.columns:
                    fig = px.scatter(
                        Xtest,
                        x="Recency",
                        y="Income",
                        color="Predicted_Class",
                        color_continuous_scale="reds"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("'Predicted_Class' not found in DataFrame.")

            # Scatter plot Time_Customer x Income - Pyplot
            with analysis_tab2:
                st.subheader("Scatter Plot Time_Customer x Income")
                if 'Time_Days_Customer' in Xtest.columns and 'Income' in Xtest.columns and 'Predicted_Class' in Xtest.columns:
                    fig = px.scatter(
                        Xtest,
                        x="Time_Days_Customer",
                        y="Income",
                        color="Predicted_Class",
                        color_continuous_scale=px.colors.sequential.Viridis,
                        hover_name="Age",
                    )
                    fig.update_layout(title="Scatter Time_Customer x Income by Predicted Class")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("One or more required columns were not found in the DataFrame.")

            # Scatter plot Income x Age - Altair
            with analysis_tab3:
                st.subheader("Income x Age by Predicted Class")
                # Verificando se as colunas necessárias existem
                if all(col in Xtest.columns for col in ['Income', 'Age', 'Predicted_Class']):
                    chart = alt.Chart(Xtest).mark_circle().encode(
                        x=alt.X('Income:Q', title='Income'),
                        y=alt.Y('Age:Q', title='Age'),
                        color=alt.Color('Predicted_Class:N', title='Predicted Class'),
                        tooltip=[alt.Tooltip('Income:Q'), alt.Tooltip('Age:Q'), alt.Tooltip('Recency:Q')]
                    ).properties(
                        width=800,
                        height=400,
                        title='Income x Age by Predicted Class'
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.error("One or more required columns were not found in the DataFrame.")
            
            # Box plot Income x Response & MntWines x Response - seaborn 
            with analysis_tab4:
                st.subheader("Box Plot for Detailed Ananlysis about Client's Characteristics")
                features_to_plot = ['Income', 'MntWines']
                
                if 'Predicted_Class' in Xtest.columns and all(feature in Xtest.columns for feature in features_to_plot):
                    plt.figure(figsize=(10, 6))
                    
                    for feature in features_to_plot:
                        if pd.api.types.is_numeric_dtype(Xtest[feature]):
                            sns.boxplot(data=Xtest, x='Predicted_Class', y=feature, palette="deep", width=0.5)
                            plt.title(f'Box Plot - {feature} by Predicted Class', fontsize=16)
                            plt.xlabel('Predicted Class', fontsize=14)
                            plt.ylabel(feature, fontsize=14)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            
                            st.pyplot(plt.gcf())
                            plt.clf()
                        else:
                            st.error(f"Error: The column {feature} is not numeric and can not be used in a boxplot.")
                else:
                    missing_columns = ['Predicted_Class'] + [col for col in features_to_plot if col not in Xtest.columns]
                    st.error(f"Error: The following columns does not appear in the dataframe: {', '.join(missing_columns)}")

        else:
            st.warning('CSV file was not loaded.')