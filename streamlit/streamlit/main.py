import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import joblib
import json
import pickle
import lightgbm
from sklearn.svm import SVC
import xgboost


def scaling(data, mean, scale):
    return (np.array(data) - mean) / scale


def show_metrics(y_test, y_pred, place):
    test_metrics = metrics.classification_report(y_test, y_pred, output_dict=True)

    df_test_metrics = pd.DataFrame({
        'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
        'Value': [
            test_metrics["accuracy"],
            test_metrics["1"]["precision"],
            test_metrics["1"]["recall"],
            test_metrics["1"]["f1-score"]
        ]
    })

    styler = (df_test_metrics.style.format(subset=['Value'], decimal=',', precision=4))
    place.write(styler.to_html(), unsafe_allow_html=True)


def show_heat_map(y_test, y_pred, title, place):
    fig, ax = plt.subplots()
    ax.set_title(title)
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_matrix = pd.DataFrame(
        data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
        index=['Predict Positive:1', 'Predict Negative:0']
    )

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
    place.write(fig)


def main():
    models = ['XGBoost', 'RandomForest','Light GBM', 'SVM']
    st.title('Прогнозування Відтоку Клієнтів для Телекомунікаційної компанії')

    current_model = st.sidebar.selectbox('Виберіть модель для прогнозу:', models)
    is_tv_subscriber = st.sidebar.checkbox('Є абонентом телебачення')
    is_movie_package_subscriber = st.sidebar.checkbox('Є абонентом пакету фільмів')
    subscription_age = st.sidebar.number_input('Вік підписки', min_value=0.00, step=0.01)
    bill_avg = st.sidebar.number_input('Середній рахунок', min_value=0.00, step=0.01)
    reamining_contract = st.sidebar.number_input('Що залишилися за договором', min_value=0.00, step=0.01)
    service_failure_count = st.sidebar.number_input('Кількість збоїв обслуговування', min_value=0)
    download_avg = st.sidebar.number_input('Середній вхідний трафік', min_value=0.00, step=0.01)
    upload_avg = st.sidebar.number_input('Середній вихідний трафік', min_value=0.00, step=0.01)
    download_over_limit = st.sidebar.number_input('Вхідний трафік понад ліміт', min_value=0.00, step=0.01)

    data = [
        int(is_tv_subscriber),
        int(is_movie_package_subscriber),
        subscription_age,
        bill_avg,
        reamining_contract,
        service_failure_count,
        download_avg,
        upload_avg,
        download_over_limit,
    ]

    y_pred = []
    y_test = []
    y_train = []
    y_pred_train = []

    if current_model == 'Light GBM':
        y_pred = json.load(open('./lightGbm/y_pred.json'))
        y_test = json.load(open('./lightGbm/y_test.json'))
        y_train = json.load(open('./lightGbm/y_train.json'))
        y_pred_train = json.load(open('./lightGbm/y_pred_train.json'))
        mean = json.load(open('./lightGbm/mean.json'))
        scale = json.load(open('./lightGbm/scale.json'))
        model = joblib.load('./lightGbm/light_gbm.pkl')
        data = scaling(data, mean, scale)
    elif current_model == 'SVM':
        y_pred = json.load(open('./svm/y_pred.json'))
        y_test = json.load(open('./svm/y_test.json'))
        y_train = json.load(open('./svm/y_train.json'))
        y_pred_train = json.load(open('./svm/y_pred_train.json'))
        mean = json.load(open('./svm/mean.json'))
        scale = json.load(open('./svm/scale.json'))
        model = pickle.load(open("./svm/model_svm.pkl", "rb"))
        data = scaling(data, mean, scale)
    elif current_model == 'XGBoost':
        y_pred = json.load(open('./XGBoost/y_pred.json'))
        y_test = json.load(open('./XGBoost/y_test.json'))
        y_train = json.load(open('./XGBoost/y_train.json'))
        y_pred_train = json.load(open('./XGBoost/y_pred_train.json'))
        model = joblib.load('./XGBoost/XGBoost_model.pkl')
    elif current_model == 'RandomForest':
        y_pred = json.load(open('./RandomForest/y_pred.json'))
        y_test = json.load(open('./RandomForest/y_test.json'))
        y_train = json.load(open('./RandomForest/y_train.json'))
        y_pred_train = json.load(open('./RandomForest/y_pred_train.json'))
        model = joblib.load('./RandomForest/RandomForest_model.pkl')

    col1, col2 = st.columns(2)

    show_heat_map(y_test, y_pred, 'Test data', col1)
    show_heat_map(y_train, y_pred_train, 'Train data', col2)
    show_metrics(y_test, y_pred, col1)
    show_metrics(y_train, y_pred_train, col2)

    if st.sidebar.button('Отримати прогноз...'):
        result = model.predict([data])

        if result == 1:
            st.sidebar.write(':red[Висока вірогідність того, що клієнт покине компанію]')
        else:
            st.sidebar.write(':green[Висока вірогідність того, що клієнт не покине компанію]')


if __name__ == '__main__':
    main()
