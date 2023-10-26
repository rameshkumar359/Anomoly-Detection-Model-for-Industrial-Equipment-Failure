import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN


icon = Image.open('apple-xxl.png')
st.set_page_config(page_title="BOILER ANAMOLY DETECTION",
                   page_icon=icon,
                   layout="wide",
                   initial_sidebar_state="expanded",
                   )
st.markdown("<h1 style='text-align: center; color: #051937;background-color:white;border-radius:15px;'>BOILER ANAMOLY DETECTION</h1>",
            unsafe_allow_html=True)


with st.sidebar:
    selected = option_menu(None, ["ANALYSIS", "PREDICTION"],
                           icons=["bi bi-clipboard-data", "bi bi-magic"],
                           default_index=0,
                           orientation="vertical",
                           styles={"nav-link": {"font-size": "20px", "text-align": "centre", "margin-top": "20px",
                                                "--hover-color": "#266c81"},
                                   "icon": {"font-size": "20px"},
                                   "container": {"max-width": "6000px"},
                                   "nav-link-selected": {"background-color": "#266c81"}, })

# setting the back-ground color


def back_ground():
    st.markdown(f""" <style>.stApp {{
                        background-image: linear-gradient(to right top, #051937, #051937, #051937, #051937, #051937);;
                        background-size: cover}}
                     </style>""", unsafe_allow_html=True)


back_ground()

if selected == 'ANALYSIS':
    df = pd.read_csv('sensor_data(2).csv')
    st.markdown("")
    st.markdown("## :white[SAMPLE DATA]")
    st.table(df.head())

    st.markdown("")
    st.markdown("")

    st.markdown("## :white[BOILER-A]")

    df['Timestamps'] = pd.to_datetime(df['Timestamp'])
    df['year'] = df['Timestamps'].apply(lambda x: x.year)
    df['month'] = df['Timestamps'].apply(lambda x: x.month)
    df['day'] = df['Timestamps'].apply(lambda x: x.day)
    df['time'] = df['Timestamps'].dt.strftime('%H:%M:%S')
    df.drop('Timestamp', axis=1, inplace=True)
    Boiler_A = df[df['Boiler Name'] == 'Boiler A']
    st.markdown("#### :white[AFTER PROCESSING THE DATA]")
    st.table(Boiler_A.head())

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.lineplot(
        data=Boiler_A, x=Boiler_A['Timestamps'], y=Boiler_A['Temperature'])
    st.markdown("#### :white[SIMPLE LINE PLOT OF TEMPERATURE OVER TIMESERIES]")
    st.pyplot(fig)

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(Boiler_A['Temperature'], kde=True)
    st.markdown("#### :white[THE DISTRIBUTION OF TEMPERATURE]")
    st.pyplot(fig)

    fig = px.line(Boiler_A, x="Timestamps", y="Temperature",
                  color='Anomaly', title='Temperature with time series')
    fig.update_layout(
        autosize=True,
        width=1200,
        height=800)
    st.markdown("#### :white[TEMPERATURE OVER TIMESERIES FOR BOILER-A]")
    st.write(fig)

    Boiler_B = df[df['Boiler Name'] == 'Boiler B']

    st.markdown("## :white[BOILER-B]")
    st.table(Boiler_B.head())

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.lineplot(
        data=Boiler_B, x=Boiler_B['Timestamps'], y=Boiler_B['Temperature'])
    st.markdown("#### :white[SIMPLE LINE PLOT OF TEMPERATURE OVER TIMESERIES]")
    st.pyplot(fig)

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(Boiler_B['Temperature'], kde=True)
    st.markdown("#### :white[THE DISTRIBUTION OF TEMPERATURE]")
    st.pyplot(fig)

    fig = px.line(Boiler_B, x="Timestamps", y="Temperature",
                  color='Anomaly', title='Temperature with time series')
    fig.update_layout(
        autosize=True,
        width=1200,
        height=800)
    st.markdown("#### :white[TEMPERATURE OVER TIMESERIES FOR BOILER-B]")
    st.write(fig)

    Boiler_C = df[df['Boiler Name'] == 'Boiler C']
    st.markdown("## :white[BOILER-C]")
    st.table(Boiler_C.head())

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.lineplot(
        data=Boiler_C, x=Boiler_C['Timestamps'], y=Boiler_C['Temperature'])
    st.markdown("#### :white[SIMPLE LINE PLOT OF TEMPERATURE OVER TIMESERIES]")
    st.pyplot(fig)

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(Boiler_C['Temperature'], kde=True)
    st.markdown("#### :white[THE DISTRIBUTION OF TEMPERATURE]")
    st.pyplot(fig)

    fig = px.line(Boiler_C, x="Timestamps", y="Temperature",
                  color='Anomaly', title='Temperature with time series')
    fig.update_layout(
        autosize=True,
        width=1200,
        height=800)
    st.markdown("#### :white[TEMPERATURE OVER TIMESERIES FOR BOILER-C]")
    st.write(fig)

    Boiler_D = df[df['Boiler Name'] == 'Boiler D']

    st.markdown("## :white[BOILER-D]")
    st.table(Boiler_D.head())

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.lineplot(
        data=Boiler_D, x=Boiler_D['Timestamps'], y=Boiler_D['Temperature'])
    st.markdown("#### :white[SIMPLE LINE PLOT OF TEMPERATURE OVER TIMESERIES]")
    st.pyplot(fig)

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.histplot(Boiler_D['Temperature'], kde=True)
    st.markdown("#### :white[THE DISTRIBUTION OF TEMPERATURE]")
    st.pyplot(fig)

    fig = px.line(Boiler_D, x="Timestamps", y="Temperature",
                  color='Anomaly', title='Temperature with time series')
    fig.update_layout(
        autosize=True,
        width=1200,
        height=800)
    st.markdown("#### :white[TEMPERATURE OVER TIMESERIES FOR BOILER-D]")
    st.write(fig)

    fig, ax = plt.subplots()  # solved by add this line
    plt.tight_layout()
    ax = sns.heatmap(df[['Temperature', 'Anomaly']].corr(),
                     cmap='coolwarm', annot=True)
    st.markdown("#### :white[HEATMAP]")
    st.pyplot(fig)

    gp_temp = df.groupby('Boiler Name')['Temperature'].mean().reset_index()
    fig = px.bar(gp_temp, x='Boiler Name', y='Temperature',
                 title='MEAN TEMPERATURES OF BOILER', color='Boiler Name')
    fig.update_layout(
        autosize=True,
        width=1200,
        height=800)
    st.markdown("#### :white[MEAN TEMPERATURE OF BOILERS OVER TIME]")
    st.write(fig)
    boiler_A_anomoly = Boiler_A[Boiler_A['Anomaly'] == 1]
    boiler_B_anomoly = Boiler_B[Boiler_B['Anomaly'] == 1]
    boiler_C_anomoly = Boiler_C[Boiler_C['Anomaly'] == 1]
    boiler_D_anomoly = Boiler_D[Boiler_D['Anomaly'] == 1]
    fp = pd.concat([boiler_A_anomoly, boiler_B_anomoly,
                   boiler_C_anomoly, boiler_D_anomoly])
    fig = px.line(fp, y="Temperature", color='Boiler Name')
    fig.update_layout(
        autosize=True,
        width=1200,
        height=800)
    st.markdown("#### :white[ANOMOLY POINTS OF DIFFERENT BOILERS]")
    st.write(fig)

    fig, ax = plt.subplots()
    X1 = Boiler_A['Temperature']
    y1 = Boiler_A['Anomaly']
    X1 = X1.values.reshape(-1, 1)

    db = DBSCAN(eps=0.6, min_samples=8).fit(X1)
    labels = db.labels_
    labels_true = y1

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    fig, ax = plt.subplots()
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X1[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X1[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    st.markdown(
        '### :white[CLUSTERING BEFORE SYNTHETIC DATA GENERATION FOR BOILER-A]')
    st.pyplot(fig)

    xr, yr = SMOTE(k_neighbors=3).fit_resample(X1, y1)
    db = DBSCAN(eps=0.6, min_samples=8).fit(xr)
    labels = db.labels_
    labels_true = yr

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    fig, ax = plt.subplots()
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = xr[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = xr[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    st.markdown(
        '### :white[CLUSTERING AFTER SYNTHETIC DATA GENERATION FOR BOILER-A]')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    X1 = Boiler_B['Temperature']
    y1 = Boiler_B['Anomaly']
    X1 = X1.values.reshape(-1, 1)

    db = DBSCAN(eps=0.6, min_samples=8).fit(X1)
    labels = db.labels_
    labels_true = y1

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    fig, ax = plt.subplots()
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X1[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X1[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    st.markdown(
        '### :white[CLUSTERING BEFORE SYNTHETIC DATA GENERATION FOR BOILER-B]')
    st.pyplot(fig)

    xr, yr = SMOTE(k_neighbors=3).fit_resample(X1, y1)
    db = DBSCAN(eps=0.6, min_samples=8).fit(xr)
    labels = db.labels_
    labels_true = yr

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    fig, ax = plt.subplots()
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = xr[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = xr[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    st.markdown(
        '### :white[CLUSTERING AFTER SYNTHETIC DATA GENERATION FOR BOILER-B]')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    X1 = Boiler_C['Temperature']
    y1 = Boiler_C['Anomaly']
    X1 = X1.values.reshape(-1, 1)

    db = DBSCAN(eps=0.6, min_samples=8).fit(X1)
    labels = db.labels_
    labels_true = y1

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    fig, ax = plt.subplots()
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X1[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X1[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    st.markdown(
        '### :white[CLUSTERING BEFORE SYNTHETIC DATA GENERATION FOR BOILER-C]')
    st.pyplot(fig)

    xr, yr = SMOTE(k_neighbors=3).fit_resample(X1, y1)
    db = DBSCAN(eps=0.6, min_samples=8).fit(xr)
    labels = db.labels_
    labels_true = yr

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    fig, ax = plt.subplots()
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = xr[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = xr[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    st.markdown(
        '### :white[CLUSTERING AFTER SYNTHETIC DATA GENERATION FOR BOILER-C]')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    X1 = Boiler_D['Temperature']
    y1 = Boiler_D['Anomaly']
    X1 = X1.values.reshape(-1, 1)

    db = DBSCAN(eps=0.6, min_samples=8).fit(X1)
    labels = db.labels_
    labels_true = y1

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    fig, ax = plt.subplots()
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X1[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X1[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    st.markdown(
        '### :white[CLUSTERING BEFORE SYNTHETIC DATA GENERATION FOR BOILER-D]')
    st.pyplot(fig)

    xr, yr = SMOTE(k_neighbors=3).fit_resample(X1, y1)
    db = DBSCAN(eps=0.6, min_samples=8).fit(xr)
    labels = db.labels_
    labels_true = yr

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    fig, ax = plt.subplots()
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = xr[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = xr[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    st.markdown(
        '### :white[CLUSTERING AFTER SYNTHETIC DATA GENERATION FOR BOILER-D]')
    st.pyplot(fig)

if selected == 'PREDICTION':
    df = pd.read_csv('sensor_data(2).csv')
    df['Timestamps'] = pd.to_datetime(df['Timestamp'])
    df['year'] = df['Timestamps'].apply(lambda x: x.year)
    df['month'] = df['Timestamps'].apply(lambda x: x.month)
    df['day'] = df['Timestamps'].apply(lambda x: x.day)
    df['time'] = df['Timestamps'].dt.strftime('%H:%M:%S')
    df.drop('Timestamp', axis=1, inplace=True)
    Boiler_A = df[df['Boiler Name'] == 'Boiler A']
    st.markdown('## :white[BOILER-A]')
    a = st.text_input(label="TEMPERATURE-A")
    if st.button('PREDICT A'):
        X1 = Boiler_A['Temperature']
        y1 = Boiler_A['Anomaly']
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy='minority')
        X1 = X1.values.reshape(-1, 1)
        X_sm, y_sm = smote.fit_resample(X1, y1)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        X_train, X_test, y_train, y_test = train_test_split(
            X_sm, y_sm, test_size=0.4, random_state=50)
        clf.fit(X_train, y_train)
        pred = clf.predict([[float(a)]])
        check = list(pred)
        if check[0] == 1:
            st.markdown('### :white[Anomoly Detection]')
        else:
            st.markdown('### :white[Normal No Anomoly]')

    Boiler_B = df[df['Boiler Name'] == 'Boiler B']
    st.markdown('## :white[BOILER-B]')
    b = st.text_input(label="TEMPERATURE-B")
    if st.button('PREDICT B'):
        X1 = Boiler_B['Temperature']
        y1 = Boiler_B['Anomaly']
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy='minority')
        X1 = X1.values.reshape(-1, 1)
        X_sm, y_sm = SMOTE(k_neighbors=3).fit_resample(X1, y1)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        X_train, X_test, y_train, y_test = train_test_split(
            X_sm, y_sm, test_size=0.4, random_state=50)
        clf.fit(X_train, y_train)
        pred = clf.predict([[float(b)]])
        check = list(pred)
        if check[0] == 1:
            st.markdown('### :white[Anomoly Detection]')
        else:
            st.markdown('### :white[Normal No Anomoly]')

    Boiler_C = df[df['Boiler Name'] == 'Boiler C']
    st.markdown('## :white[BOILER-C]')
    c = st.text_input(label="TEMPERATURE-C")
    if st.button('PREDICT C'):
        X1 = Boiler_C['Temperature']
        y1 = Boiler_C['Anomaly']
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy='minority')
        X1 = X1.values.reshape(-1, 1)
        X_sm, y_sm = SMOTE(k_neighbors=3).fit_resample(X1, y1)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        X_train, X_test, y_train, y_test = train_test_split(
            X_sm, y_sm, test_size=0.4, random_state=50)
        clf.fit(X_train, y_train)
        pred = clf.predict([[float(c)]])
        check = list(pred)
        if check[0] == 1:
            st.markdown('### :white[Anomoly Detection]')
        else:
            st.markdown('### :white[Normal No Anomoly]')

    Boiler_D = df[df['Boiler Name'] == 'Boiler D']
    st.markdown('## :white[BOILER-D]')
    d = st.text_input(label="TEMPERATURE-D")
    if st.button('PREDICT D'):
        X1 = Boiler_D['Temperature']
        y1 = Boiler_D['Anomaly']
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy='minority')
        X1 = X1.values.reshape(-1, 1)
        X_sm, y_sm = SMOTE(k_neighbors=4).fit_resample(X1, y1)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        X_train, X_test, y_train, y_test = train_test_split(
            X_sm, y_sm, test_size=0.4, random_state=50)
        clf.fit(X_train, y_train)
        pred = clf.predict([[float(d)]])
        check = list(pred)
        if check[0] == 1:
            st.markdown('### :white[Anomoly Detection]')
        else:
            st.markdown('### :white[Normal No Anomoly]')
