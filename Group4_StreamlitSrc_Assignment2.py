import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import plotly.figure_factory as ff
import os
import sklearn
import xgboost as xgb

st.title("Bike Sharing Service in Washington, D.C")

# Read the data
path = os.path.dirname(__file__)
raw_data = pd.read_csv(f"{path}/data/bike-sharing_hourly.csv")
data = pd.read_csv(f"{path}/data/bike-sharing_hourly_cleaned.csv")

data["dteday"] = pd.to_datetime(data["dteday"], format="%Y-%m-%d").dt.date

st.sidebar.title("Filters")

startDate = st.sidebar.date_input("Start date", value=pd.to_datetime("2011-01-01"))
endDate = st.sidebar.date_input("End date", value=pd.to_datetime("2012-12-31"))

data_to_show = st.sidebar.multiselect(
    "Select the Process Steps to show",
    ["Introduction", "Raw Data", "EDA Process", "Prediction Process"],
    ["Introduction", "Raw Data", "EDA Process", "Prediction Process"],
)

if "Introduction" in data_to_show:
    st.image("https://media.giphy.com/media/77RGFO6jNhMHK/giphy.gif", width=300)

    st.markdown(
        """     
                Welcome to our bike sharing dashboard. We have created this dashboard to show you the data of the bike sharing in Washington, D.C. and to show you the insights we have found during the EDA process. 
                
                Furthermore, we have created a prediction model which predicts the total count of bikes for a given day and other parameters. You can find the prediction model at the bottom of the page. 
                
                The date range you can define in the sidebar applies on the whole data set and therefore on almost every figure in this dashboard. The other filters only apply to the corresponding figure.
                """
    )


data = data.loc[(data["dteday"] >= startDate) & (data["dteday"] <= endDate)]


################# RAW DATA #################

if "Raw Data" in data_to_show:
    st.header("Raw Data")
    st.dataframe(raw_data.head(5))

################# EDA PROCESS #################
if "EDA Process" in data_to_show:
    st.markdown(
        """
        # EDA Process
        1. Understanding the data
        2. Filling missing rows
        3. Feature Engineering
        4. Outlier Detection
        5. Plotting clear and meaningful figures
        6. Insights on relevant columns for prediction
        7. Data after cleaning
    """
    )
    # Boxplots of Bike Sharing Demand
    st.header("1. Understanding the data")
    # only get numerical columns

    st.dataframe(raw_data.describe())

    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(
        raw_data[
            ["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
        ].corr(),
        annot=True,
        cmap="coolwarm",
    )
    st.pyplot(fig)

    # Conclusion
    st.write(
        """
    - The first thing we can see is that there is a high colinearity between `temp` and `atemp`. Since this columns relate to temperature and perceived temperature it makes sense to only keep one. 
    - Between `registered`, `casual` and `cnt` (target variable) there is also a high correlation. This is due to `cnt` being the sum of `registered` users and `casual` users.
    """
    )

    #############

    st.header("2. Filling missing rows")
    st.write(
        """
            In the dataset there are some missing rows. We extracted all the missing rows and first filled them with dteday and hr.
        """
    )

    st.code(
        """
# find all dates where are less than 24 hours
temp_data = raw_data.groupby('dteday').count().sort_values(by='hr', ascending=True)

# get all dates where are less than 24 hours
dates = temp_data[temp_data['hr'] < 24].index

# add missing hours to the dataset
for date in dates:
    for hour in range(24):
        if hour not in raw_data[raw_data['dteday'] == date]['hr'].values:
            raw_data = pd.concat([raw_data, pd.DataFrame({'instant': [raw_data['instant'].max() + 1], 'dteday': [date], 'hr': [hour]})])
            
# reset the index
raw_data = raw_data.sort_values(by="dteday").reset_index(drop=True)
raw_data['instant'] = raw_data.index + 1
            """
    )

    st.write(
        """
            After that we filled missing values with the mean.
        """
    )

    st.code(
        """
raw_data['dteday'] = pd.to_datetime(raw_data['dteday'])
raw_data['yr'] = raw_data['dteday'].dt.year
raw_data['mnth'] = raw_data['dteday'].dt.month
raw_data['weekday'] = raw_data['dteday'].dt.weekday
season = raw_data.groupby('dteday', group_keys=True).apply(lambda x: x['season'].mode()[0])
weathersit = raw_data.groupby('dteday', group_keys=True).apply(lambda x: x['weathersit'].mode()[0])
holiday = raw_data.groupby('dteday', group_keys=True).apply(lambda x: x['holiday'].mode()[0])
workingday= raw_data.groupby('dteday', group_keys=True).apply(lambda x: x['workingday'].mode()[0])
temp = raw_data.groupby('dteday', group_keys=True).apply(lambda x: x['temp'].mean())
atemp = raw_data.groupby('dteday', group_keys=True).apply(lambda x: x['atemp'].mean())
hum = raw_data.groupby('dteday', group_keys=True).apply(lambda x: x['hum'].mean())
windspeed = raw_data.groupby('dteday', group_keys=True).apply(lambda x: x['windspeed'].mean())
casual = raw_data.groupby('dteday', group_keys=True).apply(lambda x: x['casual'].mean())
registered = raw_data.groupby('dteday', group_keys=True).apply(lambda x: x['registered'].mean())
cnt = raw_data.groupby('dteday', group_keys=True).apply(lambda x: x['cnt'].mean())

raw_data['season'] = raw_data['season'].fillna(raw_data['dteday'].map(season))
raw_data['weathersit'] = raw_data['weathersit'].fillna(raw_data['dteday'].map(weathersit))
raw_data['holiday'] = raw_data['holiday'].fillna(raw_data['dteday'].map(holiday)) 
raw_data['workingday'] = raw_data['workingday'].fillna(raw_data['dteday'].map(workingday))
raw_data['temp'] = raw_data['temp'].fillna(raw_data['dteday'].map(temp))
raw_data['atemp'] = raw_data['atemp'].fillna(raw_data['dteday'].map(atemp))
raw_data['hum'] = raw_data['hum'].fillna(raw_data['dteday'].map(hum))
raw_data['windspeed'] = raw_data['windspeed'].fillna(raw_data['dteday'].map(windspeed))
raw_data['casual'] = raw_data['casual'].fillna(raw_data['dteday'].map(casual))
raw_data['registered'] = raw_data['registered'].fillna(raw_data['dteday'].map(registered))
raw_data['cnt'] = raw_data['cnt'].fillna(raw_data['dteday'].map(cnt))
"""
    )

    ############

    st.header("3. Feature Engineering")

    st.code(
        """
    dict_daylight = {
        1: 9.8,
        2: 10.82,
        3: 11.98,
        4: 13.26,
        5: 14.34,
        6: 14.93,
        7: 14.68,
        8: 13.75,
        9: 12.5,
        10: 11.25,
        11: 10.12,
        12: 9.5
    }

daylight_hrs = pd.DataFrame(dict_daylight.items(), columns=['mnth', 'daylight_hrs'])

data = data.join(daylight_hrs.set_index('mnth'), on='mnth')

data['daylight_hrs'] = data['daylight_hrs'].astype('float64')
            """
    )

    st.code(
        """
season = pd.get_dummies(data['season'], prefix='season')
data = pd.concat([data, season], axis=1)
weather=pd.get_dummies(data['weathersit'],prefix='weathersit')
data=pd.concat([data,weather],axis=1)
weekday=pd.get_dummies(data['weekday'],prefix='weekday')
data=pd.concat([data,weekday],axis=1)
month = pd.get_dummies(data['mnth'], prefix='mnth')
data = pd.concat([data, month], axis=1)
hours = pd.get_dummies(data['hr'], prefix='hr')
data = pd.concat([data, hours], axis=1)
year = pd.get_dummies(data['yr'], prefix='yr')
data = pd.concat([data, year], axis=1)
data['daylight'] = data['hr'].map(lambda x: 1 if (x > 6) & (x < 20) else 0)
data['night_hr'] = data['hr'].map(lambda x: 0 if (x < 7 | x == 23) else 1)
data['peak_hr'] = data['hr'].map(lambda x: 1 if x in [7,8,9,16,17,18,19] else 0)
data.drop(['season','weathersit', "weekday", "registered", "mnth", "dteday", "temp", "instant", "casual", "hr", "yr"],inplace=True,axis=1)
"""
    )

    st.write(
        """
            For the feature engineering we did the following:
            - Add columns:
                * `daylight_hrs` containing average daylight hours for each month.
                * **Dummy variables** for categorical columns.
            - Drop columns:
                * `Registered` and `casual` columns, since they are a consequence of the target variable.
                * `Instant` column since it is an index.
                * `Atemp` column as it is highly correlated to `temp`.
                * `Dteday` column because we already extracted the date parts.
                * **Categorical variables** since they have already been encoded with **dummy encoding**.
            """
    )

    ############

    st.header("4. Outlier Detection")

    data_outliers = raw_data[
        np.abs(raw_data["cnt"] - raw_data["cnt"].mean())
        >= (2.5 * raw_data["cnt"].std())
    ]

    col1, col2, col3 = st.columns(3)

    fig = plt.figure(figsize=(10, 8))
    sns.boxplot(data=raw_data, y="cnt")
    plt.xlabel(xlabel="cnt", fontsize=15)
    col1.pyplot(fig)

    fig2 = plt.figure(figsize=(10, 8))
    sns.boxplot(data=data_outliers, y="cnt", x="season", orient="v")
    col2.pyplot(fig2)

    fig3 = plt.figure(figsize=(10, 8))
    sns.boxplot(data=raw_data, y="cnt", x="season", orient="v")
    col3.pyplot(fig3)

    col1.markdown(
        """
                #### Raw Data
                Boxplot for the cnt column showing that there are a lot outliers.
                """
    )

    col2.markdown(
        """
                #### Outliers - Season Analysis
                These are the cnt values by season for the outliers.
                """
    )

    col3.markdown(
        """
                #### Data - Season Analysis
                These are the cnt values by season for the whole dataset.
                """
    )

    """
    We have seen, that there is no difference between the outliers and the cnt values by season. Therefore we decided to remove the outliers. To do that we have used the following code:
    """

    st.code(
        """data = data[np.abs(data["cnt"]-data["cnt"].mean())<=(2.5*data["cnt"].std())]"""
    )

    """
    This code removes outliers which are more than 2.5 standard deviations away from the mean.

    ### After outlier removal
    """

    fig = plt.figure(figsize=(10, 4))
    sns.boxplot(data=data, y="cnt")
    st.pyplot(fig)

    """
    As we can see, there are less outliers now. We have removed 3% of the data.
    """

    ##########

    # Boxplots of Bike Sharing Demand
    st.header("5. Plotting clear and meaningful figures")

    """
        Please use the selectbox to choose a column to plot. The plot and descriptions will be shown below.
    """

    # only get numerical columns
    options = {
        "Daylight Hours": "daylight_hrs",
        "Holiday": "holiday",
        "Hour": "hr",
        "Season": "season",
        "Temperature": "temp",
        "Year": "yr",
        "Weather": "weathersit",
        "Working Day": "workingday",
    }

    column = st.selectbox(
        "Select Column",
        options=options.keys(),
    )

    column_chosen = options[column]

    fig = px.box(data, y="cnt", x=column_chosen, orientation="v")
    st.plotly_chart(fig)

    descriptions = {
        "Daylight Hours": "This plot shows that, on average, the higher the total Daylight Hours are, the more rentals are made. This makes sense since more daylight translates in more hours with higher visibility and nicer weather to enjoy a bike ride.",
        "Holiday": "Even though both plots look similar, we can see that the median for holidays is lower. This may result from the less rentals made by people who normally commute to work by bike but that were out for holidays.",
        "Hour": "By looking at this plot, we can see that the usage of bikes throughout the day is highly hour dependant. There are clear peaks in the morning (6am-8am) and in the afternoon (4pm-8pm), which might relate with people commuting to and back from work. There are two valleys, the lowest one during the early morning hours and the second during working hours.",
        "Season": "From this plot we can see that the seasons have a clear impact on the number of rentals. Hence, for season 1, there is a lower rental behaviour overall, while Season 2 and 3 show the highest number of rentals. This may have a relationship with the temperature and daylight hours variables which, in turn, are directly affected by the season.",
        "Temperature": "In this case we can see that the higher the temperature the more rentals overall. It is worth saying that once the temperature starts getting too high, there is a slight decrease on the rentals, which can be seen at the right-side of this plot. The latter, relates to the fact that in Seasons with better weather, there tends to be more rentals.",
        "Year": "This graph shows the number of rentals for each year. We can see that the second year (2012) has more rentals, both on average and when looking at the median and Q3. This might lead us to believe that there was an increase in the number of available bikes, which was what also led to increase in the dispersion of the data (higher IQR).",
        "Weather": "We can see that the better the weather conditions are (1 and 2) the higher the number of rentals. On the other hand, we can see that once the weather conditions get worse, there is a clear decline in usage (3 and 4). Additionally, when the weather gets really harsh (4), the number of users overall decreases. This makes sense since bike users are more exposed to weather conditions.",
        "Working Day": "The behaviour for Working Day among bike users is not that different. However, when talking about the actual working days, we can see a higher average number of rentals. This can be related to the fact that bikes are being used by workers in Washington D.C. to commute to work.",
    }

    st.write(f"Description: ")
    st.write(descriptions[column])

    ##########

    st.header("6. Insights on relevant columns for prediction")

    st.markdown(
        """
                ### We have a lot of columns in our dataset. We will focus on the following columns for our prediction:
                - **Season**: 1 = spring, 2 = summer, 3 = fall, 4 = winter
                - **Year**: 0 = 2011, 1 = 2012
                - **Month**: 1 to 12
                - **Hour**: 0 to 23
                - **Holiday**: whether the day is considered a holiday
                - **Working Day**: whether the day is neither a weekend nor holiday
                - **Weather**: 1: Clear, Few clouds, Partly cloudy, Partly cloudy
                - **Temperature**: temperature in Celsius
                - **Humidity**: relative humidity
                - **Windspeed**: wind speed
                
                ### Furthermore, we have the following insights from the charts above:
                The bike sharing rental is highest during:
                  * Year 1 (2012)
                  * Summer 
                  * The months of July and August
                  * The hours of 7am and 8am, and 5pm and 6pm
                  * A working day
                  * Clear weather conditions
                  * Temperatures between 20 and 30 degrees Celsius
                  * A humidity sensation between 40 and 60 percent
                  * A windspeed condition between 0 and 20 km/h
            """
    )

    ############

    # Cleaned data

    st.header("7. Data after cleaning")

    st.dataframe(data.head(5))
    st.dataframe(data.describe())

    ###########

    # Line Chart of Bike Sharing Demand
    st.write("### Bike Sharing Demand by Date")

    # use ff to create timeseries plot
    fig = px.line(data, x="dteday", y="cnt")
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


################# PREDCITION PROCESS #################

if "Prediction Process" in data_to_show:
    st.write("# Prediction Process")

    st.header("RMSE")

    rmse_frame = pd.read_csv(f"{path}/data/rmse_frame.csv")

    st.dataframe(
        rmse_frame.sort_values(by="RMSE", ascending=True)
        .reset_index(drop=True)
        .style.background_gradient(axis=0, cmap="RdYlGn_r")
    )

    st.write(
        """
             We can see that the best model is the XGBoost model with a RMSE of 42.97. 
             
             Therefore we chose this model to predict the bike sharing demand.
             
             We fitted the model again with other parameters to get a better prediction.
             """
    )

    st.code(
        """
reg = xbg.XGBRegressor(n_estimators=3000, early_stopping_rounds=300, learning_rate=0.05, n_jobs=-1)
reg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=100)
    """
    )

    st.header("Feature Importance")
    fig = plt.figure(figsize=(10, 5))
    feat_importances = pd.read_csv(f"{path}/data/feat_importances.csv")
    feat_importances.rename(columns={"Unnamed: 0": "index"}, inplace=True)
    most_important = (
        feat_importances.sort_values(by="Importance", ascending=False)
        .head(10)
        .sort_values(by="Importance", ascending=True)
    )
    fig = px.bar(
        most_important,
        x="Importance",
        y="index",
        orientation="h",
        labels={"Importance": "Importance", "index": "Feature"},
    )
    st.plotly_chart(fig)

    ######### PREDICTION #########

    dict_daylight = {
        1: 9.8,
        2: 10.82,
        3: 11.98,
        4: 13.26,
        5: 14.34,
        6: 14.93,
        7: 14.68,
        8: 13.75,
        9: 12.5,
        10: 11.25,
        11: 10.12,
        12: 9.5,
    }

    st.header("Predictions")

    col1, col2, col3 = st.columns(3)

    yr = col1.selectbox("Year", options=[0, 1], index=1)
    hr = col3.selectbox(
        "Hour",
        options=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ],
        index=8,
    )
    mnth = col2.selectbox(
        "Month", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=5
    )
    holiday = col1.selectbox("Holiday", options=[0, 1], index=0)
    workingday = col2.selectbox("Working Day", options=[0, 1], index=1)
    weekday = col3.selectbox("Weekday", options=[0, 1, 2, 3, 4, 5, 6], index=2)
    hum = col1.slider("Humidity", min_value=0.0, max_value=1.0, value=0.5)
    windspeed = col2.slider("Windspeed", min_value=0.0, max_value=1.0, value=0.5)
    daylight_hrs = dict_daylight[mnth]
    season = (
        1
        if mnth in [3, 4, 5]
        else 2
        if mnth in [6, 7, 8]
        else 3
        if mnth in [9, 10, 11]
        else 4
    )
    temp = col3.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
    weathersit = col1.selectbox("Weather Situation", options=[1, 2, 3, 4], index=0)

    values = [holiday, workingday, temp, hum, windspeed, daylight_hrs]
    values_cat = [season, weathersit, weekday, mnth, hr, yr]

    columns = [
        "holiday",
        "workingday",
        "atemp",
        "hum",
        "windspeed",
        "daylight_hrs",
        "season_1",
        "season_2",
        "season_3",
        "season_4",
        "weathersit_1",
        "weathersit_2",
        "weathersit_3",
        "weathersit_4",
        "weekday_0",
        "weekday_1",
        "weekday_2",
        "weekday_3",
        "weekday_4",
        "weekday_5",
        "weekday_6",
        "mnth_1",
        "mnth_2",
        "mnth_3",
        "mnth_4",
        "mnth_5",
        "mnth_6",
        "mnth_7",
        "mnth_8",
        "mnth_9",
        "mnth_10",
        "mnth_11",
        "mnth_12",
        "hr_0",
        "hr_1",
        "hr_2",
        "hr_3",
        "hr_4",
        "hr_5",
        "hr_6",
        "hr_7",
        "hr_8",
        "hr_9",
        "hr_10",
        "hr_11",
        "hr_12",
        "hr_13",
        "hr_14",
        "hr_15",
        "hr_16",
        "hr_17",
        "hr_18",
        "hr_19",
        "hr_20",
        "hr_21",
        "hr_22",
        "hr_23",
        "yr_2011",
        "yr_2012",
        "daylight",
        "night_hr",
        "peak_hr",
    ]

    x_test = {}

    i = 0

    for col in columns:
        x_test[col] = values[i]
        i += 1
        if i == 6:
            break

    for i in range(4):
        if season == i:
            x_test["season_" + str(i + 1)] = 1
        else:
            x_test["season_" + str(i + 1)] = 0

    for i in range(4):
        if weathersit == i:
            x_test["weathersit_" + str(i + 1)] = 1
        else:
            x_test["weathersit_" + str(i + 1)] = 0

    for i in range(7):
        if weekday == i:
            x_test["weekday_" + str(i)] = 1
        else:
            x_test["weekday_" + str(i)] = 0

    for i in range(12):
        if mnth == i:
            x_test["mnth_" + str(i + 1)] = 1
        else:
            x_test["mnth_" + str(i + 1)] = 0

    for i in range(24):
        if hr == i:
            x_test["hr_" + str(i)] = 1
        else:
            x_test["hr_" + str(i)] = 0

    for i in range(2):
        if yr == i:
            x_test["yr_" + str(2011 + i)] = 1
        else:
            x_test["yr_" + str(2011 + i)] = 0

    x_test["daylight"] = 1 if (hr > 7 & hr < 20) else 0
    x_test["night_hr"] = 1 if (hr < 7 | hr == 23) else 0
    x_test["peak_hr"] = 1 if hr in [7, 8, 9, 16, 17, 18, 19] else 0

    x_test = pd.DataFrame(x_test, index=[0])

    st.write(
        """
             ### x_test
             """
    )
    st.dataframe(x_test)

    loaded_model = pickle.load(open(f"{path}/models/model.sav", "rb"))

    result = loaded_model.predict(x_test)

    st.metric("Predicted Rentals", f"{int(result[0])}")

    st.write(
        """
### Recommendations for Washington Bike Sharing Company:

#### Maintenance:

1. Regular maintenance should be scheduled during the non-busy times of the day, such as early morning or late night.
2. The bikes should be inspected daily, especially during peak season, to identify and address any issues promptly.
3. A system for users to report any bike issues should be put in place, and these reports should be addressed promptly.

#### Moving Trends

1. Holidays: During holidays, there may be a decrease in bike usage due to people being off work or traveling. The company could consider offering holiday-specific promotions or discounts to encourage bike usage during these times.
2. Month: As mentioned in the dataset insights, bike sharing rental is highest during summer months, specifically July and August. The company could focus on promoting bike usage during this time and offer summer-specific incentives.
3. Daytime: Bike usage may vary throughout the day, with peaks during commute times. The company could consider adjusting bike availability and pricing during peak hours to better meet rider demand.
4. Day of the week: Weekdays may see higher bike usage due to commuting, while weekends may see more recreational bike usage. The company could adjust promotions or incentives based on the day of the week to better target rider preferences.
5. Weekend: Weekends may see different patterns of bike usage compared to weekdays. The company could adjust promotions or incentives based on weekend trends, such as offering discounts for weekend rentals or promoting bike usage for recreational activities.

#### Incentives for taking bikes from a certain station:

- During commute times in the morning and afternoon, offer incentives for riders to take bikes from stations that have high bike availability.
- Offer incentives for tourist areas, such as offering discounts or rewards for riders who take bikes from stations near popular tourist attractions.

#### Incentives for returning bikes to certain bike stations:

- During commute times in the morning and afternoon, offer incentives for riders to return bikes to stations that have low bike availability.
- Offer incentives to return bikes to stations in commuter-heavy areas, so that they are available for riders who need to take bikes to work.
"""
    )
