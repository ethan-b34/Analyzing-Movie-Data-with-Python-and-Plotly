# Analyzing Movie Release Dates

test:loc[test['release_date'].isnull() == False, 'release_date'].head()

#Preprocessing Features

def fix_date(x):
    year = x.split('/')[2]
    if int(year) <= 19:
        return x[:-2] + '20' + year
    else:
        return x[:-2] + '19' + year

test.loc[test['release_date'].isnull() == True].head()

test.loc[test['release_date'].isnull() == True, 'release_date'] = '05/01/00'

train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))
test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))


#Creating features Based on Release Date

train['release_date'] = pd.to_datetime(train['release_date'])
test['release_date'] = pd.to_datetime(test['release_date'])


def process_date(df):
    date_parts = ['year', 'weekday', 'month', 'weekofyear', 'day', 'quarter']
    for part in date_parts:
        part_col = 'release_date' + '_' + part
        df[part_col] = getattr(df['release_date'].dt, part).astype(int)
    return df

train = process_date(train)
test = process_date(test)

#Using plotly to visualize number of films per year


d1 = train['release_date_year'].value_counts().sort_index()
d2 = test['release_date_year'].value_counts().sort_index()

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

data = [go.Scatter(x=d1.index, y=d1.values, name='train'),
        go.Scatter(x=d2.index, y=d2.values, name='test')]

layout = go.Layout(dict(title = "Number of films per year',
                        xaxis = dict(title = "Year"),
                        yaxis = dict(title = "Count"),
                        ), legend = dict(orientation = 'v'))
py.iplot(dict(data=data, layout=layout))



#6 Number of films and revenue per year


d1 = train['release_date_year'].value_counts().sort_index()
d2 = train.groupby(['release_date_year'])['revenue'].sum()

data = [go.Scatter(x=d1.index, y=d1.values, name='film count'),
        go.Scatter(x=d2.index, y=d2.values, name='total revenue', yaxis='y2')]

layout = go.Layout(dict(title = "Number of films and total revenue per year',
                        xaxis = dict(title = "Year"),
                        yaxis = dict(title = "Count"),
                        yaxis2=dict(title='Total revenue', overlaying = 'y', side = 'right')),
                   legend = dict(orientation = 'v'))
py.iplot(dict(data=data, layout=layout))


#Do release dates impact revenue?
sns.catplot(x='release_date_weekday', y='revenue', data=train);
plt.title('revenue of different days of the week')
















