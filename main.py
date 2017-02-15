#!/usr/bin/env python3
import solar_corr
import pandas as pd
import numpy as np
import itertools as IT
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from bokeh.charts import Bar, Donut, Histogram, output_file, show, reset_output
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter
from bokeh.charts.attributes import cat
from bokeh.layouts import gridplot


def csv_files_to_dataframes( path_to_dir, suffix=".csv" ):

    filenames = listdir( path_to_dir )
    
    csv_files = [filename for filename in filenames if filename.endswith( suffix )]
    
    csv_DF_dict = {}
    
    for file in csv_files:
        filepath = path_to_dir + file
        csv_DF_dict[str(file)] = pd.DataFrame.from_csv( path_to_dir + file ) 

    return csv_DF_dict


def column_solar_corr(data, columns, input_csv, sun, image_path):

    data = data.loc[:, columns]
    data = pd.get_dummies(data)
    
    new_columns = []
    for column_name in data.columns:
        new_column_name = column_name.replace(",", "")
        new_column_name = new_column_name.replace(" ", "")
        new_column_name = new_column_name.replace("\n", "")
        new_columns.append(new_column_name)
    
    data.columns = new_columns

    data.to_csv(input_csv, index=False)

    solar_corr.main(input_csv, sun, image_path)


def build_bar_chart(data, column, agg='count'):

    data = pd.DataFrame(data[column].value_counts()).reset_index()
    data.columns = [column, 'Count']

    bar = Bar(data,
          values='Count',
          label=cat(columns=column, sort=False),
          legend=False,
          width=800,
          title="Count by " + column)

    reset_output()
    output_file('barCharts/' + 'Countby' + column + '.html')

    show(bar)


def product_incident_summary(data, product_code):

    product_data = data.loc[data['prod1'].isin(
        product_code) | data['prod2'].isin(product_code)]

    hist = Histogram(product_data,
        values='age',
        color='steelblue',
        title="Product Code(s): " + str(product_code) + " Age Histogram",
        bins=25,
        width=600,
        height=600)

    donut_dataDF = pd.DataFrame(
        product_data['sex'].value_counts()).reset_index()
    donut_dataDF.columns = ['Sex', 'Count']
    donut_dataDF['Count'] = donut_dataDF['Count'] / donut_dataDF['Count'].sum()
    
    donut = Donut(donut_dataDF,
        label='Sex',
        values='Count',
        color=['lightpink', 'lightblue'],
        width=600,
        height=600, 
        hover_tool=True,
        text_font_size='16pt')

    diag_bar_dataDF = pd.DataFrame(
        product_data['Diagnosis'].value_counts()).reset_index()
    diag_bar_dataDF.columns = ['Diagnosis', 'Count']

    diag_bar = Bar(diag_bar_dataDF,
        label=cat(columns='Diagnosis', sort=False),
        values='Count',
        color='steelblue',
        legend=False,
        width=600,
        height=600,
        title="Product Code(s): " + str(product_code) + " Diagnosis Distribution")

    diagnosis_columns = [
        TableColumn(field="Diagnosis", title='Diagnosis'),
        TableColumn(field="Count", title='Count')
    ]
    diagnosis_source = ColumnDataSource(diag_bar_dataDF)

    diag_data_table = DataTable(source=diagnosis_source,
        columns=diagnosis_columns,
        width=600)

    sex_columns = [
        TableColumn(field='Sex', title='Sex'),
        TableColumn(field='Count', title='Percent',
            formatter=NumberFormatter(format='0.00%'))
    ]
    sex_data_source = ColumnDataSource(donut_dataDF)

    sex_data_table = DataTable(source=sex_data_source,
        columns=sex_columns,
        width=600)

    count, division = np.histogram(product_data['age'], bins=25)
    hist_list = [division, count]
    age_hist_data = pd.DataFrame(hist_list).T
    age_hist_data.columns = ['Age Bin', 'Count']
    
    age_data_columns = [
        TableColumn(field='Age Bin', title='Age Bin',
            formatter=NumberFormatter(format="0.00")),
        TableColumn(field='Count', title='Count')
    ]
    age_data_source = ColumnDataSource(age_hist_data)

    age_data_table = DataTable(source=age_data_source,
        columns=age_data_columns,
        width=600)

    print("Total " + str(product_code) + " Injuries: {}".format(
        len(product_data)))
    
    male_index = donut_dataDF.loc[donut_dataDF['Sex']=='Male'].index[0]
    print("Percent male: {}".format(
        donut_dataDF.get_value(index=male_index, col='Count')))
    
    female_index = donut_dataDF.loc[donut_dataDF['Sex']=='Female'].index[0]
    print("Percent female: {}".format(
        donut_dataDF.get_value(index=female_index, col='Count')))
    print("Mean age: {}".format(product_data['age'].mean()))

    grid = gridplot([[diag_bar, donut, hist],
        [diag_data_table, sex_data_table, age_data_table]])

    reset_output()
    output_file('dashboards/Product' + str(product_code) + 'Dashboard.html')
    show(grid)


def diagnosis_vs_disposition_cooccurrence(data, columns):

    data = data.loc[:, columns]
    data = data.applymap(lambda x: x.replace(",", ""))
    data = data.applymap(lambda x: x.replace(" ", ""))
    data = data.applymap(lambda x: x.replace("\n", ""))

    diagnosis = data['Diagnosis'].unique()
    disposition = data['Disposition'].unique()
    cooccurrenceDF = pd.DataFrame(list(IT.product(diagnosis, disposition)))
    cooccurrenceDF.columns = ['Diagnosis', 'Disposition']

    diag_rate_list = []
    dispo_rate_list = []
    co_rate_list = []
    for _, row in cooccurrenceDF.iterrows():
        diag, dispo = row

        diag_filtered_data = data.loc[(data['Diagnosis']==diag)]
        diag_rate = len(diag_filtered_data)
        diag_rate_list.append(diag_rate)

        dispo_filtered_data = data.loc[(data['Disposition']==dispo)]
        dispo_rate = len(dispo_filtered_data)
        dispo_rate_list.append(dispo_rate)
        
        co_filtered_data = data.loc[(data['Diagnosis']==diag)
            & (data['Disposition']==dispo)]
        co_rate = len(co_filtered_data)
        co_rate_list.append(co_rate)

    cooccurrenceDF['Diagnosis Support'] = diag_rate_list
    cooccurrenceDF['Disposition Support'] = dispo_rate_list
    cooccurrenceDF['coSupport'] = co_rate_list
    cooccurrenceDF['Lift'] = (cooccurrenceDF['coSupport']/(len(data))) / (
        (cooccurrenceDF['Diagnosis Support']/len(data))*
        (cooccurrenceDF['Disposition Support']/len(data)))

    columns = ['Diagnosis', 'Disposition', 'coSupport'] #'Lift']
    cooccurrenceDFpivot = cooccurrenceDF.loc[:, columns]
    cooccurrenceDFpivot = cooccurrenceDF.pivot('Diagnosis', 'Disposition', 'coSupport')
    g = sns.heatmap(cooccurrenceDFpivot, linewidths=.5, annot=True, fmt='g')
    g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=90)
    g.set_yticklabels(g.yaxis.get_majorticklabels(), rotation=0)
    sns.plt.show()

    lift_columns = ['Diagnosis', 'Disposition', 'Lift']
    liftcooccurrenceDFpivot = cooccurrenceDF.loc[:, columns]
    liftcooccurrenceDFpivot = cooccurrenceDF.pivot('Diagnosis', 'Disposition', 'Lift')
    f = sns.heatmap(liftcooccurrenceDFpivot, linewidths=.5, annot=True, fmt='g')
    f.set_xticklabels(f.xaxis.get_majorticklabels(), rotation=90)
    f.set_yticklabels(f.yaxis.get_majorticklabels(), rotation=0)
    sns.plt.show()


def main():

####Read in NEISS data and turn numerical codes #######################
####into textual categories############################################

    data_files_dictionary = (
        csv_files_to_dataframes('projectPacket/', suffix=".csv" ) )

    dispositionDF = data_files_dictionary['Disposition.csv'].reset_index()
    neiss2014DF = data_files_dictionary['NEISS2014.csv']
    diagnosisCodes = data_files_dictionary['DiagnosisCodes.csv'].reset_index()
    bodyPartDF = data_files_dictionary['BodyParts.csv'].reset_index()

    neiss2014DF = neiss2014DF.merge(bodyPartDF, how='left',
        left_on='body_part', right_on='Code')
    neiss2014DF = neiss2014DF.merge(diagnosisCodes, how='left',
        left_on='diag', right_on='Code')
    neiss2014DF = neiss2014DF.merge(dispositionDF, how='left',
        left_on='disposition', right_on='Code')

    important_columns = ['trmt_date', 'age', 'sex', 'race', 'location', 'prod1', 'prod2',
               'narrative', 'BodyPart', 'Diagnosis', 'Disposition']

    categorical_columns = ['location', 'prod1', 'prod2']

    neiss2014DF = neiss2014DF.loc[:, important_columns]

    #if age is over 200 it means they are under 2
    #capturing the age 0-23 months with the number 1
    neiss2014DF['age'].loc[neiss2014DF['age']>=200] = 1

    #if a second product wasn't involved, fill with code 0
    neiss2014DF['prod2'].fillna('0', inplace=True)

    #turn remaining numerical codes into text
    neiss2014DF.loc[:,categorical_columns] = neiss2014DF.loc[: ,
        categorical_columns].astype(int).astype(str)

####Build solar correlation chart######################################
    correlation_columns = ['age', 'sex', 'race', 'location', 'BodyPart',
                           'Diagnosis', 'Disposition']
    
    column_solar_corr(neiss2014DF, correlation_columns,
        input_csv='solarCorrelation/solar_data.csv',
        sun='age',
        image_path='solarCorrelation/solar_image_age.png')

####Build a frequency Bar Chart based on a category####################
    build_bar_chart(neiss2014DF, 'BodyPart')

####Build a dashboard to visualize product incident data###############
    product_incident_summary(neiss2014DF, product_code=['1333'])

####Build seaborn heatmaps to visualize diagnosis vs disposition#######
    diagnosis_vs_disposition_cooccurrence(neiss2014DF,
        columns=['Diagnosis', 'Disposition'])

if __name__ == '__main__':
    main()