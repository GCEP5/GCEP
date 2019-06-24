import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.neural_network import MLPRegressor
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools


def add_unique_identifier(df):
    df['chiller_number'] = df['chiller_number'].astype(str)
    df['company_number'] = df['company_number'].astype(str)
    # this column is a string
    df['unique_identifier'] = df['chiller_number'] + df['company_number']
    return df


def single_polynomial_plot(df, temp, title):
    """input a single dataframe which has been processed by both poly_model_wraper
        and add_unique_identifier functions and ouptuts graph of the polynomial
        chiller curve at the specified temp with the specified title."""
    df = df[(df['lift'] > temp - 1) & (df['lift'] < temp + 1)]
    x = df['load']
    y = df['lift']
    X = (x, y)
    # assuming the last column is already the unique identifier column
    a1, a2, a3, a4, a5, a6 = df[df.columns[-7:-1]].iloc[0]
    plt.scatter(x, df['efficiency'], alpha=.5)
    # dummy load vals to be put into function to plot curve
    curve_load_vals = np.arange(0, 1.0, 0.0001)
    curve = fxn((curve_load_vals, temp), a1, a2, a3, a4, a5, a6)
    plt.plot(curve_load_vals, curve)
    plt.xlabel('Load %')
    plt.ylabel('Efficiency (kW/Ton)')
    plt.title(title)
    plt.show()
    return


def chiller_curve_parity(df):
    curve = given_chiller_curve(df['load'], df['lift'])
    actual = df['efficiency']
    plt.scatter(actual, curve, alpha=0.5, marker='.')
    y = np.arange(0, 1, .001)
    plt.plot(y, y)
    plt.xlabel('Actual Efficiency (kW/Ton)')
    plt.ylabel('Predicted Efficiency (kW/Ton)')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.title('Chiller ' + str(df['unique_identifier'].unique()
                               [0]) + ' Current Model Parity Plot')
    plt.show()
    return


def poly_parity_plot(df):
    """assumes a preprocessed single dataframe input"""
    plt.scatter(df['efficiency'], df['prediction'], alpha=.5, marker='.')
    y = np.arange(0, 1, .001)
    plt.plot(y, y)
    plt.xlabel('Actual Efficiency (kW/Ton)')
    plt.ylabel('Predicted Efficiency (kW/Ton)')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.title('Chiller ' + str(df['unique_identifier'].unique()
                               [0]) + ' Polynomial Model Parity Plot')
    plt.show()
    return


def separate_df(df, string):
    dfs = []
    for x in df[string].unique():
        dfs.append(df[df[string] == x].copy())
    return dfs


def polynomial_fit(df):
    a = np.ones(6)
    y = df['lift'].copy()
    x = df['load'].copy()
    z = df['efficiency'].copy()
    model = curve_fit(fxn, (x, y), z, a)
    return model[0]


def error(df, model):
    actuals = df['efficiency'].copy()
    y = df['lift'].copy()
    x = df['load'].copy()
    a1, a2, a3, a4, a5, a6 = model
    predictions = fxn((x, y), a1, a2, a3, a4, a5, a6)
    error = np.sqrt(mean_squared_error(predictions, actuals))
    return error, predictions, actuals


def fxn(X, a1, a2, a3, a4, a5, a6):
    # load is x, lift is y
    x, y = X
    return (a1 * x**2 + a2 * x + a3) * (a4 * y**2 + a5 * y + a6)


def poly_model_wrapper(df):
    company_dfs = separate_df(df, 'company_number')
    chiller_dfs = []
    counter = 1
    for x in company_dfs:
        chiller_dfs.append(separate_df(x, 'chiller_number'))
        counter += 1
    processed_dfs = []
    for x in chiller_dfs:
        for y in x:
            # try except is here because if
            # the polynomial fit fxn cant
            # find a fit within 1400 iterations,
            # it will pass an error, could
            # be due to a number of things
            # not enough data, bad data, etc.
            # so I disclude all of the data that
            # cannot be fit with a polynomial
            # in this way.
            try:
                m = polynomial_fit(y)
                e = error(y, m)
                y['error'] = e[0]
                y['predictions'] = e[1]
                counter = 1
                for i in m:
                    y['a' + str(counter)] = i
                    counter += 1
                processed_dfs.append(y)
            except:
                continue
    return pd.concat(processed_dfs)


def given_chiller_curve_vs(chiller_load, LIFT):
    X_factor = 5.39*(chiller_load)**2 - 5.5155*(chiller_load) + 2.5533
    Y = .000143*LIFT**2 + 0.000954*LIFT + 0.188076
    return X_factor * Y


def linear_model_wrapper(df):
    company_dfs = separate_df(df, 'company_number')
    chiller_dfs = []
    for x in company_dfs:
        chiller_dfs.append(separate_df(x, 'chiller_number'))
    processed_dfs = []
    for x in chiller_dfs:
        for y in x:
            m = LinearRegression()
            m.fit(y[['lift', 'load']], y['efficiency'])
            y['linear_model'] = m
            y['error'] = np.sqrt(mean_squared_error(y['efficiency'],
                                                    m.predict(y[['lift',
                                                                 'load']])))
            processed_dfs.append(y)
    return pd.concat(processed_dfs)


def predictors_properties(df):
    df['chiller_number'] = df['chiller_number'].astype(str)
    df['company_number'] = df['company_number'].astype(str)
    # this column is a string
    df['unique_identifier'] = df['chiller_number'] + df['company_number']
    y = df['unique_identifier'].unique()
    predictors = pd.DataFrame(columns=['motor', 'evap_bundle', 'cond_bundle'])
    properties = pd.DataFrame(columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
    predictors[predictors['cond_bundle'] == 0] = 1
    for x in y:
        predictors = predictors.append(df[df['unique_identifier'] == x].
                                       iloc[0][['motor', 'evap_bundle',
                                                'cond_bundle']])
        properties = properties.append(df[df['unique_identifier'] == x].
                                       iloc[0][list(properties.columns)])
    return predictors, properties


def get_motor_codes(file):
    motor_codes = pd.read_csv(file)
    motor_codes.set_index('MOTOR CODE', inplace=True)
    return motor_codes


def get_bundle_dfs(codes):
    # assuming file names are of the form: manufacturer_(evap or cond)_bundles
    # _(minimum flow unit)_(design level).csv
    # for example: yk_evap_bundles_gpm_G.csv
    evap_dfs = []
    cond_dfs = []
    for x in codes:
        i = pd.read_csv('yk_evap_bundles_gpm_' + x + '.csv')
        y = pd.read_csv('yk_cond_bundles_gpm_' + x + '.csv')
        for z in [i, y]:
            z.set_index('MODEL', inplace=True)
        evap_dfs.append(i)
        cond_dfs.append(y)
    return evap_dfs, cond_dfs


def get_relevant_data(df, company_number):
    """Takes an input dataframe in the format
        of the excel files given to us for this
        project and outputs a list of dataframes
        (one for each chiller) with only
        the columns we need."""
    timestamp = df['timestamp']
    df = df[df.commfail != 1.0]
    cols = chiller_numbers(df)
    necessary_data = ['SBoolean', 'CDT', 'EVT',
                      'Load', '_kW/Ton']
    total_data = []
    for column in cols:
        new_cols = [column + key for key in necessary_data]
        for item in df.columns:
            if item[:3] == column and item[-2:] == 'Hz':
                new_cols.append(item)
                hz = False
        new_df = df[new_cols].copy()
        new_df['chiller_number'] = column
        new_df['company_number'] = company_number
        total_data.append(new_df)
    return total_data, timestamp


def chiller_numbers(df):
    """Takes a dataframe and returns a list of the chiller numbers for said
        dataframe(a single .csv file."""
    # this needs to be fixed to account for 2 digit chiller numbers
    cols = [c for c in df.columns if (c[2].isdigit() and c[:2] == 'CH') or
            (c[2:4].isdigit() and c[:2] == 'CH')]
    cols = [c[:3] for c in cols]
    cols = pd.Series(cols).unique()
    return cols


def extract_useful_data(data, motor_df,
                        evap_df, cond_df,
                        model_numbers,
                        company_number):
    """Wrapper function that takes the raw data
        (in a pandas dataframe), along with the
        motor code, evaporator code, and condensor code
        information, as well as the model numbers for each
        chiller present in the orignal .csv file and
        the company number, then returns a dataframe with
        all of the cleaned relevant data for the model for a
        single .csv file."""
    array, timestamp = get_relevant_data(data,
                                         company_number)
    clean_data(array)
    add_model_number_terms(array, motor_df, evap_df, cond_df,
                           model_numbers, company_number)
    df = pd.concat(array)
    return df


def clean_data(array):
    """Drops the rows in each dataframe in the passed
        array that fit the criteria for a bad measurement.
        This includes when the chiller is off, experiences
        communication failure, or has a NaN value in a row,
        then drops the columns no longer needed after this
        action(i.e. ALARM, and SBoolean.)
        input: array of dfs
        output: none, does work inplace"""
    # renamed columns so each dataframe can be appended to each other
    counter = 1
    for df in array:
        df.drop(df[df[df.columns[1]] == 0.0].index, inplace=True)
        df.dropna(inplace=True)
        df.insert(loc=3, column='lift',
        value=(df[df.columns[1]] - df[df.columns[2]]))
        df.drop(df.columns[:3], axis=1, inplace=True)
        try:
            df.columns = ['lift', 'load', 'efficiency', 'hz',
                          'chiller_number', 'company_number']
        except:
            df.columns = ['lift', 'load', 'efficiency',
                          'chiller_number', 'company_number']
            df.insert(loc=3, column='hz', value=0)
            df['chiller_number'] = counter
            df.drop(df[df['efficiency'] == 0.0].index, inplace=True)
            df.drop(df[df['efficiency'] > 1.0].index, inplace=True)
            counter += 1
    return


def grab_data_paths(path):
    """Takes data from the GCEP_5/gcep_data folder, assuming
        it only contains folders of .csv files for each
        company. Make sure the path you input is a string of the
        ABSOLUTE path to the GCEP_5/gcep_data folder on
        your local device.
        For example: 'C:/Users/copse/GCEP_5/gcep_data'
        Returns a list of lists, with each list representing
        each company number in order.
        For this function to line up with your model numbers list,
        they must be ordered from smallest number to largest number
        company. E.g. company1, company2, company3, etc.
        make sure your directory to the gcep_data only has one folder
        for each company containing only the .csv raw data."""
    # filenames
    names = []
    # file path for the /gcep_data folder, assuming separate folder
    # for each company
    folders = os.listdir(path)
    for folder in folders:
        string = path + '/' + folder
        temp = os.listdir(string)
        temp1 = []
        for name in temp:
            temp1.append(path + '/' + folder + '/' + name)
        names.append(temp1)
    return names


def add_model_number_terms(array, motor_df,
                           evap_dfs, cond_dfs,
                           model_numbers, company_number):
    """Takes an array of dfs(one for each csv file),
        information about the motor code, evap code,
        cond code and design level, as well as the
        company number, then adds all the necessary
        features """
    # add some sort of dictionary of all the possible
    # keys and their corresponding values
    i = 0
    codes = {'D': 0, 'E': 1, 'F': 2, 'G': 3}
    for df in array:
        design_level, cond_code, evap_code,
        motor_code = split_model_number(model_numbers[i])
        df['design_level'] = design_level
        # below is for when we start investigating impact of motor,
        # evap and cond codes, etc.
        # assuming motor code column of interest is named KW(MAX)
        df['motor'] = motor_df.loc[motor_code]['KW(MAX)']
        # index depends on the number of passes evap_dfs[codes[design_level]].
        # loc[evap_code][index]
        # must incorporate number of passes for some of the dataframes
        df['evap_bundle'] = int(evap_dfs[codes[
                                               design_level]].loc[evap_code][1])
        df['cond_bundle'] = int(cond_dfs[codes[design_level]].loc[cond_code][1])
        # for debugging purposes
        df['model_number'] = model_numbers[i]
        # eventually this must be dynamic
        df['passes'] = 2
        i += 1
    return array


def split_model_number(model_number):
    """Splits the model number into its important
        parts and returns each(not stored in any structure)"""
    # also need to add number of passes in here, default = 2
    evap_code = model_number[2:4]
    cond_code = model_number[4:6]
    motor_code = model_number[9:11]
    design_level = model_number[11]
    # evap and cond codes are referring to their respective bundles(min flows)
    return design_level, cond_code, evap_code, motor_code


def split_model_number(model_number):
    """Splits the model number into its important
        parts and returns each(not stored in any structure)"""
    # also need to add number of passes in here, default = 2
    evap_code = model_number[2:4]
    cond_code = model_number[4:6]
    motor_code = model_number[9:11]
    design_level = model_number[11]
    # evap and cond codes are referring to their respective bundles(min flows)
    return design_level, cond_code, evap_code, motor_code


def test_add_features():
    data = {'column1': [1, 2, 3], 'column2': [1, 3, 5], 'column3': [2, 4, 6]}
    df = pd.DataFrame(data)
    df2 = pd.DataFrame(data)
    assert len(df.columns) == 3
    features = (('ln', np.log), ('exp', np.exp))
    add_features(df, features, False)
    add_features(df2, features, True)
    assert len(df.columns) == 9
    assert len(df2.columns) == 12
    assert df.iloc[0]['column1'] == 1
    assert df.iloc[0]['ln_column1'] == 0
    assert type(df) == pd.DataFrame
    return


def read_all_csv(files, model_numbers,
                 company_number,
                 motor_codes,
                 evap_codes, cond_codes):
    """Reads all the csv files for one company,
        cleans data, adds all relevant terms for
        model and returns a new df of all the raw
        data for all the csv files for one company
        (one company folder,
        e.g. GCEP_5/gcep_data/company1)"""
    array = [[] for x in range(len(files))]
    for i, file in enumerate(files):
        array[i] = extract_useful_data(pd.read_csv(file),
                                       motor_codes,
                                       evap_codes,
                                       cond_codes,
                                       model_numbers,
                                       company_number)
    return pd.concat(array)


def read_all_folders(names,
                     model_numbers,
                     company_numbers,
                     motor_codes,
                     evap_codes,
                     cond_codes):
    """Grabs all the .csv files for one company and
        passes to read_all_csv for further processing
        then repeats for all companies in the /gcep_data
        folder."""
    array = [[] for x in range(len(names))]
    for i, files in enumerate(names):
        array[i] = read_all_csv(files, model_numbers[i], company_numbers[i],
                                motor_codes, evap_codes, cond_codes)
    return pd.concat(array)
