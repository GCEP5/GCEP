import pandas as pd
import numpy as np
import functions


def test_add_unique_identifier():
    test_df = pd.DataFrame({'chiller_number': [1, 2, 3],
                            'company_number': [1, 2, 3]})
    df = add_unique_identifier(test_df)
    assert df.columns[2] == 'unique_identifier', \"Create identifier failed"
    assert np.all(df['unique_identifier'].values ==
                    ['11', '22', '33']), \"Identifier error"


def test_separate_df(df, string):
    test_df = pd.DataFrame({'chiller_number': [2, 2, 3],
                           'company_number': [1, 2, 3]})
    dfs = separate_df(test_df, 'chiller_number')
    assert len(dfs) = len(test_df['chiller_number'].unique()), "Append error"


def test_polynomial_fit():
    test_df = pd.DataFrame({'lift': [1, 2, 3, 4, 5, 6],
                            'load': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                            'efficiency': [2, 6, 12, 20, 30, 42]})
    model = polynomial_fit(test_df)
    expect = [-1.70339580e-07, 6.76896357e-08, 1.09927969e+00,
              9.09686645e-01, 9.09686373e-01, 1.84607289e-07]
    assert len(model) == 6, "form of function error"
    np.testing.assert_almost_equal(model, expect, err_msg='Bad Fit')


def test_error():
    test_df = pd.DataFrame({'lift': [1, 2, 3, 4, 5, 6],
                            'load': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                            'efficiency': [2, 6, 12, 20, 30, 42]})
    model = [-1.70339580e-07, 6.76896357e-08, 1.09927969e+00,
             9.09686645e-01, 9.09686373e-01, 1.84607289e-07]
    error, predictions, actuals = error(test_df, model)
    assert error == 7.440198766693103e-08, "Calculation error"
    assert len(predictions) == len(actuals), "Broken"


def test_fxn():
    test_df = pd.DataFrame({'lift': [1, 2, 3, 4, 5, 6],
                            'load': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                            'efficiency': [2, 6, 12, 20, 30, 42]})
    y = df['lift'].copy()
    x = df['load'].copy()
    a1, a2, a3, a4, a5, a6 = [-1.70339580e-07, 6.76896357e-08, 1.09927969e+00,
                              9.09686645e-01, 9.09686373e-01, 1.84607289e-07]
    result = fxn((x, y), a1, a2, a3, a4, a5, a6)
    expect = [2, 6, 12, 20, 30, 42]
    np.testing.assert_almost_equal(result, expect, err_msg='wrong funtion')


def test_given_chiller_curve_vs():
    result = given_chiller_curve_vs(2, 3)
    expect = 2.5147451175
    np.testing.assert_almost_equal(
                                   result, expect, err_msg="Given parameters"
                                   )
