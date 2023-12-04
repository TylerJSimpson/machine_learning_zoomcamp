import requests

url = 'http://localhost:9696/predict'

applicant_dict = {
    'is_joint': 0,
    'applicant_credit_score': 750,
    'coapplicant_credit_score': -1,
    'is_boat': 1,
    'is_engine': 0,
    'is_rv': 0,
    'vehicle_age': 0,
    'is_vehicle_new': 0,
    'cash_price': 78314.61,
    'trade_in_value': 0.0,
    'trade_in_payoff': 0.0,
    'doc_fee': 299.0,
    'amount_to_finance': 68978.26,
    'payment': 287.41,
    'ltv': 104.33,
    'debt': 3235.0,
    'down_payment': 8737.0,
    'income': 15626.0,
    'proposed_dti': 43.72,
    'current_dti': 38.14,
    'applicant_dti': 43.72,
    'applicant_age': 59
}

response = requests.post(url, json=applicant_dict).json()
approval_probability = response['approval_probability']

if approval_probability >= 0.5:
    print('Application Approved')
else:
    print('Application Denied')
