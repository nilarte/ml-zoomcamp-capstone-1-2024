#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

employee_id = 'xyz-123'
# employee = {
#     "education": "Bachelors",
#     "joiningyear": 2017,
#     "city": "Bangalore",
#     "paymenttier": 3,
#     "age": 34,
#     "gender": "Male",
#     "everbenched": "No",
#     "experienceincurrentdomain": 0,
# }
employee = {
    "education": "Bachelors",
    "joiningyear": 2013,
    "city": "Pune",
    "paymenttier": 1,
    "age": 28,
    "gender": "Female",
    "everbenched": "No",
    "experienceincurrentdomain": 3,
}



response = requests.post(url, json=employee).json()
print(response)

if response['leaveornot'] == True:
    print('High probability of attrition for employee %s' % employee_id)
else:
    print('No high probability of attrition for employee %s' % employee_id)
