
import requests
from bs4 import BeautifulSoup
import re
from flask import Flask, Response, render_template, request
import numpy as np
from flask import send_from_directory
import os
app = Flask(__name__)



def get_all_clickable_elements(url):
    allElements = {}

    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    for tag in soup.find_all(True):
        if tag.name == 'a' or tag.name == 'button':

            tagText = tag.text.lower()
            tagText = tagText.replace("\r", " ")
            tagText = tagText.replace("\n", " ")
            tagText = re.sub('\s+', ' ', tagText)
            tagText = tagText.strip()
            if tagText != '':
                allElements[tagText] = str(tag.get('href'))
                if allElements[tagText] == '#main':
                    allElements[tagText] = ''
                if allElements[tagText] == None:
                    allElements[tagText] = ''
                if 'https://matchtv.ru' not in allElements[tagText]:
                    allElements[tagText] = 'https://matchtv.ru' + allElements[tagText]

    return allElements




def generate_front_dict(i=3):
    url = 'https://matchtv.ru/'
    allElements = get_all_clickable_elements(url)
    color_list=["#f8d3c5","#daedbd","#c8bfe7","#89c7cd","#ffffba"]
    front_dict={}
    c=1
    for i in range(i,i+5):
        front_dict[c]=[str.upper(list(allElements.keys())[i]),allElements[list(allElements.keys())[i]],color_list[c-1]]
        c=c+1
    return front_dict


signal=''
@app.route('/main/<datatype>', methods=['GET', 'POST'])
def monitoring_content_content_data(datatype):
    global signal	
    if request.method == 'POST':
        signal = str(request.data.decode("utf-8"))
       
    row = "data:%s\n\n" % (signal )
    print(row)
    return Response(row, mimetype='text/event-stream')
 
@app.route('/')
@app.route('/main')
def main_start():
    return render_template('main.html')


if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', threaded=True)

