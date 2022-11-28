import os
from flask import Flask, request
from flask.templating import render_template
from yolo import letMeSee


# import sys
# , Response, escape, g, make_response
# from werkzeug.utils import secure_filename


realPath = os.path.dirname(os.path.realpath(__file__))
subPath = os.path.split(realPath)[0]
os.chdir(subPath)

app = Flask(__name__)
app.debug = True


def root_path():
    # root 경로 유지
    realPath = os.path.dirname(os.path.realpath(__file__))
    subPath = "\\".join(realPath.split("\\")[:-1])
    return os.chdir(subPath)

# Main page


@app.route('/getVideo')
def getVideo():
    return render_template('upload.html')

# object Detection and Post


@app.route('/postVideo', methods=['GET', 'POST'])
def postVideo():
    if request.method == 'POST':
        root_path()
        # User Video
        userVideo = request.files['videoFile']
        print(userVideo.filename)
        userVideo.save('./flask_barbell/static/video/'+str(userVideo.filename))
        userVideoPath = '/Users/LG/web-dnn/flask_barbell/static/video/' + \
            str(userVideo.filename)

        userVideoType = str(userVideo.filename).split('.')[1]

        if userVideoType in ['webm', 'mp4', 'MP4', 'avi', 'AVI', 'wmv', 'WMV', 'mkv', 'MKV', 'mov', 'MOV', 'MPEG', 'WEBM']:
            detectedVideo = letMeSee.main(userVideoPath)
            detectedVideoPath = 'video/' + str(detectedVideo.split('/')[-1])
            print(detectedVideoPath)
        else:
            print('Check if a file is a video')

    return render_template('postResult.html', detectedVideo=detectedVideoPath)


app.run(debug=True, host='0.0.0.0')
