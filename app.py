from flask import Flask,render_template,request,Response
from functions import *
from animate import draw_stick_figure
import turtle
import time as time
from live import captureVideoDataPred
from record import inprompt
app=Flask(__name__)
@app.route("/")
async def index():
    return render_template("index.html")

ALLOWED_EXTENSIONS=["mp4"]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/main")
async def main():
    return render_template("video.html")
    
@app.route('/video_feed')
async def video_feed():
    try:
        return Response(captureVideoDataPred(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    except RuntimeError:
        return render_template("index.html")

@app.route('/upload',methods=["POST","GET"])
async def upload():
    try:
        if request.method=="POST":
            if "video" not in request.files:
                return "No video file found"
            video=request.files['video']
            if video.filename=="":
                return "No video selected"
            if video and allowed_file(video.filename):
                good_good,vals=kmeans("good.mp4",video.filename,[])
                for i in range(len(good_good)):
                    good_good[i].pop()
                    print("The angles needed to be changed are:",good_good)
                    print("Actual angles are:",vals)
                    for i in range(len(vals)):
                        if len(vals[i][0])==9:
                            vals[i][0].pop()
                        if len(vals[i][1])==9:
                            vals[i][1].pop()
                            angles1=vals[i][0]
                            angles2=vals[i][1]
                        draw_stick_figure(-150, 0, angles1,False)
                        draw_stick_figure(150, 0, angles2,True)
                        turtle.clearscreen()
    except turtle.Terminator:
        return render_template("index.html")
    return render_template("upload.html")

@app.route("/posture",methods=['GET','POST'])
async def posture():
    if request.method=="POST":
        if "video" not in request.files:
            return "No video file found"
        video=request.files['video']
        if video.filename=="":
            return "No video selected"
        if video and allowed_file(video.filename):
            inprompt(video.filename)
    return render_template("posture.html")

if __name__=="__main__":
    app.run()