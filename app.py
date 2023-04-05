import os
import cv2
import io
import json
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import vision
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, StringField
from openai import OpenAI
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = "your-secret-key"
app.config['UPLOAD_FOLDER'] = './uploaded_videos'

# Initialize OpenAI API
OpenAI.api_key = app.config['OPENAI_API_KEY']

class VideoUploadForm(FlaskForm):
    video_file = FileField('Upload Video', validators=[FileRequired(), FileAllowed(['mp4'], 'Videos only!')])
    submit = SubmitField('Upload')

class QueryForm(FlaskForm):
    question = StringField('Ask a question:', validators=[])
    submit = SubmitField('Submit')

def process_video(video_path):
    # Extract frames and audio from the video
    video = cv2.VideoCapture(video_path)
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    # Initialize Google Cloud clients
    speech_client = speech.SpeechClient()
    vision_client = vision.ImageAnnotatorClient()

    text_data = []
    audio_data = []

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        if frame_count % frame_rate == 0:  # Process one frame per second
            # Image recognition
            _, buffer = cv2.imencode('.jpg', frame)
            image = vision.Image(content=buffer.tobytes())
            response = vision_client.text_detection(image=image)
            texts = response.text_annotations
            if texts:
                text_data.append(texts[0].description)

        frame_count += 1

    # Audio recognition
    video.release()
    audio = AudioSegment.from_file(video_path, "mp4")
    audio.export("audio.wav", format="wav")

    with io.open("audio.wav", "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = speech_client.recognize(config=config, audio=audio)

    for result in response.results:
        audio_data.append(result.alternatives[0].transcript)

    return "\n".join(text_data), "\n".join(audio_data)

def ask_gpt(question, context):
    response = OpenAI.Completion.create(
        engine="text-davinci-002",
        prompt=f"{question}\n\n{context}",
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    form = VideoUploadForm()
    if form.validate_on_submit():
        video_file = form.video_file.data
        video_filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video_file.save(video_path)
        text_data, audio_data = process_video(video_path)
        return redirect(url_for('ask_question', text_data=text_data, audio_data=audio_data))
    return render_template('upload_video.html', form=form)

@app.route('/ask_question', methods=['GET', 'POST'])
def ask_question():
    text_data = request.args.get('text_data')
    audio_data = request.args.get('audio_data')
    form = QueryForm()
    answer = ''
    if form.validate_on_submit():
        question = form.question.data
        context = f"Text data: {text_data}\nAudio data: {audio_data}"
        answer = ask_gpt(question, context)
    return render_template('ask_question.html', form=form, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
