import os
import cv2
import io
import json
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import vision
import markdown
from flask import Flask, render_template, request, redirect, url_for, flash, Markup, session, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, StringField, SelectField
import openai
from werkzeug.utils import secure_filename
from difflib import SequenceMatcher


app = Flask(__name__)
app.config['SECRET_KEY'] = "your-secret-key"
app.config['UPLOAD_FOLDER'] = './uploaded_videos'

# Initialize OpenAI API
# Add the key to your env with the command: export OPENAI_API_KEY=your-key or set OPENAI_API_KEY=your-key
openai.api_key = os.getenv('OPENAI_API_KEY')
print("Available OpenAI Engines: \n", openai.Engine.list())


@app.template_filter('markdown')
def markdown_filter(text):
    return Markup(markdown.markdown(text))


class VideoUploadForm(FlaskForm):
    video_file = FileField('Upload Video', validators=[FileRequired(), FileAllowed(['mp4'], 'Videos only!')])
    submit = SubmitField('Upload')
    

class QueryForm(FlaskForm):
    question = StringField('Question', validators=[])
    submit = SubmitField('Submit')


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


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
    similarity_threshold = 0.7
    
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
                current_text = texts[0].description
                if not text_data or similar(text_data[-1], current_text) < similarity_threshold:
                    text_data.append(current_text)

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
        language_code="en-US",
        audio_channel_count = 2,
    )

    response = speech_client.recognize(config=config, audio=audio)

    for result in response.results:
        audio_data.append(result.alternatives[0].transcript)

    return "\n".join(text_data), "\n".join(audio_data)

def ask_gpt(question, context):
    print("Asking GPT...")
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Max tokens: 4000")

    if len(question)*2 > 4000:
        print("Question is too long. Please ask a shorter question... Only using the first 512 characters")
    question = question[:512]

    if len(context)*2 > 4000:
        print("Context is too long. Please provide a shorter context... Only using the first 1700 characters")
        context = context[:1700]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant who has been provided text information from a video that has converted audio-visual to text. If asked to provide the summary or text, please provide it, but fix any grammatical or formatting errors:"}, {"role": "user", "content": context}, {"role": "user", "content": question}],
        max_tokens=1500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    print("GPT-4 Response: ", response)
    return [i['message']['content'] for i in response['choices']][0] if response['choices'] else "No answer found"


@app.route('/uploaded_videos/<path:filename>')
def uploaded_videos(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def upload_video():
    form = VideoUploadForm()

    if form.validate_on_submit():
        video_file = form.video_file.data
        video_filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video_file.save(video_path)
        text_data, audio_data = process_video(video_path)
        video_url = url_for('uploaded_videos', filename=video_filename)  # Add this line
        return redirect(url_for('ask_question', text_data=text_data, audio_data=audio_data, video_url=video_url))  # Add video_url here
    return render_template('upload_video.html', form=form)


@app.route('/ask_question', methods=['GET', 'POST'])
def ask_question():
    text_data = request.args.get('text_data')
    audio_data = request.args.get('audio_data')
    video_url = request.args.get('video_url')  # Add this line
    form = QueryForm()
    answer = ''

    last_answer = request.args.get('answer', '')

    # check if the method is a get or post
    if request.method == 'GET':
        return render_template('ask_question.html', form=form, answer=last_answer, text_data=text_data, audio_data=audio_data, video_url=video_url)

    if form.validate_on_submit():
        question = form.question.data
        context = f"Text data: {text_data}\nAudio data: {audio_data}"
        answer = ask_gpt(question, context)
    return render_template('ask_question.html', form=form, answer=answer, text_data=text_data, audio_data=audio_data, video_url=video_url)


if __name__ == '__main__':
    app.run(debug=True)


