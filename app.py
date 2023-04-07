import random
import string
import cv2
import re
import json
import os
import asyncio
import youtube_dl
import googleapiclient.discovery
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from google.cloud import vision
import markdown
from flask import Flask, render_template, request, redirect, url_for, flash, Markup, session, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, StringField, SelectField
import openai
import io
import whisper
import time
from werkzeug.utils import secure_filename
import validators
from difflib import SequenceMatcher
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from concurrent.futures import ThreadPoolExecutor
from similarity import find_top_k_similar, filter_paragraphs, calculate_similarity
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load OpenAI Whisper model
model = whisper.load_model("base", device=device)

app = Flask(__name__)
app.config['SECRET_KEY'] = "your-secret-key"
app.config['UPLOAD_FOLDER'] = './uploaded_videos'

# Initialize OpenAI API
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        data = f.read().strip()
        assert data is not None
        openai.api_key = data
else:
    openai.api_key = os.getenv('OPENAI_API_KEY')

@app.template_filter('markdown')
def markdown_filter(text):
    return Markup(markdown.markdown(text))

class VideoURLForm(FlaskForm):
    video_url = StringField('Enter YouTube Video URL', validators=[])
    submit = SubmitField('Process Video')


class VideoUploadForm(FlaskForm):
    video_file = FileField('Upload Video', validators=[FileAllowed(['mp4'], 'Videos only!')])
    submit = SubmitField('Upload')
    

class QueryForm(FlaskForm):
    question = StringField('Question', validators=[])
    submit = SubmitField('Submit')

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def download_youtube_video(video_url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': os.path.join(app.config['UPLOAD_FOLDER'], '%(title)s.%(ext)s'),
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        video_filename = ydl.prepare_filename(info)
    return video_filename

def get_youtube_transcript(video_id, language="en"):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])
    results = youtube.captions().list(part="snippet", videoId=video_id).execute()
    transcript = None
    for item in results["items"]:
        if item["snippet"]["language"] == language:
            transcript_id = item["id"]
            transcript = youtube.captions().download(id=transcript_id).execute()
            break
    return transcript

def process_video(video_path, similar=0.6):
    text_data = []
    audio_data = []

    if not (validators.url(video_path) or os.path.exists(video_path)):
        raise ValueError("Invalid URL or local file path provided")

    if validators.url(video_path):
        # Download the video
        ydl_opts = {'outtmpl': os.path.join(app.config['UPLOAD_FOLDER'], '%(title)s.%(ext)s')}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_path, download=True)
            video_filename = ydl.prepare_filename(info)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

    video = cv2.VideoCapture(video_path)
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    # Concurrently process audio and text data
    def process_text(frame_path):
        # Use pytesseract for image extraction
        text = pytesseract.image_to_string(frame_path)
        return text

    def process_audio(audio_path):
        # Transcribe the audio using OpenAI Whisper
        try:
            transcript = model.transcribe(audio_path)
            return transcript['text']
        except Exception as e:
            print("Error extracting audio: " + str(e))

    with ThreadPoolExecutor() as executor:
        audio_path = "audio.wav"
        AudioSegment.from_file(video_path).export(audio_path, format="wav")
        audio_data = executor.submit(process_audio, audio_path).result()

        while True:
            ret, frame = video.read()

            if not ret:
                break

            # Extract frames every 30 frames
            if frame_count % 60 == 0:
                # Save the frame to a temporary file
                frame_path = f"frame_{frame_count}.jpg"
                cv2.imwrite(frame_path, frame)

                # Process the frame using pytesseract
                texts = process_text(frame_path)

                if texts:
                    text_data.append(texts)

                # Delete the temporary file
                os.remove(frame_path)

            frame_count += 1

    video.release()
    os.remove(audio_path)

    # Remove duplicates
    _, filtered_text = filter_paragraphs(text_data, similar)

    print("Video processed successfully")
    return filtered_text, audio_data



def ask_gpt(question, context):
    #print("Asking GPT...")
    #print(f"Question: {question}")
    #print(f"Context: {context}")
    #print(f"Max tokens to request: 1000, (up to 3000 in the context + question)")

    #print(f"\n\nQuestion: {question}\n\nContext: {context}\n\n")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant who has been provided text information from a video that has converted audio-visual to text. By speaker the user is referring to 'Audio:'. If asked to provide the summary or text from the video, please provide it, but fix any grammatical or formatting errors:"}, {"role": "user", "content": context}, {"role": "user", "content": question}],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
    )

    #print("GPT-4 Response: ", response)
    return [i['message']['content'] for i in response['choices']][0] if response['choices'] else "No answer found"


@app.route('/uploaded_videos/<path:filename>')
def uploaded_videos(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def upload_video():
    urlform = VideoURLForm()
    uploadform = VideoUploadForm()

    text_data = []
    audio_data = []

    if uploadform.validate_on_submit() or urlform.validate_on_submit():
        
        if uploadform.validate_on_submit():
            video_file = uploadform.video_file.data
            video_filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            video_file.save(video_path)
        else:
            youtube_url = urlform.video_url.data
            video_filename = f"youtube_{int(time.time())}.mp4"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
                'outtmpl': video_path,
                'quiet': True,
            }

            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])

        
            if video_url.startswith('http') or video_url.startswith('https'):
                # Download the video
                video_path = download_youtube_video(video_url)
            else:
                # Assume that video_url is a file path
                video_path = video_url

    
        text_data, audio_data = process_video(video_path)
        video_url = url_for('uploaded_videos', filename=video_filename)
        session['text_data'] = text_data
        session['audio_data'] = audio_data
        session['video_url'] = video_path if video_path else video_url

        return redirect(url_for('ask_question'))
    
    return render_template('upload_video.html', uploadform=uploadform, urlform=urlform)



@app.route('/ask_question', methods=['GET', 'POST'])
def ask_question():
    
    text_data = session.get('text_data', '')
    text_data = "Text Data/OCR Video Data:\n" + str(text_data)

    audio_data = session.get('audio_data', '')
    audio_data = "Audio Data/Speech Data:\n" + str(audio_data)
    
    video_url = session.get('video_url', '')
    answer = session.get('answer', '')
    
    form = QueryForm()
    last_context = session.get('context', '')
    last_answer = session.get('answer', '')

    if request.method == 'GET':
        return render_template('ask_question.html', form=form, context=last_context, answer=last_answer, text_data=text_data, audio_data=audio_data, video_url=video_url)

    if form.validate_on_submit():
        question = form.question.data

        # Split text and audio data into paragraphs and filter out short paragraphs
        paragraphs = re.split(r'\n+', (text_data or "") + "\n" + (audio_data or ""))

        filtered_paragraphs = [p for _, p in filter_paragraphs(paragraphs)]

        # Find the most relevant paragraphs based on the similarity to the question
        top_k_similar_paragraphs = find_top_k_similar(filtered_paragraphs, question)

        # Join the most relevant paragraphs as the context for GPT
        context = "\n".join(top_k_similar_paragraphs)
        answer = ask_gpt(question, context)
        session['answer'] = answer
        session['context'] = context
        
        return render_template('ask_question.html', form=form, context=context, answer=answer, text_data=text_data, audio_data=audio_data, video_url=video_url)




if __name__ == '__main__':
    app.run(debug=True)


