# SyntheticAlchemy - Video Q&A with AI
Have an AI watch and listen to a video for you so you can ask questions to it!


<img src="./static/images/logo.png" alt="Logo" align="center" width="200" />

SyntheticAlchemy is a web application that allows users to upload a video and ask questions about its content. Audio is transcribed with OpenAIs Whisper, and Tesseract is used to transcribe text shown on screen in the videos. To prevent oversaturation of text from translating multiple frames per second, I just limited it to 1 frame per video second.

*Note:* Make sure to install ffmpeg before running app.py

*Note:* Currently submitting YouTube URLs are not working, but they are in the works, I just need to use a different python module or version...

Plans:
- [x] Move from Google Cloud to self hosting for transcription services to save costs
- [x] DistilBERT to Vectorize/Rank pieces of relevant information related to the question prior to submittion to OpenAI to reduce token count and cost.
- [ ] YouTube URL support (in progress)
- [ ] Redis database to store queries and transcriptions (in progress) to eliminate reprocessing of videos.
- [ ] Switch out DistilBERT for OpenAIs embedding, will greatly reduce time to search through large amounts of video related data (best for long videos)
- [ ] ElevenLabs API support
     - [ ] Play back the responses to your questions with a realistic voice!
     - [ ] For videos that are sufficiently long enough support the responses to be played back with their voice 
- [ ] Adding/storing API keys via web interface
- [ ] Speech-To-Text, to talk to your video, and have it answer you back.

![Example Usage](example_usage.png)

## Features

- Video content analysis using OpenAI Whisper, and Tesseract. (GPU Recommended)
- Natural language processing for answering questions with OpenAI's GPT-3
- Simple web interface for video uploads and question submission

## Installation

1. Clone the repository:

`git clone https://github.com/0xdeadcell/SyntheticAlchemy.git`


2. Install the required packages:

`pip install -r requirements.txt`


3. Set up API keys:

- Sign up for an API key from OpenAI and replace `"your-openai-api-key"` in `app.py` with your actual API key.


## Usage

1. Start the Flask server:

`python app.py`

2. Open a web browser and visit `http://127.0.0.1:5000/` to access the application.

3. Upload a video in MP4 format, then ask questions about its content.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
