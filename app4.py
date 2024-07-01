from flask import Flask, request, jsonify, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import instaloader
from instagram_private_api import Client, ClientError
import pickle

app = Flask(__name__, template_folder="templates/")

# Function to handle Instaloader session and cookies
def setup_instaloader_session(username, password, cookies_file):
    L = instaloader.Instaloader()
    
    # Attempt to load existing session cookies
    try:
        with open(cookies_file, 'rb') as f:
            cookies = pickle.load(f)
            L.context._session.cookies.update(cookies)
            print("Session cookies loaded successfully.")
    except FileNotFoundError:
        print("No existing cookies file found.")
    except Exception as e:
        print(f"Error loading cookies: {e}")

    # If no cookies loaded, perform login and save cookies
    if not L.context.is_logged_in:
        try:
            L.context.login(username, password)
            print("Login successful!")
            
            # Save session cookies to file
            with open(cookies_file, 'wb') as f:
                pickle.dump(L.context._session.cookies, f)
                print("Session cookies saved successfully.")
                
        except instaloader.exceptions.InvalidArgumentException as e:
            print(f"Invalid arguments: {e}")
        except instaloader.exceptions.BadCredentialsException as e:
            print(f"Bad credentials: {e}")
        except instaloader.exceptions.ConnectionException as e:
            print(f"Connection error: {e}")

    return L

# Set Instagram credentials and cookies file
username = 'your_products100'
password = 'Qweasd@9016'
cookies_file = 'instagram_session_cookies.pkl'

# Initialize Instaloader session
L = setup_instaloader_session(username, password, cookies_file)

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get media ID from shortcode
def get_media_id_from_shortcode(shortcode):
    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        return post.mediaid
    except Exception as e:
        print(f"Failed to get media ID: {e}")
        return None

# Function to predict sentiment of Instagram comments using VADER
def predict_sentiment_post(shortcode, max_comments=10):
    media_id = get_media_id_from_shortcode(shortcode)

    if media_id:
        api = Client(username, password)

        comments = []
        try:
            comment_response = api.media_n_comments(media_id, n=max_comments)
            for comment in comment_response:
                text = comment['text']
                # Analyze sentiment using VADER
                vs = analyzer.polarity_scores(text)
                # Determine sentiment label based on compound score
                if vs['compound'] >= 0.05:
                    predict = 1  # Positive
                elif vs['compound'] <= -0.05:
                    predict = 0  # Negative
                else:
                    predict = -1  # Neutral or mixed

                comments.append({'text': text, 'predict': predict})
        except ClientError as e:
            print("Error occurred:", e)

        total = len(comments)
        positive = sum(1 for comment in comments if comment['predict'] == 1)
        negative = total - positive
        return positive, negative, total, comments

    return 0, 0, 0, []

# Routes definitions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    shortcode, max_comments = list(request.form.values())
    positive, negative, total, comments = predict_sentiment_post(shortcode, int(max_comments))
    content = {
        "prediction_text": 'Your comments are {:.2f}% positive'.format(positive / total * 100),
        "negative": negative,
        "positive": positive,
        "comments": comments
    }
    return render_template('index.html', content=content)

@app.route('/results', methods=['POST'])
def results():
    dict_json = request.get_json(force=True)
    positive, negative, total, comments = predict_sentiment_post(dict_json['short_code'], int(dict_json['max_comments']))
    return jsonify({'positive': positive, 'negative': negative, 'total': total, 'comments': comments})

if __name__ == "__main__":
    app.run(debug=True)
