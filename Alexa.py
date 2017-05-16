from flask import Flask, render_template
from flask_ask import Ask, statement, question, session
from gensim import corpora
import gensim
# import pandas as pd
from nltk.corpus import stopwords
from collections import defaultdict
import re
import time
from TwitterSearch import *

SKILL_NAME = "Twitter Sustainability Topic Model"

app = Flask(__name__)

ask = Ask(app, "/")


@ask.launch
def welcome():
    intro = ("Hello, please ask about a subject in sustainability to here the topics trending about it on twitter.")
    reprompt_texts = ("Please ask about a subject in sustainability to here the topics trending about it on twitter.")

    session.attributes["speech_output"] = intro
    session.attributes["reprompt_text"] = reprompt_texts

    return question(intro).reprompt(reprompt_texts)


@ask.intent("TopicIntent")
def topic(Answer):
    attributes = {}
    should_end_session = False
    user_gave_up = False

    if not Answer and user_gave_up == "DontKnowIntent":
        # If the user provided answer isn't a number > 0 and < ANSWER_COUNT,
        # return an error message to the user. Remember to guide the user
        # into providing correct values.
        reprompt = session['attributes']['speech_output']
        reprompt_text = session['attributes']['reprompt_text']
        speech_output = "Your answer must be a known sustainability subject. " + reprompt
        return build_response(
            session['attributes'],
            build_speechlet_response(
                SKILL_NAME, speech_output, reprompt_text, should_end_session
            ))
    else:
        texts, twitter_texts, texts1 = populate_tweet_topics(Answer)
        print(len(texts))
        dictionary = corpora.Dictionary(texts)

        # print("Here we go")
        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpus1 = [dictionary.doc2bow(text) for text in texts1]
        # print("again on our own")
        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)
        print(ldamodel.log_perplexity(corpus1))
        print(ldamodel.print_topics(num_topics=10, num_words=5))

        bmlda = ldamodel[corpus]
        largest = {}
        topics = []
        tweet = []
        ind = 0

        for index, t in enumerate(texts):
            z = ldamodel[dictionary.doc2bow(texts[index])]
            for i in z:
                if i[0] in largest:
                    if i[1] > largest[i[0]][0]:
                        largest[i[0]] = (i[1], index)
                else:
                    largest[i[0]] = (i[1], index)

        for i in range(10):
            tweet.append(twitter_texts[largest[i][1]])

        for i in range(10):
            well_crap = ldamodel.get_topic_terms(i, 5)
            topic_holders = ''
            for t in range(5):
                topic_holders = topic_holders + dictionary[well_crap[t][0]] + ", "
            topics.append(topic_holders)

        speech_output = "A topic was " + topics[
            ind] + " would you like to hear the top tweet from this topic? You may also say next to continue to next topic"
        reprompt_text = "Would you like to hear the tweet or continue?"

        session.attributes['speech_output'] = speech_output
        session.attributes['reprompt_text'] = reprompt_text
        session.attributes['current_topic_index'] = ind
        session.attributes['topics'] = topics
        session.attributes['tweets'] = tweet

        return question(speech_output).reprompt(reprompt_text)


def populate_tweet_topics(answer):
    twitterText = []
    print(answer)
    try:
        tso = TwitterSearchOrder()
        tso.add_keyword(answer)

        tso.set_geocode(38.328732, -85.764771, 35)

        ts = TwitterSearch(
            consumer_key='cwWYDhxAOWmysLSKZ15Dwuo1o',
            consumer_secret='5z3gFNKfPXEwiQSuv0gJt0npvpDfjS76DH5yt4s3vR76xeOrkJ',
            access_token='856904167687958528-RpiKjJOPOo0zqN8DGqYwBYW3fIdmCG7',
            access_token_secret='2EONjng5sFWuYGyLp7XiBKJcgPPHADdSZ2ZRM03JWNExH'
        )

        for tweets in ts.search_tweets_iterable(tso):
            twitterText.append(tweets['text'])
    except TwitterSearchException as e:
        print(e)

    p = int(len(twitterText) * .8)
    cp_train = twitterText[0:p]
    cp_test = twitterText[p:]

    documents = cp_train
    documents = [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z# \t])|(\w+:\/\/\S+)", " ", doc.strip()).split()) for doc
                 in documents]
    documents = [' '.join([w for w in doc.split() if not w.isdigit() and len(w) > 1]) for doc in documents]
    print('Done Reading Data')
    documents1 = cp_test
    documents1 = [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z# \t])|(\w+:\/\/\S+)", " ", doc.strip()).split()) for doc
                  in documents]
    documents1 = [' '.join([w for w in doc.split() if not w.isdigit() and len(w) > 1]) for doc in documents]

    # Remove stopwords
    cached_stopwords = set(stopwords.words('english'))
    cached_stopwords.update(
        [word for line in open('stopwords.txt', 'r') for word in line.split()])  # read from stopwords file
    cached_stopwords.update(
        ['rt', 'https', 'http', 'htt', 'gt', 'p', 'amp', 'a', 'about', 'above', 'above', 'across', 'after',
         'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always',
         'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything',
         'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes',
         'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between',
         'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could',
         'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'dont', 'down', 'due', 'during', 'each', 'eg',
         'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every',
         'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first',
         'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get',
         'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein',
         'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc',
         'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least',
         'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most',
         'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next',
         'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on',
         'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over',
         'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming',
         'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so',
         'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system',
         'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter',
         'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this', 'those',
         'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward',
         'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we',
         'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby',
         'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom',
         'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself',
         'yourselves', 'the'])

    texts = [[word for word in document.lower().split() if word not in cached_stopwords] for document in documents]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in
             texts]  # keep only the words that occured atleast twice in the dataset
    print("Got here as well")

    texts1 = [[word for word in document.lower().split() if word not in cached_stopwords] for document in documents1]
    frequency = defaultdict(int)
    for text in texts1:
        for token in text:
            frequency[token] += 1
    texts1 = [[token for token in text if frequency[token] > 1] for text in
              texts1]  # keep only the words that occured atleast twice in the dataset

    return texts, twitterText, texts1


@ask.intent("NextIntent")
def handle_next():
    if 'current_topic_index' not in session['attributes'].keys():
        speech_output = "Please ask about a subject first"
        reprompt_text = "Say a subject"
        session.attributes["speech_output"] = speech_output
        session.attributes["reprompt_text"] = reprompt_text
        return question(speech_output).reprompt(reprompt_text)
    else:
        index = session['attributes']['current_topic_index']
        tweets = session['attributes']['tweets']
        topics = session['attributes']['topics']
        index += 1
        if index == 10:
            speech_output = "A topic was " + topics[index] + " would you like to hear the top tweet from this topic?"
        elif index == 11:
            return statement("All topics finished. Goodbye!")
        else:
            speech_output = "A topic was " + topics[
                index] + " would you like to hear the top tweet from this topic? You may also say next to continue to next topic"
        reprompt_text = "Say next to continue to next topic"
        session.attributes['speech_output'] = speech_output
        session.attributes['reprompt_text'] = reprompt_text
        session.attributes['current_topic_index'] = index
        session.attributes['topics'] = topics
        session.attributes['tweets'] = tweets

        return question(speech_output).reprompt(reprompt_text)


@ask.intent("AMAZON.YesIntent")
def handle_yes():
    if 'current_topic_index' not in session['attributes'].keys():
        speech_output = "Please ask about a subject first"
        reprompt_text = "Say a subject"
        session.attributes["speech_output"] = speech_output
        session.attributes["reprompt_text"] = reprompt_text

        return question(speech_output).reprompt(reprompt_text)
    else:
        ind = session['attributes']['current_topic_index']
        tweets = session['attributes']['tweets']
        topics = session['attributes']['topics']
        if ind == 10:
            speech_output = "The tweet was " + tweets[ind] + " . Goodbye!"
            return statement(speech_output)
        else:
            speech_output = "The tweet was " + tweets[ind] + ". You may also say next to continue to next topic"
        reprompt_text = "Say next to continue to next topic"
        session.attributes['speech_output'] = speech_output
        session.attributes['reprompt_text'] = reprompt_text
        session.attributes['current_topic_index'] = ind
        session.attributes['topics'] = topics
        session.attributes['tweets'] = tweets

        return question(speech_output).reprompt(reprompt_text)


@ask.intent("AMAZON.NoIntent")
def handle_no():
    if 'current_topic_index' not in session['attributes'].keys():
        speech_output = "Please ask about a subject first"
        reprompt_text = "Say a subject"
        session.attributes["speech_output"] = speech_output
        session.attributes["reprompt_text"] = reprompt_text
        return question(speech_output).reprompt(reprompt_text)
    else:
        index = session['attributes']['current_topic_index']
        tweets = session['attributes']['tweets']
        topics = session['attributes']['topics']
        index += 1
        if index == 10:
            speech_output = "A topic was " + topics[index] + " would you like to hear the top tweet from this topic?"
        elif index == 11:
            return statement("All topics finished. Goodbye!")
        else:
            speech_output = "A topic was " + topics[
                index] + " would you like to hear the top tweet from this topic? You may also say next to continue to next topic"
        reprompt_text = "Say next to continue to next topic"
        session.attributes['speech_output'] = speech_output
        session.attributes['reprompt_text'] = reprompt_text
        session.attributes['current_topic_index'] = index
        session.attributes['topics'] = topics
        session.attributes['tweets'] = tweets

        return question(speech_output).reprompt(reprompt_text)


@ask.intent("AMAZON.RepeatIntent")
def handle_repeats():
    if 'attributes' not in session or 'speech_output' not in session['attributes']:
        return get_welcome_response()
    else:
        speech_output = session.attributes["speech_output"]
        reprompt_text = session.attributes["reprompt_text"]
        session.attributes["speech_output"] = speech_output
        session.attributes["reprompt_text"] = reprompt_text
        return question(speech_output).reprompt(reprompt_text)


@ask.intent("AMAZON.StopIntent")
def handle_stop():
    return statement("Goodbye!")


def build_speechlet_response(title, output, reprompt_text, should_end_session):
    return {
        'outputSpeech': {
            'type': 'PlainText',
            'text': output
        },
        'card': {
            'type': 'Simple',
            'title': title,
            'content': output
        },
        'reprompt': {
            'outputSpeech': {
                'type': 'PlainText',
                'text': reprompt_text
            }
        },
        'shouldEndSession': should_end_session
    }


def build_speechlet_response_without_card(output, reprompt_text, should_end_session):
    return {
        'outputSpeech': {
            'type': 'PlainText',
            'text': output
        },
        'reprompt': {
            'outputSpeech': {
                'type': 'PlainText',
                'text': reprompt_text
            }
        },
        'shouldEndSession': should_end_session
    }


def build_response(attributes, speechlet_response):
    return {
        'version': '1.0',
        'sessionAttributes': attributes,
        'response': speechlet_response
    }


if __name__ == '__main__':
    app.run(debug=False)