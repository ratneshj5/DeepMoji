import json
import re

from deepmoji.tokenizer import RE_URL
from sprinklr_global import language


def pre_process(message):
    if json.loads(message)['language'] == language:
        text = re.sub(RE_URL, '', json.loads(message)['text'])
        return convert_to_unicode(text)
    return ''


def convert_to_unicode(text):
    try:
        return unicode(text)

    except UnicodeDecodeError:
        try:
            return text.decode('utf-8')
        except UnicodeDecodeError:
            return text.decode('utf-16')
