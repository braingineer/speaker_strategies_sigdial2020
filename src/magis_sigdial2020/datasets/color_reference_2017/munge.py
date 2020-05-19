import json
import pandas as pd

LUX_LABELS = set()
LUX_SMALL_SET = set()
LUX_EXPANSION_MAP = {}

DIFFICULTY_LABELS = [
    "single-utterance-full-match",
    "mulitple-utterances-full-match",
    "single-utterance-partial-match",
    "multiple-utterances-partial-match",
    "single-utterance-no-match",
    "multiple-utterances-no-match"
]


def populate_lux_globals(xkcd_vocab):
    global LUX_LABELS, LUX_SMALL_SET, LUX_EXPANSION_MAP
    LUX_LABELS.update(set(xkcd_vocab.keys()))

    for label in list(xkcd_vocab.keys()):
        word_variant1 = label.replace(" ", "-")
        word_variant2 = label.replace("-", " ")

        if word_variant1 != label:
            LUX_LABELS.add(word_variant1)
            LUX_EXPANSION_MAP[word_variant1] = label

        if word_variant2 != label:
            LUX_LABELS.add(word_variant2)
            LUX_EXPANSION_MAP[word_variant2] = label

        for word in label.replace("-", " ").split(" "):
            LUX_SMALL_SET.add(word)

    LUX_LABELS.add('gray')
    LUX_SMALL_SET.add('gray')
    LUX_LABELS.add('fuschia')
    LUX_SMALL_SET.add('fuschia')

    LUX_EXPANSION_MAP['gray'] = 'grey'
    LUX_EXPANSION_MAP['fuschia'] = 'fuchsia'


def classify_utterance(utterances):
    utterance = utterances[0]
    full_text = json.dumps(utterances)
    replacements = [('-', ' '),
                    ('!', ' '),
                    ('?', ' '),
                    ('.', ' '),
                    ('/', ' '),
                    ('\\', ' '),
                    (',', ' '),
                    (';', ' '),
                    (':', ' '),
                    ('gray', 'grey'),
                    ('fuschia', 'fuchsia')]
    text = utterance['text'].lower()
    for r_from, r_to in replacements:
        text = text.replace(r_from, r_to)
    text = text.strip()
    # text = utterance['text'].lower().replace("-", "").replace("!"," !").replace("?", " ?")

    best_label = ''

    if len(utterances) == 1 and text in LUX_LABELS:
        difficulty_level = 0
        best_label = text
    elif len(utterances) > 1 and text in LUX_LABELS:
        difficulty_level = 1
        best_label = text
    else:
        text_set = set(text.split(" "))
        if len(text_set.intersection(LUX_SMALL_SET)) > 0:
            for label in LUX_LABELS:
                if label in text and len(label) > len(best_label):
                    best_label = label
        if len(utterances) == 1 and len(best_label) > 0:
            difficulty_level = 2
        elif len(utterances) > 1 and len(best_label) > 0:
            difficulty_level = 3
        elif len(utterances) == 1:
            difficulty_level = 4
        else:
            difficulty_level = 5

    difficulty_label = DIFFICULTY_LABELS[difficulty_level]

    if best_label in LUX_EXPANSION_MAP:
        best_label = LUX_EXPANSION_MAP[best_label]

    return difficulty_label, best_label, full_text, difficulty_level


def transform_to_yzz(stimuli):
    if stimuli['clicked']['is_target']:
        target = stimuli['clicked']
        success = True
        alt1 = stimuli['alt1']
        alt2 = stimuli['alt2']
        clicked = 'target'
    elif stimuli['alt1']['is_target']:
        target = stimuli['alt1']
        success = False
        alt1 = stimuli['clicked']
        alt2 = stimuli['alt2']
        clicked = 'alt1'
    else:
        target = stimuli['alt2']
        success = False
        alt1 = stimuli['alt1']
        alt2 = stimuli['clicked']
        clicked = 'alt2'
    return success, target, alt1, alt2, clicked


def get_round_info(round_i, split):
    difficulty_rating, lux_label, full_text, difficulty_int = classify_utterance(round_i['utterance_events'])
    stimuli = round_i['stimuli']
    success, target, alt1, alt2, actual_clicked = transform_to_yzz(stimuli)
    utterance_events = json.loads(full_text)
    num_utterances = len(utterance_events)
    utterance_events_pattern = "".join([utterance_event['role'][0].upper() for utterance_event in utterance_events])

    return {'gameid': round_i['gameid'],
            'roundid': round_i['key'],
            'round_num': round_i['round_num'],
            'lux_difficulty_label': difficulty_rating,
            'lux_difficulty_rating': difficulty_int,
            'condition': round_i['condition'],
            'lux_label': lux_label,
            'matcher_succeeded': success,
            'split': split,
            'target': target['color'],
            'alt1': alt1['color'],
            'alt2': alt2['color'],
            'clicked': actual_clicked,
            'full_text': full_text, 
            'utterance_events': utterance_events, 
            'num_utterances': num_utterances, 
            'utterance_events_pattern': utterance_events_pattern}


def get_round_df(data):
    all_rounds = []
    for game in data.train.games.values():
        game_contents = game.get_serializable_contents()
        flat_rounds = [get_round_info(round_i, 'train') for round_i in game_contents['rounds'].values()]
        all_rounds.extend(flat_rounds)

    for game in data.dev.games.values():
        game_contents = game.get_serializable_contents()
        flat_rounds = [get_round_info(round_i, 'dev') for round_i in game_contents['rounds'].values()]
        all_rounds.extend(flat_rounds)

    for game in data.test.games.values():
        game_contents = game.get_serializable_contents()
        flat_rounds = [get_round_info(round_i, 'test') for round_i in game_contents['rounds'].values()]
        all_rounds.extend(flat_rounds)

    return pd.DataFrame(all_rounds)
