import abc
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
from six import with_metaclass
from tqdm import tqdm


from magis_sigdial2020.utils.colorlib import Color
from magis_sigdial2020.utils.serializers import JSONSerializable


class ProtoObject(with_metaclass(abc.ABCMeta)):
    ''' this is specific to this dataset '''
    
    @classmethod
    @abc.abstractmethod
    def from_row(cls, row):
        ''' process row from dataset '''

    @abc.abstractmethod
    def __str__(self):
        ''' this must be implemented for all objects '''
        
    def __repr__(self):
        return str(self)
        
class ProtoDatum(with_metaclass(abc.ABCMeta)):
    @abc.abstractmethod
    def get_data(self):
        ''' common method for returning data '''
        

class Subject(ProtoObject, JSONSerializable):
    def __init__(self, worker_id, role):
        self.worker_id = worker_id
        self.role = role
        
    @classmethod
    def from_row(cls, row):
        return cls(row.workerid_uniq, row.role)
    
    def __hash__(self):
        return hash(self.worker_id)
    
    def __str__(self):
        return "({}) {}".format(self.role, self.worker_id)
        
    def get_serializable_contents(self):
        return {'worker_id': self.worker_id, 'role': self.role}
    
    @classmethod
    def deserialize_from_contents(cls, contents):
        return cls(**contents)
    
class Stimuli(JSONSerializable, ProtoDatum):
    def __init__(self, color, clicked, is_target, loc, object_type):
        self.color = color
        self.clicked = clicked
        self.is_target = is_target
        self.loc = loc
        self.object_type = object_type
    
    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1,1)
        r = mpl.patches.Rectangle((0,0), 100,100, color=self.color.rgb)
        ax.add_patch(r)
        ax.set_xlim(0,100)
        ax.set_ylim(0,100)
        ax.axis('off')
        return ax
        
    def get_serializable_contents(self):
        return {'color': tuple(map(float,self.color.hsl)), 
                'clicked': self.clicked,
                'is_target': self.is_target,
                'loc': tuple(map(int,self.loc)),
                'object_type' : self.object_type}
                
    def get_data(self):
        return self.color.hsv
        
    @classmethod
    def deserialize_from_contents(cls, contents):
        contents['color'] = Color(hsl=contents['color'])
        return cls(**contents)
        
    
class StimuliSet(ProtoObject, ProtoDatum, JSONSerializable):
    def __init__(self,  clicked, alt1, alt2, target_index):
        self.clicked = clicked
        self.alt1 = alt1
        self.alt2 = alt2
        self.target_index = target_index
        
    def plot(self, message='', axes=None):
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(6,2))
        self.clicked.plot(axes[0])
        axes[0].text(33, 25, '[C]', size=20)
        self.alt1.plot(axes[1])
        self.alt2.plot(axes[2])
        axes[self.target_index].text(33,75,'[T]', size=20)
        axes[1].set_title(message, size=14)
        
    @classmethod
    def from_row(cls, row):

        clicked = Stimuli(color=Color(hsl=(row.clickColH/360., row.clickColS/100., row.clickColL/100.)),
                          clicked=True,
                          is_target=row.clickStatus=='target',
                          loc=(row.clickLocS, row.clickLocL),
                          object_type=row.clickStatus)
        alt1 =  Stimuli(color=Color(hsl=(row.alt1ColH/360., row.alt1ColS/100., row.alt1ColL/100.)),
                       clicked=True,
                       is_target=row.alt1Status=='target',
                       loc=(row.alt1LocS, row.alt1LocL),
                       object_type=row.alt1Status)
        alt2 = Stimuli(color=Color(hsl=(row.alt2ColH/360., row.alt2ColS/100., row.alt2ColL/100.)),
                       clicked=True,
                       is_target=row.alt2Status=='target',
                       loc=(row.alt2LocS, row.alt2LocL),
                       object_type=row.alt2Status)

        if clicked.is_target:
            target_index = 0
        elif alt1.is_target:
            target_index = 1
        else:
            target_index = 2
            
        return cls(clicked, alt1, alt2, target_index)
        
    def __str__(self):
        return "StimuliSet({},{},{})".format(self.clicked.color.web, self.alt1.color.web, self.alt2.color.web)
        
    def get_serializable_contents(self):
        return {'clicked': self.clicked.get_serializable_contents(), 
                'alt1': self.alt1.get_serializable_contents(),
                'alt2': self.alt2.get_serializable_contents(),
                'target_index': self.target_index}
            
    @classmethod
    def deserialize_from_contents(cls, contents):
        for key in ['clicked', 'alt1', 'alt2']:
            contents[key] = Stimuli.deserialize_from_contents(contents[key])        
        return cls(**contents)
        
    def get_data(self):
        stim_data = (self.clicked.get_data(), 
                       self.alt1.get_data(), 
                       self.alt2.get_data())
        ordered = (stim_data[self.target_index],)
        ordered += stim_data[:self.target_index]
        ordered += stim_data[self.target_index+1:]
        
        return self.target_index, ordered
        

class UtteranceEvent(ProtoObject,ProtoDatum,JSONSerializable):
    """
    Attributes:
        text (str): the text of the utterance
        subject (Subject): object representing person for event 
        role (str): 'speaker' or 'listener'
            Note: 'speaker' is more like director than 'speaker'
        person_type (str): 'human'
        timestamp (float)
    """
    def __init__(self, text, subject, role, person_type, timestamp):
        self.text = text
        self.subject = subject
        self.role = role
        self.person_type = person_type
        self.timestamp = timestamp
        
    @classmethod
    def from_row(cls, row):
        return cls(text=row['contents'],
                        subject=Subject.from_row(row),
                        role=row['role'],
                        person_type=row['source'],
                        timestamp=row['msgTime'])
        
    def __str__(self):
        return "{}: {}".format(self.subject, self.text)
    
    def get_serializable_contents(self):
        return {'text': self.text,
                'subject': self.subject.get_serializable_contents(), 
                'role': self.role,
                'person_type': self.person_type, 
                'timestamp': self.timestamp}
        
    @classmethod
    def deserialize_from_contents(cls, contents):
        contents['subject'] = Subject.deserialize_from_contents(contents['subject'])
        return cls(**contents)
        
    def get_data(self):
        return (self.subject, self.text)
    

class Round(ProtoObject,ProtoDatum,JSONSerializable):
    @staticmethod
    def compute_key(row):
        return "{}+{}".format(row.gameid, int(row.roundNum))
    
    def __init__(self, gameid, round_num, stimuli, key, condition,
                       utterance_events=None):
                           
        self.gameid = gameid
        self.round_num = round_num
        self.stimuli = stimuli
        self.key = key
        self.condition = condition
        
        if utterance_events is None:
            utterance_events = []
        self.utterance_events = utterance_events
        
    def get_data(self):
        utterances = []
        for utterance_event in self.utterance_events:
            utterances.append(utterance_event.get_data())
        target, stims = self.stimuli.get_data()
        return {"condition": self.condition, 
                "target": target, 
                "stims": stims, 
                "utterance": utterances}
        
    def __hash__(self):
        return hash(self.key)
    
    def add_utterance_event(self, row):
        self.utterance_events.append(UtteranceEvent.from_row(row))
    
    def plot(self, axes=None):
        texts = "round: {}\n{}".format(self.round_num, 
                                       "\n".join("{}:{}".format(u.role,u.text) 
                                                  for u in self.utterance_events))
        self.stimuli.plot(texts, axes)
    
    @classmethod
    def from_row(cls, row):
        stimuli = StimuliSet.from_row(row)
        out = cls(row.gameid, row.roundNum, stimuli, Round.compute_key(row), row.condition)
        out.add_utterance_event(row)
        return out
    
    def __str__(self):
        return "[round::{}] {} utterance events".format(self.key, len(self.utterance_events))
        
    def get_serializable_contents(self):
        return {'gameid': self.gameid,
                'round_num': int(self.round_num),
                'stimuli': self.stimuli.get_serializable_contents(),
                'key': self.key,
                'utterance_events': [ue.get_serializable_contents() 
                                     for ue in self.utterance_events],
                'condition': self.condition}

        
    @classmethod
    def deserialize_from_contents(cls, contents):
        contents['utterance_events'] = [UtteranceEvent.deserialize_from_contents(ue)
                                        for ue in contents['utterance_events']]
        contents['stimuli'] = StimuliSet.deserialize_from_contents(contents['stimuli'])
        
        return cls(**contents)
        
    def simple_transcript(self):
        print("{} [{}].".format(self.round_num, self.condition))
        for ue in self.utterance_events:
            print("\t{:<10}".format(ue.role+':'), ue.text)
    
        
class Game(JSONSerializable,ProtoDatum):
    def __init__(self, gameid, rounds=None):
        self.gameid = gameid
        if rounds is None:
            rounds = OrderedDict()
        else:
            assert isinstance(rounds, OrderedDict), 'expected an ordered dict'
        self.rounds = rounds
        self.rounds_list = list(rounds.values())

    def add_round(self, row):
        round_key = Round.compute_key(row)
        if round_key not in self.rounds:
            round_ = Round.from_row(row)
            self.rounds[round_key] = round_
            self.rounds_list.append(round_)
        else:
            self.rounds[round_key].add_utterance_event(row)
            
    def get_data(self):
        rounds = sorted(self.rounds.values(), key=lambda r:r.round_num)
        return [r.get_data() for r in rounds]
        
    def get_round(self, i, count_from_zero=False):
        if not count_from_zero:
            i = i - 1
        return list(self.rounds.values())[i]
        
    def __str__(self):
        return "Game({})".format(self.gameid)
    
    def __repr__(self):
        return str(self)
    
    def get_serializable_contents(self):
        return {'gameid': self.gameid, 
                'rounds': {key: r.get_serializable_contents() 
                            for key, r in self.rounds.items()}}
    
    @classmethod
    def deserialize_from_contents(cls, contents):
        contents['rounds'] = {k:Round.deserialize_from_contents(r) for 
                              k, r in contents['rounds'].items()}
        contents['rounds'] = OrderedDict(sorted(contents['rounds'].items(), 
                                                key=lambda x:x[1].round_num))
        return cls(**contents)
        
    def simple_transcript(self):
        for round_ in sorted(self.rounds.values(), key=lambda x: x.round_num):
            round_.simple_transcript()
            
    def plotted_transcript(self, single_fig=False, fig_kws=None):
        if single_fig:
            fig_kws = fig_kws or {}
            _, axes = plt.subplots(len(self.rounds), 3, **fig_kws)
            
        for idx, round_ in enumerate(sorted(self.rounds.values(), key=lambda x: x.round_num)):
            if single_fig:
                ax = axes[idx]
            else:
                ax = None
            round_.plot(axes=ax)
            
        
            
            
class Dataset(JSONSerializable,ProtoDatum):
    '''
    A wrapper around a dictionary of games, indexed by gameid.
    Each game has 50 rounds.
    Each round has at least 1 utterance and a set of color stimuli.
    
    Args:
        :dict game_dictionary:         {gameid: Game object, ...}
        
        :dict gameid2index:            {gameid: index, ...}
            The dataset comes ordered, and ordering matters. 
            
        :dict index2gameid:            {index: gameid, ...}
            The dataset comes ordered, and ordering matters.                         
    '''
    def __init__(self, game_dictionary, gameid2index, index2gameid, name=''):
        self.games = game_dictionary
        self.name = name
        
        self.game2index = gameid2index
        self._index2game = index2gameid
        self._current_iter_index = 0
        
        
    @classmethod
    def from_dataframe(cls, df, name=''):
        g2i = {}
        i2g = {}
        games = {}
        
        if 'gameid' not in df.columns:
            df['gameid'] = df.index
        
        for row_index in tqdm(range(len(df)), total=len(df)):
            row = df.iloc[row_index]
            if row.gameid not in games:
                games[row.gameid] = Game(row.gameid)
            
            games[row.gameid].add_round(row)
            g2i[row.gameid] = row_index
            i2g[row_index] = row.gameid
        
        return cls(games, g2i, i2g, name)
        
    def get_serializable_contents(self):
        return {'game_dictionary': {gameid:game.get_serializable_contents() 
                                    for gameid, game in self.games.items()},
                'gameid2index': self.game2index,
                'index2gameid': self._index2game,
                'name': self.name}
                
    def get_game(self, game_index):
        """Get the game for the corresponding game index

        Args:
            game_index (int): integer index to the list of games
        Returns:
            Game: the game corresponding to game_index in this dataset
        """
        return self.games[self._index2game[game_index]]
    
    @classmethod            
    def deserialize_from_contents(cls, contents):
        contents['game_dictionary'] = {
            gameid:Game.deserialize_from_contents(game) 
            for gameid, game in contents['game_dictionary'].items()
            }
        contents['gameid2index'] = {v:int(k) 
                                    for v,k in contents['gameid2index'].items()}
                                    
        contents['index2gameid'] = {int(k):v
                                    for k,v in contents['index2gameid'].items()}
                                    
        return cls(**contents)
                
    def __iter__(self):
        self._current_iter_index = 0
        return self

    def __next__(self):
        if self._current_iter_index >= len(self._index2game):
            raise StopIteration
            
        out = self.games[self._index2game[self._current_iter_index]]
        self._current_iter_index += 1
        return out

    def __str__(self):
        return "Dataset({}); ({} games)".format(self.name, len(self._index2game))
        
    def __repr__(self):
        return str(self)
    
    def get_data(self):
        return [g.get_data() for g in self.games.values()]
    
    next = __next__  # Python 2



class ColorsInContext:
    def __init__(self, train, dev, test):
        self.train = train
        self.dev = dev
        self.test = test
    
    @classmethod
    def load_from_premade(cls, train_file, dev_file, test_file):
        return cls(Dataset.deserialize_from_file(train_file),
                   Dataset.deserialize_from_file(dev_file),
                   Dataset.deserialize_from_file(test_file))
                   
    @classmethod
    def make_from_csv(cls, csv_file, save_dataframes=False, train_file='train.json',
                      dev_file='dev.json', test_file='test.json'):

        # from Monroe et al
        train_games = {'start':'1124-1', 'end':'8994-5'}
        dev_games = {'start':'2641-2', 'end': '8574-6'}
        test_games = {'start': '5913-4', 'end': '8452-5'}
        
            
        df = pd.read_csv(csv_file)
            
        index2gameid = {i:df.iloc[i].gameid for i in range(len(df))}
        gameid2indices = {}
        for i, gameid in index2gameid.items():
            gameid2indices.setdefault(gameid,[]).append(i)

        def format_indices(gameid2indices, gameids):
            return {'start': min(gameid2indices[gameids['start']]), 
                    'end': max(gameid2indices[gameids['end']])}
        
        train_indices = format_indices(gameid2indices, train_games)
        dev_indices = format_indices(gameid2indices, dev_games)
        test_indices = format_indices(gameid2indices, test_games)
                
        train_df = df.iloc[train_indices['start']:train_indices['end']+1]
        dev_df = df.iloc[dev_indices['start']:dev_indices['end']+1]
        test_df = df.iloc[test_indices['start']:test_indices['end']+1]
        
        train_data = Dataset.from_dataframe(train_df, 'train')
        dev_data = Dataset.from_dataframe(dev_df, 'dev')
        test_data = Dataset.from_dataframe(test_df, 'test')
        
        if save_dataframes:
            train_data.serialize_to_file(train_file)
            dev_data.serialize_to_file(dev_file)
            test_data.serialize_to_file(test_file)
        
        return cls(train_data, dev_data, test_data)