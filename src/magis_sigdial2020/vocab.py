import numpy as np

class Vocabulary(object):
    """Class to process text and
    extract vocabulary for mapping"""

    def __init__(self, token_to_idx=None, use_unk=False, unk_token="<UNK>", use_mask=False, mask_token="<MASK>", 
                 use_start_end=False, start_token="<START>", end_token="<END>"):
        self.use_mask = use_mask
        self.use_unk = use_unk
        self.use_start_end = use_start_end
        self.mask_index = -1
        self.mask_token = mask_token
        self.unk_index = -1
        self.unk_token = unk_token
        self.start_index = -1
        self.start_token = start_token
        self.end_index = -1
        self.end_token = end_token

        if token_to_idx is None:
            self._token_to_idx = {}
            self._idx_to_token = {}
        else:
            self._token_to_idx = token_to_idx
            self._idx_to_token = {idx: token 
                                  for token, idx in self._token_to_idx.items()}
            
        # if starting with new token_to_idx or loaded one previously, this will 
        # either retrieve the index or add it
        # TODO: if token_to_idx already had a special token but with a different string, this will silently break
        # I'm not handling this for now. 
        if self.use_mask:
            self.mask_index = self.add_token(self.mask_token)
        if self.use_unk:
            self.unk_index = self.add_token(self.unk_token)
        if self.use_start_end:
            self.start_index = self.add_token(self.start_token)
            self.end_index = self.add_token(self.end_token)
            

    def add_token(self, token):
        """Update mapping dicts based on the token.

        Args: 
            token (str): the string token to add to the vocabulary
        Returns:
            int: the index for the corresponding token
        """
        if token not in self._token_to_idx:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        else:
            index = self._token_to_idx[token]
        return index
    
    def map(self, tokens, as_numpy=False, wrap_start_end=False, add_tokens=True):
        if add_tokens:
            indices = [self.add_token(token) for token in tokens]
        else:
            indices = [self.lookup_token(token) for token in tokens]
        if wrap_start_end and not self.use_start_end:
            raise Exception("conflicting usage: `self.use_start_end` is False, `wrap_start` is true")
        if wrap_start_end:
            indices = [self.start_index] + indices + [self.end_index]
        if as_numpy:
            indices = np.array(indices).astype(np.int64)
        return indices
    
    def items(self):
        return list(self._token_to_idx.items())
    
    def keys(self):
        return list(self._token_to_idx.keys())

    def lookup_token(self, token):
        if token not in self._token_to_idx:
            raise KeyError(f"{token} is Out Of Vocabulary (and implicit adds not supported)")
        return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token corresponding to the index

        Args:
            index (int): an index in this Vocabulary's bijection
        Returns:
            str: the token corresponding to the token 
        Raises:
            KeyError: if the index is not in this Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __getitem__(self, token):
        """Return the index for the corresponding token if in the Vocabulary
        
        Returns:
            int: the integer index for the token
        Raises:
            KeyError: if token not in Vocabulary
        """
        return self.lookup_token(token)

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._token_to_idx)

    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)
    
    def to_serializable(self):
        raise NotImplemented