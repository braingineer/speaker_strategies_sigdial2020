# coding=utf-8
import abc
import json

import six


@six.add_metaclass(abc.ABCMeta)
class Serializer(object):
    """
    """

    @staticmethod
    @abc.abstractmethod
    def save(contents, file_path):
        """ """

    @staticmethod
    @abc.abstractmethod
    def load(file_path):
        """ """


class JSONSerializer(Serializer):
    @staticmethod
    def save(contents, file_path):
        with open(file_path, "w") as file_pointer:
            json.dump(contents, file_pointer)

    @staticmethod
    def load(file_path):
        with open(file_path, "r") as file_pointer:
            return json.load(file_pointer)


################################################################################


@six.add_metaclass(abc.ABCMeta)
class Serializable(object):
    def set_serializer(self, serializer):
        self._serializer = serializer

    @abc.abstractmethod
    def get_serializable_contents(self):
        """
        return an object that can be serialized and
            fully specifies the instance

        Best Practice:
            return a dictionary where each key corresponds exactly to an
            argument in the __init__ function signature
        """

    @classmethod
    @abc.abstractmethod
    def deserialize_from_contents(cls, contents, **kwargs):
        """
        deserialize using the contents. Subclasses should also have **kwargs
            in the function signature.

        Best Practice:
            use this method as the single deserialization implementation and
            have all other methods point here.
        """

    def serialize_to_file(self, file_path, serializer=None):
        """
        given a file_path, open the file and save in the preferred method

        Best Practice:
            use `self.get_serializable_contents()` to get the contents being saved
        """
        if serializer is None and not hasattr(self, "_serializer"):
            raise RuntimeError("serializer is None and _serializer not set for {}".format(self))
        elif serializer is None:
            serializer = self._serializer
        serializer.save(self.get_serializable_contents(), file_path)

    @classmethod
    def deserialize_from_file(cls, file_path, serializer=None):
        """
        this method should accept as input the string location
            of a file, open it, load its contents, and return the
            instantiated class.

        Best Practice:
            Deserialize the contents of the file using the
            `cls.deserialize_from_contents` method.
        """
        if serializer is None and not hasattr(cls, "_serializer"):
            raise RuntimeError("serializer is None and _serializer not set for {}".format(cls))
        elif serializer is None:
            serializer = cls._serializer

        out = cls.deserialize_from_contents(serializer.load(file_path))
        out.set_serializer(serializer)
        return out


########## Implementations #####################

class JSONSerializable(Serializable):
    """
    Subclasses still need to be implement `get_serializable_contents`
        and `deserialize_from_contents`.
    """

    @classmethod
    def deserialize_from_contents(cls, contents):
        pass

    def get_serializable_contents(self):
        pass

    _serializer = JSONSerializer


_serializers = {"local": JSONSerializer}


def get_serializer(serializer_name):
    if serializer_name in _serializers:
        return _serializers[serializer_name]
    elif serializer_name.upper() in _serializers:
        return _serializers[serializer_name.lower()]
    else:
        s_names = "; ".join(_serializers.keys())
        raise Exception("the serializer {} does not exist;".format(serializer_name)
                        + " available options: {}".format(s_names))