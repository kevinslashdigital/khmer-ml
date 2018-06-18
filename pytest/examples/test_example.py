import sys, os
import pytest

syspath = 'slash-ml'
sys.path.append(os.path.abspath(os.path.join('..', syspath)))

class TestExample:
    def test_ehlo(self, smtp):
        response, msg = smtp.ehlo()
        assert response == 250
        assert 1 # for demo purposes
