
import logging

class Log(object):

  def __init__(self, *args, **kwargs):
    self.kwargs = kwargs
    self.filename = args[0]

  def log(self,data):
    logging.basicConfig(filename= self.filename,level=logging.DEBUG)
    logging.debug(data)


