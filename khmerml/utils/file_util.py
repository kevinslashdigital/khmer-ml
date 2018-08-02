"""
"""
import csv
import os
import pickle
import zipfile
import shutil

from numpy import genfromtxt

class FileUtil(object):
  """
    File Utilities for read and write file
  """

  @staticmethod
  def read_file(filename, file_extension):
    """
      Read file
      return: dataset as list
    """
    _data_matrix = []
    if file_extension == 'CSV':
      _data_matrix = FileUtil.load_csv(filename)

    return _data_matrix

  @staticmethod
  def load_csv(filename, use_numpy=True):
    """
      Read data from csv file using python or numpy lib
    """
    if not use_numpy:
      return FileUtil.load_csv_py(filename)
    else:
      return FileUtil.load_csv_np(filename)

  @staticmethod
  def load_csv_py(filename):
    """
      Read data from csv file.
    """
    file_obj = open(filename, "rU")
    file_csv = csv.reader(file_obj)
    _dataset = list(file_csv)
    for index, subset in enumerate(_dataset):
      _dataset[index] = [float(x) for x in subset]
    file_obj.close()
    return _dataset

  @staticmethod
  def load_csv_np(filename):
    """
      Load data from a text file
      When spaces are used as delimiters, or when no delimiter has been given as input\
      , there should not be any missing data between two fields.
      When the variables are named (either by a flexible dtype or with names\
      , there must not be any header in the file (else a ValueError exception is raised).
      Individual values are not stripped of spaces by default. When using a custom converter
      , make sure the function does remove spaces
    """
    # Data read from the text file. If usemask is True, this is a masked array.
    _dataset = genfromtxt(filename, delimiter=',')

    return _dataset

  @staticmethod
  def dataset_path(config, filename):
    """
      path to file
    """
    path = os.path.join(os.getcwd(), config['dataset'], filename)
    return path

  @staticmethod
  def path_to_file(dirname, filename):
    """
      path to file
    """

    path = os.path.join(os.getcwd(), dirname, filename)
    return path

  @staticmethod
  def join_path(dirname):
    """
      path to file
    """
    path = os.path.join(os.getcwd(), dirname)
    return path

  @staticmethod
  def save_pickle_dataset(pickle_filename, dataset):
    """
      The pickle module implements binary protocols
      for serializing and de-serializing a Python object structure.
    """
    path_to_pickle = os.path.join(os.getcwd(), pickle_filename)
    with open(path_to_pickle, 'wb') as handle:
      pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return True

  @staticmethod
  def save_model(file, model):
    """
      The pickle module implements binary protocols
      for serializing and de-serializing a Python object structure.
    """
    path_to_pickle = os.path.join(os.getcwd(), file)
    try:
      FileUtil.create_folder(FileUtil.get_folder_path(path_to_pickle))
      with open(path_to_pickle, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except pickle.PickleError as error:
      raise Exception(error)
    else:
        return True

  @staticmethod
  def load_model(file):
    """
      Read .pickle file
    """
    path_to_pickle = os.path.join(os.getcwd(), file)
    try:
      with open(path_to_pickle, 'rb') as handle:
        model = pickle.load(handle)
    except pickle.UnpicklingError as error:
      raise Exception(error)
    else:
      return model

  @staticmethod
  def load_pickle(pickle_filename):
    """
      Read .pickle file
    """
    path_to_pickle = os.path.join(os.getcwd(), pickle_filename)
    try:
      with open(path_to_pickle, 'rb') as handle:
        model = pickle.load(handle)
    except pickle.UnpicklingError as error:
      raise Exception(error)
    else:
      return model

  @staticmethod
  def extract_zipfile(path_to_zipefile, dest_path):
    """
      Extract zipfile sent from client
      Store temporarily in server
    """
    try:
      with zipfile.ZipFile(path_to_zipefile) as opened_rar:
        opened_rar.extractall(dest_path)
    except OSError as error:
        raise Exception(error)
    else:
        return True

  @staticmethod
  def move_file(source, destination):
    """
      This function is used to move file from source to destination
    """
    try:
      shutil.move(source, destination)
    except OSError as error:
      print('error %s' % error)
      return False
    return True

  @staticmethod
  def remove_file(path, ignore_errors=False, onerror=None):
    """
      Delete an entire directory tree; path must point to a directory \
      (but not a symbolic link to a directory). If ignore_errors is true, errors \
      resulting from failed removals will be ignored; if false or omitted, \
      such errors are handled by calling a handler specified by onerror or, \
      if that is omitted, they raise an exception.
    """

    ## check if a file exists on disk ##
    ## if exists, delete it else show message on screen ##
    if os.path.exists(path):
      if os.path.isfile(path):
        try:
          os.remove(path)
        except OSError as error:
          print("Error: %s - %s." % (error.filename, error.strerror))
      else:
        try:
          shutil.rmtree(path, ignore_errors, onerror)
        except OSError as error:
          print('error %s' % error)
          return False
    else:
      print("Sorry, I can not find %s file." % path)
      return False
    return True

  @staticmethod
  def _print(*args):
    try:
      # print(args)
      pass
    except UnicodeEncodeError:
      print('cannot print')

  @staticmethod
  def create_folder(folder):
    if not os.path.exists(folder):
      os.makedirs(folder)

  @staticmethod
  def get_folder_path(file):
    return os.path.dirname(file)
