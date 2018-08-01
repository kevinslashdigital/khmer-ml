import os
import subprocess
import time
from .khmersegment import KhmerSegment

class UnicodeSplit:
  def __init__(self):
    pass

  # Process Khmer Segmentation
  def unicode_split(self,text):
    segment = KhmerSegment()
    file_name = segment.add_to_file(text)

    # Run command to convert the sentence into separated words
    command = 'cd ' + segment.PATH + ' && ./km-5tag-seg-test.sh model/km-5tag-seg-model sample/' + file_name + ' sample-out/'
    KhmerSegment.run_command(command)

    input_file = segment.PATH + '/sample/' + file_name
    output_file_w = segment.PATH + '/sample-out/' + file_name + '.w'
    output_file_c = segment.PATH + '/sample-out/' + file_name + '.c'

    # Read temporary file content
    result = KhmerSegment.read_file(output_file_w).split(' ')
    result = list(set(result))
    # Clean up temporary files
    KhmerSegment.remove_file(input_file)
    KhmerSegment.remove_file(output_file_w)
    KhmerSegment.remove_file(output_file_c)

    return result
