import pytest

@pytest.hookimpl
def pytest_addoption(parser):
  print('hello')
  parser.addoption("--stringvalue", action="append", default=[], help="list of stringinputs to pass to test functions")

def pytest_generate_tests(metafunc):
  if 'stringvalue' in metafunc.fixturenames:
    metafunc.parametrize("stringvalue", metafunc.config.getoption('stringvalue'))
