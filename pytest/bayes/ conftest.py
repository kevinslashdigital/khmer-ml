import pytest

@pytest.hookimpl
def pytest_addoption(parser):
  parser.addoption("--stringvalue", action="append", default=[], help="list of stringinputs to pass to test functions")

def pytest_generate_tests(metafunc):
  if 'stringinput' in metafunc.fixturenames:
    metafunc.parametrize("stringinput", metafunc.config.getoption('stringinput'))
