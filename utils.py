import re

def replace_tag(series):
  return series.apply(lambda x: re.sub(r'@([A-Za-z0-9_]+)','xxxname ', x))

def replace_url(series):
  return series.apply(lambda x: re.sub(r'(http://|https://)([A-Za-z0-9_./]+)','xxxurl ', x))

def clean(series):
  x = replace_tag(series)
  x = replace_url(x)
  x = x.str.lower()
  return x