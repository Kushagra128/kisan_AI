from django.templatetags.static import static
from django.urls import reverse
from jinja2 import Environment

def url_for(endpoint, **kwargs):
    if endpoint == 'static':
        return static(kwargs.get('filename'))
    return reverse(endpoint, kwargs=kwargs)

def environment(**options):
    env = Environment(**options)
    env.globals.update({
        'static': static,
        'url': reverse,
        'url_for': url_for,
    })
    return env
